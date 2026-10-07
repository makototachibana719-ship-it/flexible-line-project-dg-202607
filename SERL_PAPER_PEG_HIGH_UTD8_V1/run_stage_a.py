#!/usr/bin/env python3
"""Execute the paused-checkpoint UTD8 causal comparison and adjudicate it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
RUNNER = BRANCH / "run_high_utd8.py"
EVALUATOR = BRANCH / "evaluate.py"
RUN = BRANCH / "runs/stage_a_seed0_attempt2"
CONFIG = yaml.safe_load((BRANCH / "config.yaml").read_text())
TARGETS = tuple(CONFIG["stage_a"]["mandatory_checkpoints"] +
                CONFIG["stage_a"]["conditional_checkpoints"])
UTD1_BEST_POSITION = 0.009441925311298176


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes():
    paths = [
        ROOT / "SERL_PAPER_PEG_BASELINE_V1/run_baseline.py",
        ROOT / "SERL_PAPER_PEG_BASELINE_V1/evaluate.py",
        ROOT / "SERL_PAPER_PEG_BASELINE_V1/config.yaml",
        ROOT / CONFIG["demo"]["path"],
        ROOT / CONFIG["demo"]["manifest"],
        ROOT / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py",
        ROOT / "external/serl/serl_launcher/serl_launcher/utils/launcher.py",
        ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/peg_env_adapter.py",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def run_eval(step: int, episodes: int, label: str, seed_offset: int = 0):
    log = (RUN / f"eval_{step}_{label}.log").open("w")
    try:
        proc = subprocess.run([
            sys.executable, "-u", str(EVALUATOR), "--run", str(RUN),
            "--step", str(step), "--episodes", str(episodes),
            "--seed-offset", str(seed_offset), "--label", label,
        ], stdout=log, stderr=subprocess.STDOUT, timeout=900,
           env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl"))
    finally:
        log.close()
    require(proc.returncode == 0, f"evaluation failed at {step}/{label}")
    return json.loads((RUN / f"eval_{step}_{label}.json").read_text())


def clear_regression(previous, current):
    p, c = previous["aggregate"], current["aggregate"]
    position_bad = (c["median_min_position_error"] >=
                    p["median_min_position_error"] *
                    CONFIG["stage_a"]["clear_regression_position_ratio"])
    return_bad = (c["mean_return"] <= p["mean_return"] -
                  max(abs(p["mean_return"]) *
                      CONFIG["stage_a"]["clear_regression_return_drop_fraction"], 100.0))
    near_bad = c["near_hole_count"] < p["near_hole_count"]
    success_not_better = c["success_count"] <= p["success_count"]
    result = bool(success_not_better and position_bad and (return_bad or near_bad))
    return result, {"position_bad": position_bad, "return_bad": return_bad,
                    "near_hole_bad": near_bad, "success_not_better": success_not_better}


def behavior_improving(previous, current):
    p, c = previous["aggregate"], current["aggregate"]
    checks = {
        "success": c["success_count"] > p["success_count"],
        "near_hole": c["near_hole_count"] > p["near_hole_count"],
        "position_10pct": c["median_min_position_error"] <= p["median_min_position_error"] * .90,
        "return_10pct": c["mean_return"] >= p["mean_return"] + max(abs(p["mean_return"]) * .10, 100.0),
    }
    return bool(any(checks.values())), checks


def matched(primary, confirmation):
    keys = ("success_count", "near_hole_count", "contact_health",
            "median_min_position_error", "median_min_lateral_error",
            "median_min_axial_error", "median_min_orientation_error",
            "mean_return", "max_force", "mean_action_saturation")
    comparisons = {}
    for key in keys:
        left, right = primary["aggregate"][key], confirmation["aggregate"][key]
        comparisons[key] = (left == right if isinstance(left, (bool, int)) else
                            abs(float(left) - float(right)) <= 1e-6)
    return bool(all(comparisons.values())), comparisons


def choose_best(evaluations):
    def rank(item):
        step, result = item
        agg = result["aggregate"]
        return (agg["success_count"], int(agg["contact_health"]),
                agg["near_hole_count"], -agg["median_min_position_error"],
                agg["mean_return"], step)
    return max(evaluations.items(), key=rank)


def main() -> int:
    require(not RUN.exists(), "Stage-A output already exists")
    require(CONFIG["training"]["utd_ratio"] == 8 and
            CONFIG["training"]["critic_actor_ratio"] == 1,
            "causal UTD binding")
    before = protected_hashes()
    subprocess.run([sys.executable, str(RUNNER), "--role", "preflight",
                    "--output", str(RUN), "--updates", "2000"], check=True)
    RUN.mkdir(parents=True, exist_ok=False)
    dump(RUN / "protected_before.json", before)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl",
               WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    processes = []
    streams = []
    error = None
    evaluations = {}
    decisions = []
    final_eval = None
    confirmation = None
    consecutive_regressions = 0
    conditional_authorized = False

    def launch(role):
        stream = (RUN / f"{role}.log").open("w")
        streams.append(stream)
        process = subprocess.Popen([
            sys.executable, "-u", str(RUNNER), "--role", role,
            "--output", str(RUN), "--updates", "2000",
        ], stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        processes.append(process)
        return process

    monitor_stream = (RUN / "gpu_dmon.log").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "dmon", "-d", "1"],
                               stdout=monitor_stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    learner = actor = None
    try:
        learner = launch("learner")
        deadline = time.monotonic() + 300
        while not (RUN / "server_ready.json").exists():
            require(learner.poll() is None, "learner exited before ready")
            require(time.monotonic() < deadline, "learner readiness timeout")
            time.sleep(.2)
        actor = launch("actor")
        previous = None
        for step in TARGETS:
            if step > 1000 and not conditional_authorized:
                break
            ready = RUN / f"checkpoint_{step}.ready.json"
            deadline = time.monotonic() + CONFIG["stage_a"]["process_timeout_seconds"]
            while not ready.exists():
                require(learner.poll() is None, f"learner exited before checkpoint {step}")
                require(actor.poll() is None, f"actor exited before checkpoint {step}")
                require(time.monotonic() < deadline, f"checkpoint {step} timeout")
                time.sleep(1)
            primary = run_eval(step, CONFIG["stage_a"]["canonical_episodes"], "primary")
            evaluations[step] = primary
            regression = False
            regression_checks = {}
            improving = False
            improvement_checks = {}
            if previous is not None:
                regression, regression_checks = clear_regression(previous, primary)
                improving, improvement_checks = behavior_improving(previous, primary)
                consecutive_regressions = consecutive_regressions + 1 if regression else 0

            action, reason = "continue", "mandatory checkpoint sequence"
            agg = primary["aggregate"]
            if agg["success_count"] >= 2 and agg["contact_health"]:
                confirmation = run_eval(step, CONFIG["stage_a"]["confirmation_episodes"],
                                        "matched")
                match_ok, match_checks = matched(primary, confirmation)
                dump(RUN / f"matched_adjudication_{step}.json",
                     {"status": "PASS" if match_ok else "FAIL", "checks": match_checks})
                if match_ok:
                    final_eval = run_eval(step, CONFIG["stage_a"]["final_episodes"], "final5")
                    converged = (final_eval["aggregate"]["success_count"] >= 3 and
                                 final_eval["aggregate"]["contact_health"])
                    action, reason = "stop", ("UTD8 convergence confirmed" if converged else
                                               "5-episode convergence threshold failed")
                else:
                    action, reason = "stop", "matched confirmation failed"
            elif consecutive_regressions >= 2:
                action, reason = "stop", "two consecutive clear behavioral regressions"
            elif step == 1000:
                conditional_authorized = bool(improving)
                if not conditional_authorized:
                    action, reason = "stop", "behavior not still improving at 1000"
                else:
                    reason = "1000 behavior improving; authorize 1500/2000"
            elif step == 2000:
                action, reason = "stop", "Stage-A 2000-update ceiling"

            decision = {"checkpoint": step, "action": action, "reason": reason,
                        "regression": regression, "regression_checks": regression_checks,
                        "consecutive_regressions": consecutive_regressions,
                        "improving": improving, "improvement_checks": improvement_checks,
                        "aggregate": agg, "timestamp_ns": time.time_ns()}
            dump(RUN / f"decision_{step}.json", decision)
            decisions.append(decision)
            previous = primary
            if action == "stop":
                break
        learner.wait(timeout=300)
        actor.wait(timeout=60)
        require(learner.returncode == actor.returncode == 0, "actor/learner clean exit")
    except Exception as exc:
        error = repr(exc)
    finally:
        for process in processes + [monitor]:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for stream in streams + [monitor_stream]:
            stream.close()

    after = protected_hashes()
    dump(RUN / "protected_after.json", after)
    protected_unchanged = before == after
    require(evaluations, f"no checkpoint evaluation: {error}")
    best_step, best = choose_best(evaluations)
    best_agg = best["aggregate"]
    material = bool(best_agg["success_count"] > 0 or
                    best_agg["median_min_position_error"] <= UTD1_BEST_POSITION * .80 or
                    best_agg["near_hole_count"] >= 2)
    converged = bool(final_eval and final_eval["aggregate"]["success_count"] >= 3 and
                     final_eval["aggregate"]["contact_health"])
    if converged:
        comparison = "PASS"
    elif material:
        comparison = "PARTIAL"
    else:
        comparison = "FAIL"
    result = {
        "UTD8_CAUSAL_COMPARISON": comparison,
        "BEST_UTD8_CHECKPOINT": best_step,
        "BEST_UTD8_SUCCESS": f"{best_agg['success_count']}/{best_agg['episodes']}",
        "BEST_UTD8_MIN_POSITION_ERROR": best_agg["median_min_position_error"],
        "HIGH_UTD_PARTIAL_SUPPORT": "YES" if material and not converged else "NO",
        "HIGH_UTD_SUPPORTS_PAPER_HYPOTHESIS": ("YES" if converged else "MIXED" if material else "NO"),
        "PAPER_FIDELITY_REBUILD_REQUIRED": "NO" if converged else "YES",
        "PAPER_FIDELITY_20_DEMO_SHA": "PENDING_STAGE_B" if not converged else "NOT_REQUIRED",
        "PAPER_FIDELITY_RESET_RANDOMIZATION": "PENDING_STAGE_B" if not converged else "NO",
        "PAPER_FIDELITY_RELATIVE_ACTION": "PENDING_STAGE_B" if not converged else "NO",
        "PAPER_FIDELITY_CONVERGED": "NOT_RUN" if converged else "PENDING_STAGE_B",
        "FINAL_CANONICAL_SUCCESS": (f"{final_eval['aggregate']['success_count']}/5"
                                    if final_eval else "NOT_RUN"),
        "CONTACT_HEALTH": ("PASS" if final_eval and final_eval["aggregate"]["contact_health"]
                           else "PASS" if best_agg["contact_health"] else "FAIL"),
        "PURE_SERL_RLPD_AUBO_PEG_CONVERGENCE": "YES" if converged else "NO",
        "NEXT": "PACKAGE_SUCCESSFUL_PEG" if converged else "STAGE_B_PAPER_FIDELITY_REBUILD",
        "UTD_RATIO": 8,
        "CRITIC_ACTOR_RATIO": 1,
        "DEMO_SHA": CONFIG["demo"]["sha256"],
        "PROTECTED_ARTIFACTS_UNCHANGED": protected_unchanged,
        "RESIDUAL_PROCESSES": "NONE" if all(p.poll() is not None for p in processes) else "PRESENT",
        "ERROR": error,
        "DECISIONS": decisions,
    }
    dump(RUN / "STAGE_A_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if error is None and protected_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
