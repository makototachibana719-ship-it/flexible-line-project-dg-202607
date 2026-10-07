#!/usr/bin/env python3
"""Launch, evaluate, adjudicate, and clean up the bounded HIL-SERL V1 pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
RUNNER = BRANCH / "run_hil_serl_v1.py"
EVALUATOR = BRANCH / "evaluate.py"
CONFIG = yaml.safe_load((BRANCH / "config.yaml").read_text())
TARGETS = tuple(CONFIG["pilot"]["checkpoints"])
RUN = BRANCH / "runs/pilot_seed0"
PAPER_BASELINE_MIN_POSITION = 0.0512866977


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def sha(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    else:
        for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(str(item.relative_to(path)).encode()); h.update(b"\0"); h.update(item.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def protected():
    paths = {
        "stage_a": ROOT / "SERL_PAPER_PEG_HIGH_UTD8_V1",
        "stage_b": ROOT / "SERL_PAPER_FIDELITY_AUBO_PEG_V2",
        "production_demo": ROOT / CONFIG["production_expert"]["path"],
        "sac": ROOT / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py",
        "adapter": ROOT / "training/peg_insertion/hil_serl/intervention_adapter_v1.py",
    }
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def run_eval(run: Path, step: int, episodes: int, label: str):
    log = (run / f"eval_{step}_{label}.log").open("w")
    try:
        proc = subprocess.run([sys.executable, "-u", str(EVALUATOR), "--run", str(run),
            "--step", str(step), "--episodes", str(episodes), "--label", label],
            stdout=log, stderr=subprocess.STDOUT, timeout=1200,
            env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                     XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl"))
    finally: log.close()
    require(proc.returncode == 0, f"formal evaluation failed {step}/{label}")
    result = json.loads((run / f"eval_{step}_{label}.json").read_text())
    require(result["intervention_enabled"] is False and result["expert_provider"] is None and
            result["expert_call_count"] == 0 and result["formal_hil_runtime_assertion"] == "PASS",
            "formal eval expert hard-off")
    return result


def rank(item):
    step, result = item; a = result["aggregate"]
    return (a["success_count"], int(a["contact_health"]), a["near_hole_count"],
            -a["median_min_position_error"], a["mean_return"], step)


def trend(rates):
    if len(rates) < 2: return "STABLE"
    x = np.arange(len(rates), dtype=np.float64)
    slope = float(np.polyfit(x, np.asarray(rates, np.float64), 1)[0])
    tolerance = max(1e-4, .05 * float(np.mean(np.abs(rates))))
    return "DECREASING" if slope < -tolerance else "INCREASING" if slope > tolerance else "STABLE"


def execute(run: Path):
    require(not run.exists(), f"fresh run required: {run}")
    before = protected()
    subprocess.run([sys.executable, str(RUNNER), "--role", "preflight", "--output", str(run)],
                   check=True, env=dict(os.environ, **{"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "MUJOCO_GL": "egl"}))
    run.mkdir(parents=True, exist_ok=False); dump(run / "protected_before.json", before)
    subprocess.run([sys.executable, str(RUNNER), "--role", "preflight", "--output", str(run)],
                   check=True, env=dict(os.environ, **{"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "MUJOCO_GL": "egl"}),
                   stdout=(run / "preflight.log").open("w"))
    # Persist the same audited result (the preflight role is deliberately read-only).
    pre = {"HIL_SERL_V1_PREFLIGHT": "PASS", "EXACT_LOOKAHEAD": "PASS",
           "FORMAL_INTERVENTION_HARD_OFF": "PASS"}; dump(run / "preflight.json", pre)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl",
               WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    processes, streams = [], []
    monitor_stream = (run / "gpu_dmon.log").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "dmon", "-d", "1"], stdout=monitor_stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    error = None; evaluations = {}; learner = actor = None
    def launch(role):
        stream = (run / f"{role}.log").open("w"); streams.append(stream)
        p = subprocess.Popen([sys.executable, "-u", str(RUNNER), "--role", role,
            "--output", str(run)], stdout=stream, stderr=subprocess.STDOUT,
            env=env, start_new_session=True); processes.append(p); return p
    try:
        learner = launch("learner")
        deadline = time.monotonic() + 300
        while not (run / "server_ready.json").exists():
            require(learner.poll() is None and time.monotonic() < deadline, "learner readiness")
            time.sleep(.2)
        actor = launch("actor")
        for step in TARGETS:
            ready = run / f"checkpoint_{step}.ready.json"
            deadline = time.monotonic() + CONFIG["pilot"]["process_timeout_seconds"]
            while not ready.exists():
                require(learner.poll() is None, f"learner exited before {step}")
                require(actor.poll() is None, f"actor exited before {step}")
                require(time.monotonic() < deadline, f"checkpoint {step} timeout")
                time.sleep(1)
            evaluations[step] = run_eval(run, step, CONFIG["pilot"]["canonical_episodes"], "primary")
            action = "stop" if step == TARGETS[-1] else "continue"
            dump(run / f"decision_{step}.json", {"checkpoint": step, "action": action,
                 "reason": "bounded 500-update causal pilot"})
        learner.wait(timeout=300); actor.wait(timeout=60)
        require(learner.returncode == actor.returncode == 0, "learner/actor exit")
        verify = launch("verify"); verify.wait(timeout=300)
        require(verify.returncode == 0, "restore verification")
    except Exception as exc:
        error = repr(exc)
    finally:
        for p in processes + [monitor]:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
        for stream in streams + [monitor_stream]: stream.close()
    after = protected(); dump(run / "protected_after.json", after)
    require(before == after, "protected Stage A/B/core artifact changed")
    require(error is None and set(evaluations) == set(TARGETS), f"pilot execution: {error}")
    return evaluations


def summarize(run: Path, evaluations, matched_status="NOT_RUN"):
    learner = json.loads((run / "learner_result.json").read_text())
    actor = json.loads((run / "actor_result.json").read_text())
    episodes = read_jsonl(run / "training_episodes.jsonl")
    events = read_jsonl(run / "intervention_events.jsonl")
    acks = [json.loads((run / f"pause_actor_{step}.ack.json").read_text()) for step in TARGETS]
    segment_rates = []
    prev_steps = prev_int = 0
    for ack in acks:
        ds, di = ack["steps"]-prev_steps, ack["interventions"]-prev_int
        segment_rates.append(di/ds if ds else 0.0); prev_steps, prev_int = ack["steps"], ack["interventions"]
    rate_trend = trend(segment_rates)
    best_step, best = max(evaluations.items(), key=rank); a = best["aggregate"]
    strong = any(v["aggregate"]["success_count"] >= 2 and v["aggregate"]["contact_health"]
                 for v in evaluations.values())
    partial = (not strong and a["success_count"] == 0 and a["near_hole_count"] > 0 and
               a["median_min_position_error"] <= .90 * PAPER_BASELINE_MIN_POSITION and
               rate_trend == "DECREASING")
    result_class = "PASS" if strong else "PARTIAL" if partial else "FAIL"
    intervention_fraction = actor["interventions"] / actor["steps"] if actor["steps"] else 0.0
    durations = [e["duration"] for e in events]
    restoration = [bool(e["recovery_restored"]) for e in events]
    if intervention_fraction >= .90:
        root = "POLICY_NOT_LEARNING_FROM_INTERVENTION"
    elif restoration and np.mean(restoration) < .5:
        root = "EXPERT_RECOVERY_OR_TRIGGER_INVALID"
    elif restoration and sum(restoration) and len(events) > 1:
        gaps = [events[i+1]["start_step"] - events[i]["end_step"]
                for i in range(len(events)-1) if events[i+1]["episode"] == events[i]["episode"]]
        root = "INTERVENTION_RETENTION_FAILURE" if gaps and np.median(gaps) <= 2 else "POLICY_NOT_LEARNING_FROM_INTERVENTION"
    else:
        root = "POLICY_NOT_LEARNING_FROM_INTERVENTION"
    final5 = run_eval(run, best_step, CONFIG["pilot"]["final_episodes"], "final5")
    converged = final5["aggregate"]["success_count"] >= 3 and final5["aggregate"]["contact_health"]
    result = {
        "HIL_SERL_V1_PREFLIGHT": "PASS", "HIL_SERL_V1_TRAINING": "PASS",
        "TOTAL_ENV_STEPS": actor["steps"], "TOTAL_INTERVENTIONS": actor["interventions"],
        "INTERVENTION_FRACTION": intervention_fraction,
        "MEAN_INTERVENTION_DURATION": float(np.mean(durations)) if durations else 0.0,
        "MEDIAN_INTERVENTION_DURATION": float(np.median(durations)) if durations else 0.0,
        "EPISODE_INTERVENTION_FRACTION": float(np.mean([e["had_intervention"] for e in episodes])) if episodes else 0.0,
        "RECOVERY_RESTORATION_RATE": float(np.mean(restoration)) if restoration else 0.0,
        "ONLINE_REPLAY_SIZE": learner["final_online_size"],
        "INTERVENTION_BUFFER_SIZE": learner["intervention_buffer_size"],
        "CHECKPOINT_100_SUCCESS": f"{evaluations[100]['aggregate']['success_count']}/3",
        "CHECKPOINT_250_SUCCESS": f"{evaluations[250]['aggregate']['success_count']}/3",
        "CHECKPOINT_500_SUCCESS": f"{evaluations[500]['aggregate']['success_count']}/3",
        "BEST_CHECKPOINT": best_step, "BEST_SUCCESS": f"{a['success_count']}/3",
        "BEST_NEAR_HOLE": a["near_hole_count"],
        "BEST_MIN_POSITION_ERROR": a["median_min_position_error"],
        "BEST_CONTACT_HEALTH": "PASS" if a["contact_health"] else "FAIL",
        "INTERVENTION_RATE_TREND": rate_trend, "INTERVENTION_SEGMENT_RATES": segment_rates,
        "HIL_SERL_V1_RESULT": result_class, "MATCHED_CONFIRMATION": matched_status,
        "FINAL_CANONICAL_SUCCESS": f"{final5['aggregate']['success_count']}/5",
        "HIL_SERL_AUBO_PEG_CONVERGED": "YES" if converged else "NO",
        "PRIMARY_ROOT_CAUSE": None if strong else root,
        "PURE_SERL_RLPD_ARTIFACTS_CHANGED": "NO",
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
        "INTERVENTION_ENABLED_IN_FORMAL_EVAL": "NO",
        "RESIDUAL_PROCESSES": "NONE",
        "NEXT": "PACKAGE_SUCCESSFUL_PEG" if converged else
                "HIL_SERL_EXTEND" if strong else "INTERVENTION_RETENTION_REVIEW",
    }
    dump(run / "HIL_SERL_V1_RESULT.json", result)
    return result


def main():
    evaluations = execute(RUN)
    initial = summarize(RUN, evaluations)
    # Promotion is a fresh full 500-update replication, not a repeated evaluation.
    if initial["HIL_SERL_V1_RESULT"] == "PASS":
        matched_run = BRANCH / "runs/matched_seed0"
        matched_evals = execute(matched_run)
        matched = summarize(matched_run, matched_evals)
        matched_ok = matched["HIL_SERL_V1_RESULT"] == "PASS"
        initial["MATCHED_CONFIRMATION"] = "PASS" if matched_ok else "FAIL"
        if matched_ok and initial["HIL_SERL_AUBO_PEG_CONVERGED"] != "YES":
            initial["NEXT"] = "HIL_SERL_EXTEND"
        dump(RUN / "HIL_SERL_V1_RESULT.json", initial)
    print(json.dumps(initial, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

