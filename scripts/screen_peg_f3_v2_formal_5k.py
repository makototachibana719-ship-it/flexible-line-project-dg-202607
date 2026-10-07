#!/usr/bin/env python3
"""Inference-only, uniform screening of the frozen Peg F3 V2 formal checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE="false",
                  XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl")

import jax
import numpy as np
import yaml

from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.observation_contract import V2, validate_config
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
RUN = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run"
EXPECTED_CONFIG = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
EXPECTED_DATASET = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
STEPS = (1000, 2000, 3000, 4000, 5000)
QUICK_SEEDS = (0, 1, 2)
EXPANDED_SEEDS = tuple(range(10))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def state_signature(state) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        if not hasattr(leaf, "dtype"):
            continue
        a = np.asarray(jax.device_get(leaf))
        require(np.isfinite(a).all(), "nonfinite parameter")
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def raw_env(env):
    seen = set()
    while hasattr(env, "env") and id(env) not in seen:
        seen.add(id(env))
        env = env.env
    return env


def contact_flags(env):
    raw = raw_env(env)
    model, data = raw.model, raw.data
    link01 = unexpected = False
    pairs = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        tokens, bodies = [], []
        for gid in (int(c.geom1), int(c.geom2)):
            body_id = int(model.geom_bodyid[gid])
            data_id = int(model.geom_dataid[gid])
            geom_name = model.geom(gid).name
            body_name = model.body(body_id).name
            mesh_name = (model.mesh(data_id).name
                         if int(model.geom_type[gid]) == 7 and data_id >= 0 else None)
            tokens.extend((geom_name, body_name, mesh_name))
            bodies.append(body_name)
        is_link01 = frozenset(bodies) == frozenset(("base_link", "shoulder_Link"))
        expected = any("hole" in str(token).lower() for token in tokens if token is not None)
        link01 |= is_link01
        unexpected |= not expected
        pairs.append(bodies)
    return link01, unexpected, pairs


def aggregate(rows):
    def mean(key): return float(np.mean([r[key] for r in rows]))
    def median(key): return float(np.median([r[key] for r in rows]))
    return {
        "episodes": len(rows),
        "success_count": sum(r["success"] for r in rows),
        "success_rate": mean("success"),
        "mean_return": mean("return"),
        "median_return": median("return"),
        "near_hole_episode_rate": mean("near_hole_reached"),
        "mean_near_hole_transitions": mean("near_hole_transition_count"),
        "mean_approach_reduction": mean("approach_reduction"),
        "median_min_position": median("min_position_error"),
        "median_min_lateral": median("min_lateral_error"),
        "median_min_axial": median("min_axial_error"),
        "median_min_orientation": median("min_orientation_error"),
        "max_force": max(r["max_force"] for r in rows),
        "contact_failures": sum(not r["contact_health"] for r in rows),
        "mean_action_saturation": mean("action_saturation"),
    }


def rank_key(item):
    _, a = item
    return (a["success_rate"], a["near_hole_episode_rate"],
            a["mean_approach_reduction"], -a["median_min_lateral"],
            -a["median_min_axial"], -a["median_min_orientation"],
            -a["contact_failures"], -a["mean_action_saturation"], a["mean_return"])


def trend_class(quick, expanded):
    ordered = [quick[str(s)] for s in STEPS]
    stable_success = any(a["success_count"] >= 2 for a in expanded.values())
    contact_instability = sum(a["contact_failures"] for a in ordered) > 0
    early, late = ordered[:2], ordered[-2:]
    em = lambda key: float(np.mean([x[key] for x in early]))
    lm = lambda key: float(np.mean([x[key] for x in late]))
    improvements = {
        "near_hole": lm("near_hole_episode_rate") > em("near_hole_episode_rate") + 1e-9,
        "approach": lm("mean_approach_reduction") > em("mean_approach_reduction") + 1e-5,
        "position": lm("median_min_position") < em("median_min_position") - 1e-5,
        "lateral": lm("median_min_lateral") < em("median_min_lateral") - 1e-5,
        "axial": lm("median_min_axial") < em("median_min_axial") - 1e-5,
        "orientation": lm("median_min_orientation") < em("median_min_orientation") - 1e-5,
    }
    regressions = sum((lm(k) > em(k) + 1e-5) for k in
                      ("median_min_position", "median_min_lateral", "median_min_axial", "median_min_orientation"))
    n_improved = sum(improvements.values())
    sustained = n_improved >= 2 and ordered[-1]["mean_approach_reduction"] >= ordered[0]["mean_approach_reduction"] - 1e-5
    if stable_success and not contact_instability:
        label = "CLEAR_POSITIVE"
    elif contact_instability or (regressions >= 3 and not stable_success):
        label = "REGRESSION"
    elif sustained:
        label = "PARTIAL"
    else:
        label = "NONE"
    return label, {"stable_success": stable_success, "contact_instability": contact_instability,
                   "improvements": improvements, "sustained_improvement": sustained,
                   "geometry_regressions": regressions}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    require(ROOT in out.parents and not out.exists(), "fresh isolated output required")
    out.mkdir(parents=True, exist_ok=False)
    result = {"PEG_F3_V2_FORMAL_5K_SCREENING": "FAIL", "TRAINING_STARTED": False,
              "FORMAL_10K_CONTINUE": "BLOCKED", "DIAGNOSTIC": "READY", "NEXT": "BLOCKED"}
    env = None
    protected = {}
    episodes = []
    try:
        cfg = yaml.safe_load(CONFIG.read_text())
        dataset = ROOT / cfg["demo"]["path"]
        require(sha(CONFIG) == EXPECTED_CONFIG, "formal config SHA256")
        require(sha(dataset) == EXPECTED_DATASET == cfg["demo"]["sha256"], "dataset provenance")
        require(validate_config(cfg, for_training=True) == V2, "V2 config")
        require(cfg["fresh_initialization"] is True and cfg["load_legacy_checkpoint"] is False, "V1 contamination")
        process_text = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                                      text=True, check=True).stdout
        residual = [x for x in process_text.splitlines() if
                    ("run_peg_f3_v2_rlpd_smoke.py" in x or "async_sac_state_sim.py" in x)
                    and str(os.getpid()) not in x]
        require(not residual, "training process already running")
        cps = {step: RUN / "checkpoints" / f"checkpoint_{step}" for step in STEPS}
        protected_paths = [CONFIG, dataset]
        for step, cp in cps.items():
            protected_paths.extend((cp, cp.with_name(cp.name + "_online_replay.pkl"),
                                    cp.with_name(cp.name + "_resume_metadata.json")))
        require(all(path.is_file() for path in protected_paths), "protected artifact missing")
        protected = {str(path): sha(path) for path in protected_paths}
        dump(out / "protected_before.json", protected)
        require(jax.default_backend() == "gpu", "GPU inference backend")
        pkg = ROOT / "external/packages/peg_insertion/peg_insertion"
        sys.path[:0] = [str(pkg / "code/example/async_rlpd_peg_sim"), str(pkg / "code/env")]
        from async_sac_state_sim import make_sac_agent
        from peg_env_adapter import make_peg_state_env
        with portable_peg_runtime(ROOT):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
            obs, _ = env.reset(seed=0)
            require(obs.shape == (20,) and obs.dtype == np.float32 and np.isfinite(obs).all(), "obs contract")
            require(env.action_space.shape == (7,), "action contract")
            task = raw_env(env).task
            require(task.position_tolerance == .002 and task.orientation_tolerance == .05, "success contract")
            fresh = make_sac_agent(seed=0, sample_obs=np.zeros(20, np.float32),
                                   sample_action=np.zeros(7, np.float32))
            checkpoint_detail = {}
            for step, cp in cps.items():
                meta = json.loads(cp.with_name(cp.name + "_resume_metadata.json").read_text())
                replay = cp.with_name(cp.name + "_online_replay.pkl")
                state = restore_fresh_state(cp.parent, fresh.state, step)
                checkpoint_detail[str(step)] = {
                    "step_exact": meta.get("checkpoint_step") == meta.get("learner_update_steps") == step,
                    "config_binding": meta.get("config_hash") == canonical(cfg),
                    "dataset_binding": meta.get("demo_sha256") == EXPECTED_DATASET,
                    "v2_binding": meta.get("observation_contract_version") == V2,
                    "model_hash": meta.get("model_sha256") == sha(cp),
                    "replay_hash": meta.get("online_replay_sha256") == sha(replay),
                    "parameters_finite": bool(state_signature(state)),
                }
                require(all(checkpoint_detail[str(step)].values()), f"checkpoint {step} preflight")
            result["preflight"] = {"status": "PASS", "backend": jax.default_backend(),
                                   "config_sha256": EXPECTED_CONFIG, "dataset_sha256": EXPECTED_DATASET,
                                   "deterministic": True, "exploration_noise": False,
                                   "online_update": False, "replay_constructed": False,
                                   "max_steps": 500, "quick_seeds": list(QUICK_SEEDS),
                                   "expanded_seeds": list(EXPANDED_SEEDS),
                                   "near_hole": "lateral<=0.010 and axial<=0.020",
                                   "checkpoint_details": checkpoint_detail}
            print("EVAL_PREFLIGHT=PASS", flush=True)

            def evaluate(step, seed, stage):
                cp = cps[step]
                state = restore_fresh_state(cp.parent, fresh.state, step)
                agent = fresh.replace(state=state)
                initial_sig = state_signature(agent.state)
                obs, _ = env.reset(seed=seed)
                first = float(np.linalg.norm(obs[13:16]))
                mins = {"position": first, "lateral": float("inf"), "axial": float("inf"),
                        "orientation": float("inf")}
                ret = 0.0; near = sat = 0; max_force = 0.0
                link01 = unexpected = False; terminated = truncated = success = False
                t0 = time.monotonic()
                for index in range(500):
                    action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
                    require(action.shape == (7,) and np.isfinite(action).all() and
                            np.all(np.abs(action) <= 1.000001), "deterministic action contract")
                    sat += int((np.abs(action) >= .99).sum())
                    obs, reward, terminated, truncated, info = env.step(action)
                    require(obs.shape == (20,) and obs.dtype == np.float32 and np.isfinite(obs).all(), "step obs")
                    keys = ("position_error", "lateral_error", "axial_error", "orientation_error", "force_norm", "is_success")
                    require(all(k in info for k in keys) and np.isfinite([info[k] for k in keys]).all(), "metric schema")
                    expected_success = (info["lateral_error"] <= .002 and info["axial_error"] <= .003
                                        and info["orientation_error"] <= .05)
                    require(bool(info["is_success"]) == expected_success, "success semantics")
                    expected_reward = (-80 * info["lateral_error"] - 20 * info["axial_error"]
                                       - 2 * info["orientation_error"] + 2 * (info["lateral_error"] < .01)
                                       + 25 * expected_success)
                    require(np.isclose(reward, expected_reward), "reward semantics")
                    mins["position"] = min(mins["position"], float(info["position_error"]))
                    for key in ("lateral", "axial", "orientation"):
                        mins[key] = min(mins[key], float(info[key + "_error"]))
                    near += int(info["lateral_error"] <= .010 and info["axial_error"] <= .020)
                    max_force = max(max_force, float(info["force_norm"]))
                    l, u, _ = contact_flags(env); link01 |= l; unexpected |= u
                    ret += float(reward); success = bool(info["is_success"])
                    if terminated or truncated:
                        break
                require(state_signature(agent.state) == initial_sig, "policy mutated during inference")
                row = {"stage": stage, "checkpoint": step, "episode": seed, "seed": seed,
                       "success": success, "steps": index + 1, "return": ret,
                       "first_position_error": first, "min_position_error": mins["position"],
                       "approach_reduction": first - mins["position"],
                       "min_lateral_error": mins["lateral"], "min_axial_error": mins["axial"],
                       "min_orientation_error": mins["orientation"],
                       "near_hole_transition_count": near, "near_hole_reached": near > 0,
                       "max_force": max_force, "link01_contact": link01,
                       "unexpected_contact": unexpected, "contact_health": not (link01 or unexpected),
                       "action_saturation": sat / (7 * (index + 1)),
                       "termination_reason": "SUCCESS" if success else ("TRUNCATED" if truncated else
                                             ("TERMINATED" if terminated else "EVALUATOR_HORIZON")),
                       "terminated": bool(terminated), "truncated": bool(truncated),
                       "final_lateral": float(info["lateral_error"]),
                       "final_axial": float(info["axial_error"]),
                       "final_orientation": float(info["orientation_error"]),
                       "wall_seconds": time.monotonic() - t0}
                episodes.append(row)
                with (out / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps({k: row[k] for k in ("stage", "checkpoint", "seed", "success",
                      "near_hole_reached", "approach_reduction", "contact_health")}), flush=True)
                return row

            quick_rows = {str(step): [evaluate(step, seed, "quick") for seed in QUICK_SEEDS]
                          for step in STEPS}
            quick = {step: aggregate(rows) for step, rows in quick_rows.items()}
            selected = [int(step) for step, _ in sorted(quick.items(), key=rank_key, reverse=True)[:2]]
            expanded_rows = {}
            for step in selected:
                rows = list(quick_rows[str(step)])
                rows.extend(evaluate(step, seed, "expansion") for seed in EXPANDED_SEEDS[3:])
                expanded_rows[str(step)] = rows
            expanded = {step: aggregate(rows) for step, rows in expanded_rows.items()}
            best_step = int(max(expanded.items(), key=rank_key)[0])
            best = expanded[str(best_step)]
            trend, evidence = trend_class(quick, expanded)
            ready = trend == "CLEAR_POSITIVE" or (trend == "PARTIAL" and evidence["sustained_improvement"])
            result.update({"PEG_F3_V2_FORMAL_5K_SCREENING": "PASS", "quick": quick,
                           "selected_checkpoints": selected, "expanded": expanded,
                           "BEST_CHECKPOINT": best_step, "best": best,
                           "EXPANDED_EVAL_EPISODES": {step: a["episodes"] for step, a in expanded.items()},
                           "LEARNING_TREND": trend, "trend_evidence": evidence,
                           "FORMAL_10K_CONTINUE": "READY" if ready else "BLOCKED",
                           "DIAGNOSTIC": "NOT_REQUIRED" if ready else "READY",
                           "NEXT": "USER_AUTHORIZATION_REQUIRED_FOR_10K" if ready else "DIAGNOSTIC"})
    except Exception as exc:
        result["first_failed_gate"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        after = {path: sha(Path(path)) for path in protected}
        dump(out / "protected_after.json", after)
        result["PROTECTED_ARTIFACTS_CHANGED"] = protected != after
        if protected != after:
            result.update(PEG_F3_V2_FORMAL_5K_SCREENING="FAIL", FORMAL_10K_CONTINUE="BLOCKED",
                          DIAGNOSTIC="READY", NEXT="BLOCKED", first_failed_gate="protected artifact changed")
        result["evaluated_episodes"] = len(episodes)
        dump(out / "validation.json", result)
        (out / "REPORT.md").write_text("# Peg F3 V2 Formal-5K Checkpoint Screening\n\n```json\n" +
                                             json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["PEG_F3_V2_FORMAL_5K_SCREENING"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
