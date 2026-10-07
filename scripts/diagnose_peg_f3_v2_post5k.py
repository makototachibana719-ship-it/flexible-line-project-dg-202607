#!/usr/bin/env python3
"""Read-only D1-D5 root-cause diagnostic for Peg F3 V2 formal 5K."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
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
DATASET_DIR = ROOT / "results/datasets/peg_v2_accepted_11_20260906"
DATASET = DATASET_DIR / "peg_v2_accepted_11.pkl"
PROVENANCE = DATASET_DIR / "episode_provenance.json"
MANIFEST = DATASET_DIR / "manifest.json"
RUN = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run"
SCREEN = ROOT / "results/validation/peg/f3_v2_formal_5k_screening_20260906/validation.json"
EXPECTED_CONFIG = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
EXPECTED_DATASET = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
STEPS = (1000, 2000, 3000, 4000, 5000)


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


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stats(values):
    a = np.asarray(values)
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "min": float(np.min(a)), "max": float(np.max(a)),
            "positive_fraction": float(np.mean(a > 0))}


def error_stats(observations):
    obs = np.asarray(observations, dtype=np.float64)
    relative = obs[:, 13:16]
    axial = np.abs(relative[:, 0])
    lateral = np.linalg.norm(relative[:, 1:3], axis=1)
    q1, q2 = obs[:, 3:7], obs[:, 16:20]
    q1 /= np.maximum(np.linalg.norm(q1, axis=1, keepdims=True), 1e-12)
    q2 /= np.maximum(np.linalg.norm(q2, axis=1, keepdims=True), 1e-12)
    orientation = 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0, 1))
    force = np.linalg.norm(obs[:, 7:10], axis=1)
    return {"lateral": stats(lateral), "axial": stats(axial),
            "orientation": stats(orientation), "force_norm": stats(force),
            "force_nonzero_fraction": float(np.mean(force > 1e-6))}


def tree_arrays(tree):
    return [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(tree)
            if hasattr(x, "dtype")]


def tree_digest(tree):
    h = hashlib.sha256()
    for a in tree_arrays(tree):
        require(np.isfinite(a).all(), "nonfinite checkpoint state")
        h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()


def tree_l2_delta(left, right):
    la, ra = tree_arrays(left), tree_arrays(right)
    require(len(la) == len(ra) and all(a.shape == b.shape for a, b in zip(la, ra)), "parameter tree mismatch")
    delta = math.sqrt(sum(float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
                          for a, b in zip(la, ra)))
    base = math.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in la))
    return {"absolute_l2": delta, "relative_l2": delta / max(base, 1e-12)}


def raw_env(env):
    seen = set()
    while hasattr(env, "env") and id(env) not in seen:
        seen.add(id(env)); env = env.env
    return env


def contacts(env):
    raw = raw_env(env); model, data = raw.model, raw.data; result = []
    for i in range(int(data.ncon)):
        c = data.contact[i]; sides = []
        for gid in (int(c.geom1), int(c.geom2)):
            bid, did = int(model.geom_bodyid[gid]), int(model.geom_dataid[gid])
            sides.append({"geom": model.geom(gid).name, "body": model.body(bid).name,
                          "mesh": model.mesh(did).name if int(model.geom_type[gid]) == 7 and did >= 0 else None})
        bodies = [s["body"] for s in sides]
        tokens = [v for s in sides for v in s.values() if v is not None]
        result.append({"pair": sides,
                       "link01": frozenset(bodies) == frozenset(("base_link", "shoulder_Link")),
                       "expected_task_contact": any("hole" in str(v).lower() for v in tokens),
                       "distance": float(c.dist)})
    return result


def cosine_rows(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 1e-12
    values = np.full(len(a), np.nan)
    values[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return float(np.nanmean(values)) if valid.any() else None, float(np.mean(valid))


def agreement(demo, policy, mask):
    d, p = demo[mask], policy[mask]
    require(len(d) > 0, "empty action agreement subset")
    cosine, valid = cosine_rows(d, p)
    return {"count": len(d), "mae": float(np.mean(np.abs(d - p))),
            "per_dimension_mae": np.mean(np.abs(d - p), axis=0).tolist(),
            "cosine_similarity": cosine, "cosine_valid_fraction": valid}


def q_preference(agent, obs, demo_action, policy_action, mask):
    o, d, p = obs[mask], demo_action[mask], policy_action[mask]
    require(len(o) > 0, "empty Q subset")
    def q(a):
        values = np.asarray(jax.device_get(agent.forward_critic(o, a, rng=None, train=False)))
        require(np.isfinite(values).all(), "nonfinite Q")
        return values.mean(axis=0)
    qd, qp, qz = q(d), q(p), q(np.zeros_like(d))
    gap = qd - qp
    return {"count": len(o), "q_demo_gt_policy_rate": float(np.mean(gap > 0)),
            "q_policy_gt_demo_rate": float(np.mean(gap < 0)),
            "q_equal_rate": float(np.mean(gap == 0)),
            "mean_q_demo_minus_policy": float(np.mean(gap)),
            "mean_q_demo": float(np.mean(qd)), "mean_q_policy": float(np.mean(qp)),
            "mean_q_zero": float(np.mean(qz))}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    require(ROOT in out.parents and not out.exists(), "fresh isolated output required")
    out.mkdir(parents=True, exist_ok=False)
    result = {"PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC": "FAIL",
              "D1_TRAJECTORY_CAUSAL_TRACE": "NOT_RUN", "D2_DEMO_REPLAY_SEMANTICS": "NOT_RUN",
              "D3_POLICY_DEMO_ALIGNMENT": "NOT_RUN", "D4_CRITIC_PREFERENCE": "NOT_RUN",
              "D5_CHECKPOINT_SANITY": "NOT_RUN", "FORMAL_10K_CONTINUE": "BLOCKED",
              "TRAINING_STARTED": False, "NEXT": "BLOCKED", "PROTECTED_ARTIFACTS_CHANGED": "NO"}
    env = None; protected = {}
    try:
        cfg = yaml.safe_load(CONFIG.read_text()); manifest = json.loads(MANIFEST.read_text())
        prov = json.loads(PROVENANCE.read_text()); screening = json.loads(SCREEN.read_text())
        require(sha(CONFIG) == EXPECTED_CONFIG and sha(DATASET) == EXPECTED_DATASET, "frozen hash")
        require(validate_config(cfg, for_training=True) == V2, "V2 config")
        require(manifest["v2_only"] is True and manifest["episodes"] == 11 and manifest["transitions"] == 1921, "manifest")
        require(screening["PEG_F3_V2_FORMAL_5K_SCREENING"] == "PASS" and screening["LEARNING_TREND"] == "REGRESSION", "screening gate")
        cps = {s: RUN / "checkpoints" / f"checkpoint_{s}" for s in STEPS}
        protected_paths = [CONFIG, DATASET, MANIFEST, PROVENANCE, SCREEN]
        for e in prov: protected_paths.append(ROOT / e["episode_path"] / "transitions.jsonl")
        for s, cp in cps.items(): protected_paths += [cp, cp.with_name(cp.name + "_online_replay.pkl"), cp.with_name(cp.name + "_resume_metadata.json")]
        require(all(p.is_file() for p in protected_paths), "protected artifact missing")
        protected = {str(p): sha(p) for p in protected_paths}; write_json(out / "protected_before.json", protected)
        ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True).stdout
        require(not any(("async_sac_state_sim.py" in x or "run_peg_f3_v2_rlpd_smoke.py" in x) and str(os.getpid()) not in x for x in ps.splitlines()), "training process running")
        require(jax.default_backend() == "gpu", "GPU inference backend")

        # D2: rebuild the exact training representation from every accepted source transition.
        dataset = pickle.loads(DATASET.read_bytes())
        require(isinstance(dataset, list) and len(dataset) == 1921, "dataset cardinality")
        phase, episode_type, raw_metrics = [], [], []
        equal = {k: True for k in ("OBS_EQUAL", "NEXT_OBS_EQUAL", "ACTION_EQUAL", "REWARD_EQUAL", "MASK_EQUAL", "DONE_EQUAL", "EPISODE_BOUNDARY_EQUAL")}
        cursor = 0
        for e in prov:
            rows = [json.loads(line) for line in (ROOT / e["episode_path"] / "transitions.jsonl").open()]
            require(cursor == e["dataset_start"] and cursor + len(rows) == e["dataset_stop_exclusive"], "provenance boundary")
            for local, row in enumerate(rows):
                item = dataset[cursor]; obs = np.asarray(row["obs"], np.float32); nxt = np.asarray(row["next_obs"], np.float32); act = np.asarray(row["action"], np.float32)
                equal["OBS_EQUAL"] &= np.array_equal(item["observations"], obs)
                equal["NEXT_OBS_EQUAL"] &= np.array_equal(item["next_observations"], nxt)
                equal["ACTION_EQUAL"] &= np.array_equal(item["actions"], act)
                equal["REWARD_EQUAL"] &= np.array_equal(np.asarray(item["rewards"]), np.asarray(row["reward"], np.float32))
                equal["MASK_EQUAL"] &= float(item["masks"]) == (0.0 if row["terminated"] else 1.0)
                equal["DONE_EQUAL"] &= bool(item["dones"]) == bool(row["terminated"] or row["truncated"])
                equal["EPISODE_BOUNDARY_EQUAL"] &= bool(item["dones"]) == (local == len(rows) - 1)
                phase.append(row["phase"]); episode_type.append(e["type"]); raw_metrics.append(row["after"]["metrics"]); cursor += 1
        require(cursor == 1921 and all(equal.values()), "Demo to replay semantic mismatch")
        arrays = {k: np.stack([np.asarray(x[k]) for x in dataset]) if k in ("observations", "next_observations", "actions") else np.asarray([x[k] for x in dataset]) for k in dataset[0]}
        require(arrays["observations"].shape == (1921, 20) and arrays["actions"].shape == (1921, 7), "dataset schema")
        require(all(np.isfinite(v).all() for v in arrays.values()), "dataset nonfinite")
        online_payload = pickle.loads(cps[5000].with_name("checkpoint_5000_online_replay.pkl").read_bytes())
        online = online_payload["dataset_dict"]; size = int(online_payload["size"])
        online = {k: np.asarray(v)[:size] for k, v in online.items()}
        require(size == 36451 and online["observations"].shape == (size, 20), "online replay")
        demo_distribution = {"reward": stats(arrays["rewards"]), "state": error_stats(arrays["observations"]),
                             "masks": {str(x): int(np.sum(arrays["masks"] == x)) for x in (0, 1)},
                             "dones": {str(x): int(np.sum(arrays["dones"] == x)) for x in (False, True)}}
        online_distribution = {"reward": stats(online["rewards"]), "state": error_stats(online["observations"]),
                               "masks": {str(x): int(np.sum(online["masks"] == x)) for x in (0, 1)},
                               "dones": {str(x): int(np.sum(online["dones"] == x)) for x in (False, True)},
                               "contact_frequency": "NOT_RECONSTRUCTABLE_FROM_REPLAY; force_nonzero_fraction reported as wrench proxy"}
        d2 = {**equal, "DEMO_TO_REPLAY_EXACT": "PASS", "REWARD_SEMANTICS": "PASS",
              "MASK_SEMANTICS": "PASS", "TERMINAL_SEMANTICS": "PASS", "ACTION_SEMANTICS": "PASS",
              "OBS_SEMANTICS": "PASS", "demo_distribution": demo_distribution,
              "online_distribution": online_distribution,
              "DEMO_ONLINE_REWARD_SHIFT": online_distribution["reward"]["mean"] - demo_distribution["reward"]["mean"],
              "DEMO_ONLINE_STATE_SHIFT": {k: online_distribution["state"][k]["mean"] - demo_distribution["state"][k]["mean"] for k in ("lateral", "axial", "orientation")}}
        result["D2_DEMO_REPLAY_SEMANTICS"] = "PASS"; result["D2"] = d2
        print("D2_DEMO_REPLAY_SEMANTICS=PASS", flush=True)

        pkg = ROOT / "external/packages/peg_insertion/peg_insertion"
        sys.path[:0] = [str(pkg / "code/example/async_rlpd_peg_sim"), str(pkg / "code/env")]
        from async_sac_state_sim import make_sac_agent
        from peg_env_adapter import make_peg_state_env
        with portable_peg_runtime(ROOT):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
            fresh = make_sac_agent(seed=0, sample_obs=np.zeros(20, np.float32), sample_action=np.zeros(7, np.float32))
            agents, states = {}, {}
            for step, cp in cps.items():
                meta = json.loads(cp.with_name(cp.name + "_resume_metadata.json").read_text())
                require(meta["checkpoint_step"] == meta["learner_update_steps"] == step and meta["config_hash"] == canonical(cfg), "checkpoint step/config")
                state = restore_fresh_state(cp.parent, fresh.state, step); require(bool(tree_digest(state)), "checkpoint finite")
                states[step] = state; agents[step] = fresh.replace(state=state)

            # D1: one necessary deterministic causal trace with exact contact pairs.
            agent = agents[5000]; obs, _ = env.reset(seed=0); trace = []; first_contact = None; previous = None
            for index in range(500):
                action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
                obs, reward, terminated, truncated, info = env.step(action); cs = contacts(env)
                row = {"step": index + 1, "position_error": float(info["position_error"]),
                       "lateral_error": float(info["lateral_error"]), "axial_error": float(info["axial_error"]),
                       "orientation_error": float(info["orientation_error"]), "action": action.tolist(),
                       "commanded_translation": (action[:3] * .002).tolist(),
                       "commanded_rotation": (action[3:6] * np.deg2rad(1)).tolist(),
                       "contacts": cs, "reward": float(reward), "done": bool(terminated or truncated),
                       "mask": 0.0 if terminated else 1.0}
                trace.append(row)
                if cs and first_contact is None: first_contact = (row, previous)
                previous = row
                if terminated or truncated: break
            require(first_contact is not None, "screened contact failure not reproduced")
            contact_row, pre = first_contact; pre = pre or contact_row
            window = trace[max(0, contact_row["step"] - 21):contact_row["step"] - 1]
            require(window, "no pre-contact window")
            acts = np.asarray([r["action"] for r in window])
            aligned_before = any(r["lateral_error"] <= .01 for r in trace[:contact_row["step"] - 1])
            axial_advance = any(abs(r["commanded_translation"][0]) > 1e-5 and r["lateral_error"] > .01 for r in trace[:contact_row["step"] - 1])
            d1 = {"FIRST_CONTACT_PAIR": contact_row["contacts"], "FIRST_CONTACT_STEP": contact_row["step"],
                  "PRE_CONTACT_LATERAL": pre["lateral_error"], "PRE_CONTACT_AXIAL": pre["axial_error"],
                  "PRE_CONTACT_ORIENTATION": pre["orientation_error"],
                  "pre_contact_20": {"mean_abs_action_x": float(np.mean(np.abs(acts[:, 0]))),
                                     "mean_abs_action_y": float(np.mean(np.abs(acts[:, 1]))),
                                     "mean_abs_action_z": float(np.mean(np.abs(acts[:, 2]))),
                                     "mean_abs_rotation_actions": float(np.mean(np.abs(acts[:, 3:6])))},
                  "AXIAL_ADVANCE_BEFORE_ALIGNMENT": "YES" if axial_advance and not aligned_before else "NO",
                  "LATERAL_CORRECTION_ACTIVE": "YES" if float(np.mean(np.abs(acts[:, 1:3]))) >= .05 else "NO",
                  "ROTATION_CORRECTION_ACTIVE": "YES" if float(np.mean(np.abs(acts[:, 3:6]))) >= .05 else "NO"}
            d1["LATERAL_CORRECTION_STRENGTH"] = float(np.mean(np.abs(acts[:, 1:3])))
            d1["ORIENTATION_CORRECTION_STRENGTH"] = float(np.mean(np.abs(acts[:, 3:6])))
            d1["FAILURE_BEHAVIOR_CLASS"] = "EARLY_CONTACT_BEFORE_ALIGNMENT" if not aligned_before else "CONTACT_AFTER_ALIGNMENT"
            write_json(out / "d1_checkpoint5000_trace.json", trace)
            result["D1_TRAJECTORY_CAUSAL_TRACE"] = "PASS"; result["D1"] = d1
            print("D1_TRAJECTORY_CAUSAL_TRACE=PASS", flush=True)

            # D3: deterministic checkpoint-policy agreement on every demo state.
            obs_demo, actions_demo = arrays["observations"], arrays["actions"]
            phase_a, type_a = np.asarray(phase), np.asarray(episode_type)
            masks = {t: type_a == t for t in ("standard", "lateral_plus", "lateral_minus", "rotation_roll", "rotation_pitch")}
            masks["lateral_correction"] = np.isin(type_a, ["lateral_plus", "lateral_minus"]) & (phase_a == "lateral_alignment") & (np.abs(actions_demo[:, 1]) >= .99)
            masks["rotation_correction"] = np.isin(type_a, ["rotation_roll", "rotation_pitch"]) & (phase_a == "orientation_correction")
            raw_lat = np.asarray([m["lateral_error"] for m in raw_metrics]); raw_ax = np.asarray([m["axial_error"] for m in raw_metrics])
            masks["near_hole"] = (raw_lat <= .010) & (raw_ax <= .020)
            require(all(mask.any() for mask in masks.values()), "specialized subset missing")
            d3 = {}; policies = {}
            for step in STEPS:
                policy_actions = np.asarray(jax.device_get(agents[step].sample_actions(observations=obs_demo, argmax=True)))
                require(policy_actions.shape == actions_demo.shape and np.isfinite(policy_actions).all(), "policy action batch")
                policies[step] = policy_actions
                d3[str(step)] = {"overall": agreement(actions_demo, policy_actions, np.ones(1921, bool)),
                                 "by_subset": {name: agreement(actions_demo, policy_actions, mask) for name, mask in masks.items()}}
            first_mae, last_mae = d3["1000"]["overall"]["mae"], d3["5000"]["overall"]["mae"]
            trend = "IMPROVING" if last_mae < .95 * first_mae else ("DEGRADING" if last_mae > 1.05 * first_mae else "STABLE")
            d3["ALIGNMENT_TREND"] = trend
            result["D3_POLICY_DEMO_ALIGNMENT"] = "PASS"; result["D3"] = d3
            print("D3_POLICY_DEMO_ALIGNMENT=PASS", flush=True)

            # D4: critic preference on all representative correction/near-hole demo states.
            q_masks = {"lateral": masks["lateral_correction"], "rotation": masks["rotation_correction"], "near_hole": masks["near_hole"]}
            d4 = {}
            for step in (1000, 3000, 5000):
                union = np.logical_or.reduce(list(q_masks.values()))
                d4[str(step)] = {"overall": q_preference(agents[step], obs_demo, actions_demo, policies[step], union),
                                 "by_subset": {name: q_preference(agents[step], obs_demo, actions_demo, policies[step], mask) for name, mask in q_masks.items()}}
            gaps = [d4[str(s)]["overall"]["mean_q_demo_minus_policy"] for s in (1000, 3000, 5000)]
            d4["CRITIC_TREND"] = "DEMO_PREFERENCE_IMPROVING" if gaps[-1] > gaps[0] else "DEMO_PREFERENCE_DEGRADING"
            result["D4_CRITIC_PREFERENCE"] = "PASS"; result["D4"] = d4
            print("D4_CRITIC_PREFERENCE=PASS", flush=True)

            # D5: actor/critic/temperature and optimizer state are finite and changing.
            d5 = {"steps": {}}
            for step in STEPS:
                state = states[step]
                params = state.params
                d5["steps"][str(step)] = {"state_digest": tree_digest(state),
                    "actor_digest": tree_digest(params["modules_actor"]), "critic_digest": tree_digest(params["modules_critic"]),
                    "temperature": float(np.asarray(jax.device_get(agents[step].forward_temperature()))),
                    "optimizer_finite": all(np.isfinite(a).all() for a in tree_arrays(state.opt_states))}
            d5["actor_1000_to_5000"] = tree_l2_delta(states[1000].params["modules_actor"], states[5000].params["modules_actor"])
            d5["critic_1000_to_5000"] = tree_l2_delta(states[1000].params["modules_critic"], states[5000].params["modules_critic"])
            d5["temperature_1000_to_5000"] = tree_l2_delta(states[1000].params["modules_temperature"], states[5000].params["modules_temperature"])
            d5.update(ACTOR_UPDATED=d5["actor_1000_to_5000"]["absolute_l2"] > 0,
                      CRITIC_UPDATED=d5["critic_1000_to_5000"]["absolute_l2"] > 0,
                      EVAL_ACTOR_BINDING=True, CHECKPOINT_STEP_BINDING=True)
            require(d5["ACTOR_UPDATED"] and d5["CRITIC_UPDATED"] and all(x["optimizer_finite"] for x in d5["steps"].values()), "frozen/wrong parameters")
            result["D5_CHECKPOINT_SANITY"] = "PASS"; result["D5"] = d5
            print("D5_CHECKPOINT_SANITY=PASS", flush=True)

        # Evidence-only classification: no algorithm/environment change is made here.
        d3_bad = d3["5000"]["overall"]["mae"] > 0.25 and trend != "IMPROVING"
        q5000 = d4["5000"]["overall"]
        critic_bad = q5000["q_policy_gt_demo_rate"] > .5 and q5000["mean_q_demo_minus_policy"] < 0
        contact_dominant = d1["FIRST_CONTACT_STEP"] <= 100 and d1["FAILURE_BEHAVIOR_CLASS"] == "EARLY_CONTACT_BEFORE_ALIGNMENT"
        supported = []
        if d3_bad: supported.append("C_POLICY_NOT_LEARNING_DEMO_CORRECTIONS")
        if critic_bad: supported.append("D_CRITIC_PREFERS_BAD_ONLINE_ACTIONS")
        if contact_dominant: supported.append("F_CONTACT_DYNAMICS_DOMINATE_LEARNING")
        if len(supported) > 1: primary, secondary = "G_MULTIPLE_SUPPORTED_CAUSES", supported[0]
        elif supported: primary, secondary = supported[0], "NONE"
        else: primary, secondary = "H_UNRESOLVED", "NONE"
        next_map = {"C_POLICY_NOT_LEARNING_DEMO_CORRECTIONS": "TRAINING_OBJECTIVE_DIAGNOSTIC_READY",
                    "D_CRITIC_PREFERS_BAD_ONLINE_ACTIONS": "RLPD_Q_BIAS_DIAGNOSTIC_READY",
                    "F_CONTACT_DYNAMICS_DOMINATE_LEARNING": "CONTACT_POLICY_DIAGNOSTIC_READY",
                    "G_MULTIPLE_SUPPORTED_CAUSES": "TRAINING_OBJECTIVE_DIAGNOSTIC_READY",
                    "H_UNRESOLVED": "BLOCKED"}
        result.update(PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC="PASS", PRIMARY_ROOT_CAUSE=primary,
                      SECONDARY_CAUSE=secondary, SUPPORTED_CAUSES=supported,
                      SUPPORTED_BY=["Demo-to-replay arrays/scalars/boundaries are exact",
                                    f"policy/demo MAE 1000={first_mae:.6f}, 5000={last_mae:.6f}, trend={trend}",
                                    f"checkpoint5000 critic Qdemo-Qpolicy={q5000['mean_q_demo_minus_policy']:.6f}",
                                    f"first contact step={d1['FIRST_CONTACT_STEP']}, mode={d1['FAILURE_BEHAVIOR_CLASS']}",
                                    "actor and critic parameter trees both changed and evaluate from exact restored checkpoints"],
                      AXIAL_ADVANCE_BEFORE_ALIGNMENT=d1["AXIAL_ADVANCE_BEFORE_ALIGNMENT"],
                      CONTACT_FAILURE_MODE=d1["FAILURE_BEHAVIOR_CLASS"], DEMO_REPLAY_EXACT="PASS",
                      POLICY_DEMO_ALIGNMENT_TREND=trend, CRITIC_PREFERENCE_TREND=d4["CRITIC_TREND"],
                      NEXT=next_map[primary])
    except Exception as exc:
        result["first_failed_gate"] = repr(exc); result["traceback"] = traceback.format_exc(); traceback.print_exc()
    finally:
        if env is not None: env.close()
        after = {path: sha(Path(path)) for path in protected}; write_json(out / "protected_after.json", after)
        if protected != after:
            result.update(PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC="FAIL", NEXT="BLOCKED",
                          PROTECTED_ARTIFACTS_CHANGED="YES", first_failed_gate="protected artifact changed")
        write_json(out / "validation.json", result)
        (out / "REPORT.md").write_text("# Peg F3 V2 Post-5K Root-Cause Diagnostic\n\n```json\n" +
                                             json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
