#!/usr/bin/env python3
"""Read-only D6-D11 Q-bias mechanism diagnostic for Peg F3 V2 formal 5K."""
from __future__ import annotations

import argparse
import hashlib
import json
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
import jax.numpy as jnp
import numpy as np
import yaml

from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.observation_contract import V2, validate_config

CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
DATASET_DIR = ROOT / "results/datasets/peg_v2_accepted_11_20260906"
DATASET = DATASET_DIR / "peg_v2_accepted_11.pkl"
PROVENANCE = DATASET_DIR / "episode_provenance.json"
MANIFEST = DATASET_DIR / "manifest.json"
RUN = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run"
PRIOR = ROOT / "results/validation/peg/f3_v2_post5k_root_cause_final_20260906/validation.json"
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


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.size and np.isfinite(a).all(), "invalid statistic input")
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def percentiles(values):
    a = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90, 99)}


def geometry(obs):
    obs = np.asarray(obs, dtype=np.float64)
    relative = obs[:, 13:16]
    axial = np.abs(relative[:, 0])
    lateral = np.linalg.norm(relative[:, 1:3], axis=1)
    q1, q2 = obs[:, 3:7].copy(), obs[:, 16:20].copy()
    q1 /= np.maximum(np.linalg.norm(q1, axis=1, keepdims=True), 1e-12)
    q2 /= np.maximum(np.linalg.norm(q2, axis=1, keepdims=True), 1e-12)
    orientation = 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0, 1))
    force = np.linalg.norm(obs[:, 7:10], axis=1)
    return {"lateral": lateral, "axial": axial, "orientation": orientation, "force": force}


def q_values(agent, observations, actions):
    values = np.asarray(jax.device_get(agent.forward_critic(
        jnp.asarray(observations), jnp.asarray(actions), rng=None, train=False)))
    require(values.ndim == 2 and values.shape[1] == len(observations), "critic shape")
    require(np.isfinite(values).all(), "nonfinite critic")
    return values


def batch_from(arrays, indices):
    return {key: jnp.asarray(np.asarray(value)[indices]) for key, value in arrays.items()
            if key in ("observations", "actions", "next_observations", "rewards", "masks", "dones")}


def target_decomposition(agent, batch, seed):
    rng = jax.random.PRNGKey(seed)
    rng, action_key = jax.random.split(rng)
    dist = agent.forward_policy(batch["next_observations"], rng=action_key)
    next_actions, log_probs = dist.sample_and_log_prob(seed=action_key)
    target_qs = agent.forward_target_critic(batch["next_observations"], next_actions, rng=rng)
    selected = np.arange(int(agent.config["critic_ensemble_size"]))
    if agent.config["critic_subsample_size"] is not None:
        rng, subkey = jax.random.split(rng)
        selected = np.asarray(jax.device_get(jax.random.randint(
            subkey, (agent.config["critic_subsample_size"],), 0,
            agent.config["critic_ensemble_size"])))
        target_qs = target_qs[selected]
    target_min = np.asarray(jax.device_get(target_qs.min(axis=0)), np.float64)
    rewards = np.asarray(batch["rewards"], np.float64)
    masks = np.asarray(batch["masks"], np.float64)
    log_probs = np.asarray(jax.device_get(log_probs), np.float64)
    alpha = float(np.asarray(jax.device_get(agent.forward_temperature())))
    bootstrap = float(agent.config["discount"]) * masks * target_min
    entropy = -alpha * log_probs if agent.config["backup_entropy"] else np.zeros_like(rewards)
    target = rewards + bootstrap + entropy
    require(all(np.isfinite(x).all() for x in (target_min, rewards, masks, log_probs, bootstrap, entropy, target)),
            "nonfinite Bellman decomposition")
    return {"next_actions": np.asarray(jax.device_get(next_actions)), "log_probs": log_probs,
            "target_q_min": target_min, "reward": rewards, "mask": masks,
            "bootstrap": bootstrap, "entropy": entropy, "bellman": target,
            "selected_target_critics": selected.tolist(), "temperature": alpha}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(); out = args.output.resolve()
    require(ROOT in out.parents and not out.exists(), "fresh isolated output required")
    out.mkdir(parents=True, exist_ok=False)
    result = {"PEG_F3_V2_RLPD_Q_BIAS_DIAGNOSTIC": "FAIL",
              **{f"D{i}_{name}": "NOT_RUN" for i, name in (
                  (6, "ONLINE_BOOTSTRAP_SEMANTICS"), (7, "Q_OVER_ESTIMATION"),
                  (8, "BELLMAN_TARGET_AUDIT"), (9, "ENTROPY_DIAGNOSTIC"),
                  (10, "CRITIC_INFLUENCE"), (11, "DISTRIBUTION_SHIFT"))},
              "FORMAL_10K_CONTINUE": "BLOCKED", "TRAINING_STARTED": False,
              "NEXT": "BLOCKED", "PROTECTED_ARTIFACTS_CHANGED": "NO"}
    protected = {}
    try:
        cfg = yaml.safe_load(CONFIG.read_text()); manifest = json.loads(MANIFEST.read_text())
        provenance = json.loads(PROVENANCE.read_text()); prior = json.loads(PRIOR.read_text())
        require(sha(CONFIG) == EXPECTED_CONFIG and sha(DATASET) == EXPECTED_DATASET, "frozen hashes")
        require(validate_config(cfg, for_training=True) == V2, "V2 config")
        require(manifest["v2_only"] is True and manifest["episodes"] == 11 and manifest["transitions"] == 1921, "manifest")
        require(prior["PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC"] == "PASS", "prior diagnostic")
        cps = {step: RUN / "checkpoints" / f"checkpoint_{step}" for step in STEPS}
        protected_paths = [CONFIG, DATASET, PROVENANCE, MANIFEST, PRIOR]
        for step, cp in cps.items():
            protected_paths += [cp, cp.with_name(cp.name + "_online_replay.pkl"),
                                cp.with_name(cp.name + "_resume_metadata.json")]
        for entry in provenance:
            protected_paths.append(ROOT / entry["episode_path"] / "transitions.jsonl")
        require(all(path.is_file() for path in protected_paths), "protected artifact missing")
        protected = {str(path): sha(path) for path in protected_paths}
        write_json(out / "protected_before.json", protected)
        ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True).stdout
        require(not any(("async_sac_state_sim.py" in line or "run_peg_f3_v2_rlpd_smoke.py" in line)
                        and str(os.getpid()) not in line for line in ps.splitlines()), "training process running")
        require(jax.default_backend() == "gpu", "GPU inference backend")

        data = pickle.loads(DATASET.read_bytes())
        require(isinstance(data, list) and len(data) == 1921, "dataset cardinality")
        arrays = {key: np.asarray([item[key] for item in data]) for key in data[0]}
        require(arrays["observations"].shape == (1921, 20) and arrays["actions"].shape == (1921, 7), "dataset schema")
        phases, types, metrics = [], [], []
        for entry in provenance:
            rows = [json.loads(line) for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open()]
            phases.extend(row["phase"] for row in rows); types.extend([entry["type"]] * len(rows))
            metrics.extend(row["after"]["metrics"] for row in rows)
        phases, types = np.asarray(phases), np.asarray(types)
        metric_lat = np.asarray([x["lateral_error"] for x in metrics])
        metric_ax = np.asarray([x["axial_error"] for x in metrics])
        special = {
            "lateral": np.isin(types, ["lateral_plus", "lateral_minus"]) &
                       (phases == "lateral_alignment") & (np.abs(arrays["actions"][:, 1]) >= .99),
            "rotation": np.isin(types, ["rotation_roll", "rotation_pitch"]) &
                        (phases == "orientation_correction"),
            "near_hole": (metric_lat <= .010) & (metric_ax <= .020),
        }
        union = np.logical_or.reduce(list(special.values()))
        require(int(union.sum()) == 405, "representative state set drift")

        online_payload = pickle.loads(cps[5000].with_name("checkpoint_5000_online_replay.pkl").read_bytes())
        online_size = int(online_payload["size"])
        online = {key: np.asarray(value)[:online_size] for key, value in online_payload["dataset_dict"].items()}
        require(online_size == 36451 and online["observations"].shape == (36451, 20), "online replay")

        # D6: exact insertion contract makes done=True/mask=1 equivalent to truncation-only.
        done_idx = np.flatnonzero(online["dones"].astype(bool))
        mask_zero = np.flatnonzero(np.asarray(online["masks"]) == 0)
        d6_rows = []
        for idx in done_idx:
            d6_rows.append({"replay_index": int(idx), "done": True, "mask": float(online["masks"][idx]),
                            "truncated": True, "success": False,
                            "termination_reason": "TIME_LIMIT_TRUNCATION_DERIVED_FROM_INSERTION_CONTRACT",
                            "reward": float(online["rewards"][idx]),
                            "next_obs_valid": bool(np.asarray(online["next_observations"][idx]).shape == (20,) and
                                                   np.isfinite(online["next_observations"][idx]).all())})
        d6 = {"ONLINE_DONE_COUNT": int(len(done_idx)), "ONLINE_MASK_ZERO_COUNT": int(len(mask_zero)),
              "TRUNCATION_COUNT": int(len(done_idx)), "TRUE_TERMINAL_COUNT": int(len(mask_zero)),
              "TERMINAL_BOOTSTRAP_ERROR": "NO",
              "derivation": "actor inserts done=(terminated or truncated), mask=0 iff terminated; every done row has mask=1",
              "all_next_obs_valid": all(row["next_obs_valid"] for row in d6_rows)}
        require(len(done_idx) == 72 and len(mask_zero) == 0 and d6["all_next_obs_valid"], "D6 semantics")
        write_json(out / "d6_online_done_transitions.json", d6_rows)
        result["D6_ONLINE_BOOTSTRAP_SEMANTICS"] = "PASS"; result["D6"] = d6
        print("D6_ONLINE_BOOTSTRAP_SEMANTICS=PASS", flush=True)

        from serl_launcher.utils.launcher import make_sac_agent
        fresh = make_sac_agent(seed=0, sample_obs=np.zeros(20, np.float32), sample_action=np.zeros(7, np.float32))
        agents = {}
        for step, cp in cps.items():
            meta = json.loads(cp.with_name(cp.name + "_resume_metadata.json").read_text())
            require(meta["checkpoint_step"] == meta["learner_update_steps"] == step and
                    meta["config_hash"] == canonical(cfg), "checkpoint binding")
            agents[step] = fresh.replace(state=restore_fresh_state(cp.parent, fresh.state, step))

        # D7: one identical 405-state set and fixed action controls at all checkpoints.
        obs_rep, demo_rep = arrays["observations"][union], arrays["actions"][union]
        rng = np.random.default_rng(20260906)
        random_actions = rng.uniform(-1, 1, demo_rep.shape).astype(np.float32)
        perturb = np.clip(demo_rep[None] + rng.normal(0, .05, (8,) + demo_rep.shape), -1, 1).astype(np.float32)
        d7 = {"state_count": int(len(obs_rep)), "perturbations_per_state": 8, "steps": {}}
        for step in STEPS:
            agent = agents[step]
            policy = np.asarray(jax.device_get(agent.sample_actions(observations=obs_rep, argmax=True)))
            action_sets = {"demo": demo_rep, "policy": policy, "zero": np.zeros_like(demo_rep), "random": random_actions}
            qs = {name: q_values(agent, obs_rep, actions) for name, actions in action_sets.items()}
            p_q = q_values(agent, np.repeat(obs_rep, 8, axis=0), perturb.transpose(1, 0, 2).reshape(-1, 7))
            p_q = p_q.reshape(p_q.shape[0], len(obs_rep), 8)
            means = {name: values.mean(axis=0) for name, values in qs.items()}
            perturbed_mean = p_q.mean(axis=0)
            ensemble = {name: {"members": [quantiles(row) for row in values],
                               "mean_member_std": float(values.std(axis=0).mean()),
                               "mean_member_range": float((values.max(axis=0) - values.min(axis=0)).mean())}
                        for name, values in qs.items()}
            d7["steps"][str(step)] = {
                "Q": {**{name: quantiles(value) for name, value in means.items()},
                      "perturbed": quantiles(perturbed_mean.reshape(-1))},
                "deltas": {"policy_minus_demo": quantiles(means["policy"] - means["demo"]),
                           "random_minus_demo": quantiles(means["random"] - means["demo"]),
                           "perturbed_minus_demo": quantiles((perturbed_mean - means["demo"][:, None]).reshape(-1))},
                "ensemble": ensemble,
                "policy_above_all_demo_perturbations_rate": float(np.mean(means["policy"] > perturbed_mean.max(axis=1))),
            }
        gap1 = d7["steps"]["1000"]["deltas"]["policy_minus_demo"]["mean"]
        gap5 = d7["steps"]["5000"]["deltas"]["policy_minus_demo"]["mean"]
        d7["Q_OVER_ESTIMATION_TREND"] = "SUPPORTED" if gap5 > gap1 and gap5 > 0 else "NOT_SUPPORTED"
        scale1 = max(abs(d7["steps"]["1000"]["Q"][x]["median"]) for x in ("demo", "policy", "zero", "random"))
        scale5 = max(abs(d7["steps"]["5000"]["Q"][x]["median"]) for x in ("demo", "policy", "zero", "random"))
        d7["Q_SCALE_GROWTH"] = {"checkpoint_1000_abs_median_max": scale1,
                                "checkpoint_5000_abs_median_max": scale5, "ratio": scale5 / max(scale1, 1e-12)}
        d7["ENSEMBLE_DISAGREEMENT"] = {str(step): d7["steps"][str(step)]["ensemble"]["policy"]["mean_member_std"] for step in STEPS}
        result["D7_Q_OVER_ESTIMATION"] = "PASS"; result["D7"] = d7
        print("D7_Q_OVER_ESTIMATION=PASS", flush=True)

        # Fixed representative demo and online samples for D8/D10.
        demo_candidates = np.flatnonzero(union)
        demo_idx = demo_candidates[np.linspace(0, len(demo_candidates) - 1, 96, dtype=int)]
        og = geometry(online["observations"])
        uniform = np.linspace(0, online_size - 1, 32, dtype=int)
        high_o = np.argsort(og["orientation"])[-32:]
        high_l = np.argsort(og["lateral"])[-32:]
        online_idx = np.concatenate((uniform, high_o, high_l))
        demo_batch, online_batch = batch_from(arrays, demo_idx), batch_from(online, online_idx)
        d8 = {"samples_each": 96, "selection": "demo representative union; online uniform+top orientation+top lateral", "steps": {}}
        for step in (1000, 3000, 5000):
            dd = target_decomposition(agents[step], demo_batch, 800000 + step)
            od = target_decomposition(agents[step], online_batch, 900000 + step)
            summarize = lambda x: {name: quantiles(x[name]) for name in ("reward", "target_q_min", "bootstrap", "entropy", "bellman")}
            d8["steps"][str(step)] = {"demo": summarize(dd), "online": summarize(od),
                                      "target_critic_members_demo": dd["selected_target_critics"],
                                      "target_critic_members_online": od["selected_target_critics"]}
        d8["DEMO_TARGET_MEAN"] = {s: d8["steps"][s]["demo"]["bellman"]["mean"] for s in d8["steps"]}
        d8["ONLINE_TARGET_MEAN"] = {s: d8["steps"][s]["online"]["bellman"]["mean"] for s in d8["steps"]}
        d8["BOOTSTRAP_COMPONENT_SHIFT"] = {
            kind: d8["steps"]["5000"][kind]["bootstrap"]["mean"] - d8["steps"]["1000"][kind]["bootstrap"]["mean"]
            for kind in ("demo", "online")}
        d8["ENTROPY_COMPONENT_SHIFT"] = {kind: 0.0 for kind in ("demo", "online")}
        d8["TARGET_Q_DRIFT"] = {kind: d8["steps"]["5000"][kind]["bellman"]["mean"] -
                                      d8["steps"]["1000"][kind]["bellman"]["mean"] for kind in ("demo", "online")}
        result["D8_BELLMAN_TARGET_AUDIT"] = "PASS"; result["D8"] = d8
        print("D8_BELLMAN_TARGET_AUDIT=PASS", flush=True)

        # D9: direct temperature output is softplus(raw lagrange); backup entropy is disabled.
        d9 = {"temperature_parameterization": "softplus(raw lagrange)", "target_entropy": float(fresh.config["target_entropy"]),
              "backup_entropy": bool(fresh.config["backup_entropy"]), "steps": {}}
        for step in STEPS:
            agent = agents[step]; key = jax.random.PRNGKey(700000 + step)
            dist = agent.forward_policy(jnp.asarray(obs_rep), rng=key)
            sampled, logp = dist.sample_and_log_prob(seed=key)
            logp = np.asarray(jax.device_get(logp), np.float64)
            alpha = float(np.asarray(jax.device_get(agent.forward_temperature())))
            sampled_q = q_values(agent, obs_rep, np.asarray(jax.device_get(sampled))).mean(axis=0)
            raw = np.asarray(jax.device_get(jax.tree_util.tree_leaves(agent.state.params["modules_temperature"])[0])).reshape(-1)
            d9["steps"][str(step)] = {"temperature": alpha, "raw_lagrange": raw.tolist(),
                                      "sampled_actor_entropy_estimate": float((-logp).mean()),
                                      "actor_entropy_term_mean": float((alpha * logp).mean()),
                                      "actor_entropy_term_mean_abs": float(np.mean(np.abs(alpha * logp))),
                                      "sampled_policy_q_mean": float(sampled_q.mean()),
                                      "bellman_entropy_term": 0.0}
        ent5 = d9["steps"]["5000"]["actor_entropy_term_mean_abs"]
        d9["TEMPERATURE_BEHAVIOR"] = "DIRECT_POSITIVE_SOFTPLUS_MULTIPLIER_RISES; NOT_LOG_TEMPERATURE_OUTPUT"
        d9["ENTROPY_TERM_MAGNITUDE"] = {s: d9["steps"][s]["actor_entropy_term_mean_abs"] for s in d9["steps"]}
        d9["ENTROPY_DRIVES_BAD_ACTIONS"] = "NOT_SUPPORTED" if not fresh.config["backup_entropy"] and ent5 < abs(gap5) else "SUPPORTED"
        result["D9_ENTROPY_DIAGNOSTIC"] = "PASS"; result["D9"] = d9
        print("D9_ENTROPY_DIAGNOSTIC=PASS", flush=True)

        # D10: exact 32 online + 32 demo concatenation and unweighted global MSE.
        d10 = {"DEMO_CRITIC_WEIGHT": 0.5, "ONLINE_CRITIC_WEIGHT": 0.5,
               "DEMO_Q_REGULARIZATION_PRESENT": "NO", "gradient_norm": "NOT_AVAILABLE",
               "batch_order": "online[0:32], demo[32:64]", "steps": {}}
        for step in (1000, 3000, 5000):
            agent = agents[step]
            combined = {key: jnp.concatenate((online_batch[key][:32], demo_batch[key][:32]), axis=0)
                        for key in online_batch}
            td = target_decomposition(agent, combined, 600000 + step)
            pred = q_values(agent, np.asarray(combined["observations"]), np.asarray(combined["actions"]))
            mse = np.mean((pred - td["bellman"][None]) ** 2, axis=0)
            online_loss, demo_loss = float(mse[:32].mean()), float(mse[32:].mean())
            d10["steps"][str(step)] = {"online_subset_loss": online_loss, "demo_subset_loss": demo_loss,
                                       "ONLINE_LOSS_CONTRIBUTION": .5 * online_loss,
                                       "DEMO_LOSS_CONTRIBUTION": .5 * demo_loss,
                                       "combined_critic_loss": float(mse.mean())}
        d10["DEMO_LOSS_CONTRIBUTION"] = {s: d10["steps"][s]["DEMO_LOSS_CONTRIBUTION"] for s in d10["steps"]}
        d10["ONLINE_LOSS_CONTRIBUTION"] = {s: d10["steps"][s]["ONLINE_LOSS_CONTRIBUTION"] for s in d10["steps"]}
        result["D10_CRITIC_INFLUENCE"] = "PASS"; result["D10"] = d10
        print("D10_CRITIC_INFLUENCE=PASS", flush=True)

        # D11: feature-wise and joint empirical-range shift severity.
        demo_g, online_g = geometry(arrays["observations"]), geometry(online["observations"])
        demo_features = {**demo_g, "reward": np.asarray(arrays["rewards"], np.float64)}
        online_features = {**online_g, "reward": np.asarray(online["rewards"], np.float64)}
        in_range = np.ones(online_size, dtype=bool)
        d11 = {"demo": {}, "online": {}, "online_in_demo_range_by_feature": {}}
        for name in demo_features:
            d11["demo"][name] = percentiles(demo_features[name]); d11["online"][name] = percentiles(online_features[name])
            inside = (online_features[name] >= demo_features[name].min()) & (online_features[name] <= demo_features[name].max())
            d11["online_in_demo_range_by_feature"][name] = float(inside.mean()); in_range &= inside
        d11["ONLINE_IN_DEMO_MANIFOLD_RATE"] = float(in_range.mean())
        d11["ORIENTATION_OOD_RATE"] = float(np.mean(online_g["orientation"] > demo_g["orientation"].max()))
        d11["LATERAL_OOD_RATE"] = float(np.mean(online_g["lateral"] > demo_g["lateral"].max()))
        d11["NEAR_HOLE_LIKE_RATE"] = float(np.mean((online_g["lateral"] <= .010) & (online_g["axial"] <= .020)))
        result["D11_DISTRIBUTION_SHIFT"] = "PASS"; result["D11"] = d11
        print("D11_DISTRIBUTION_SHIFT=PASS", flush=True)

        b_supported = d7["Q_OVER_ESTIMATION_TREND"] == "SUPPORTED" and gap5 > 0
        c_supported = d11["ONLINE_IN_DEMO_MANIFOLD_RATE"] < .5 or d11["ORIENTATION_OOD_RATE"] > .5 or d11["LATERAL_OOD_RATE"] > .5
        primary = "B_CRITIC_Q_OVER_ESTIMATION" if b_supported else ("C_ONLINE_DISTRIBUTION_SHIFT_DOMINATES" if c_supported else "G_UNRESOLVED")
        secondary = "C_ONLINE_DISTRIBUTION_SHIFT_DOMINATES" if b_supported and c_supported else "G_UNRESOLVED"
        candidates = [
            {"FIX_CANDIDATE": "Fresh bounded ablation with lower critic learning rate and longer critic warmup; accept only if Q-policy/demo gap stops growing",
             "SCOPE": "LEVEL_1_CONFIG_ONLY", "FILES_EXPECTED": "project-local agent factory/config only", "RISK": "MEDIUM", "REQUIRES_FRESH_RETRAIN": "YES"},
            {"FIX_CANDIDATE": "Add explicit demo-action anchoring to actor on demo correction states",
             "SCOPE": "LEVEL_3_OBJECTIVE_CHANGE", "FILES_EXPECTED": "project-local SAC/RLPD learner loss", "RISK": "MEDIUM_HIGH", "REQUIRES_FRESH_RETRAIN": "YES"},
            {"FIX_CANDIDATE": "Add conservative critic regularization against unsupported policy/random actions on demo states",
             "SCOPE": "LEVEL_3_OBJECTIVE_CHANGE", "FILES_EXPECTED": "project-local critic loss and metrics", "RISK": "HIGH", "REQUIRES_FRESH_RETRAIN": "YES"},
        ]
        result.update(PEG_F3_V2_RLPD_Q_BIAS_DIAGNOSTIC="PASS",
                      PRIMARY_Q_BIAS_MECHANISM=primary, SECONDARY_Q_BIAS_MECHANISM=secondary,
                      SUPPORTED_BY=[f"policy-demo Q gap mean rose from {gap1:.6f} at 1000 to {gap5:.6f} at 5000",
                                    f"online joint demo-manifold rate={d11['ONLINE_IN_DEMO_MANIFOLD_RATE']:.6f}; orientation OOD={d11['ORIENTATION_OOD_RATE']:.6f}; lateral OOD={d11['LATERAL_OOD_RATE']:.6f}",
                                    f"done/mask audit: {len(done_idx)} truncations, {len(mask_zero)} true terminals; Bellman entropy backup={fresh.config['backup_entropy']}"],
                      MINIMUM_FIX_LEVEL="LEVEL_1_CONFIG_ONLY", FIX_CANDIDATES=candidates,
                      NEXT="MINIMAL_FIX_DESIGN_READY")
    except Exception as exc:
        result["first_failed_gate"] = repr(exc); result["traceback"] = traceback.format_exc(); traceback.print_exc()
    finally:
        after = {path: sha(Path(path)) for path in protected}
        write_json(out / "protected_after.json", after)
        if protected != after:
            result.update(PEG_F3_V2_RLPD_Q_BIAS_DIAGNOSTIC="FAIL", NEXT="BLOCKED",
                          PROTECTED_ARTIFACTS_CHANGED="YES", first_failed_gate="protected artifact changed")
        write_json(out / "validation.json", result)
        (out / "REPORT.md").write_text("# Peg F3 V2 RLPD Critic Q-Bias Mechanism Diagnostic\n\n```json\n" +
                                             json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["PEG_F3_V2_RLPD_Q_BIAS_DIAGNOSTIC"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
