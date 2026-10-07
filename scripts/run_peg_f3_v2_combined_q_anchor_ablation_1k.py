#!/usr/bin/env python3
"""Authorized fresh 1K conservative-Q plus Demo-anchor ablation on the proven F3 harness."""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_peg_f3_v2_qbias_ablation_1k as shared

CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
EXPECTED_CONFIG = "b1cd4126c10673c267e0e2382a44eef97cfc7b0611a5774568d46ab6d8f2ff41"
OUTPUT = ROOT / "results/ablation/peg_f3_v2_combined_q_anchor_ablation_1k_seed0_run_001"
RUN_ID = "wandb-disabled-peg-f3-v2-combined-q-anchor-ablation-1k"
BASE_Q_GAP = 1.578540
BASE_MAE = 0.713376
BASE_LAT_MAE = 0.734863
BASE_A1_MAE = 1.015701
DESIGN_GRAD_RATIO = 0.09475306297110436


def sha(path):
    return shared.sha(path)


def require(value, message):
    return shared.require(value, message)


class VirtualConfig(dict):
    _smoke = {"learner_updates": 1000, "checkpoint_period": 500,
              "actor_max_steps": 1000000, "process_timeout_seconds": 7200,
              "update_timeout_seconds": 120}

    def __getitem__(self, key):
        if key == "smoke" and key not in self:
            return self._smoke
        return super().__getitem__(key)


def protected_snapshot():
    """Hash protected inputs and earlier results without touching them."""
    result = shared.protected_snapshot()
    roots = [
        ROOT / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py",
        ROOT / "configs/peg_insertion/peg_v2_qbias_ablation_1k_seed0.yaml",
        ROOT / "training/peg_insertion/conservative_q_ablation.py",
        ROOT / "training/peg_insertion/combined_q_anchor_ablation.py",
        CONFIG_PATH,
    ]
    prior = ROOT / "results/ablation/peg_f3_v2_qbias_ablation_1k_seed0_retry_002"
    if prior.exists():
        roots.extend(path for path in prior.rglob("*") if path.is_file())
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    for entry in provenance:
        episode = ROOT / entry["episode_path"]
        roots.extend(path for path in episode.rglob("*") if path.is_file())
    require(all(path.is_file() for path in roots), "protected artifact missing")
    result.update({str(path): sha(path) for path in roots})
    return result


def runtime_binding_gate(base):
    old_config, old_expected = shared.CONFIG_PATH, shared.EXPECTED_CONFIG
    try:
        shared.CONFIG_PATH, shared.EXPECTED_CONFIG = CONFIG_PATH, EXPECTED_CONFIG
        result = shared.runtime_binding_gate(base)
    finally:
        shared.CONFIG_PATH, shared.EXPECTED_CONFIG = old_config, old_expected
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path[:0]=[{str(ROOT)!r}]; "
        "from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent; "
        "print('PASS')"
    )
    check = subprocess.run([str(Path('/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python')), "-c", code],
                           cwd=ROOT, env=base.runtime_env(), text=True, capture_output=True, timeout=120)
    require(check.returncode == 0 and check.stdout.strip().splitlines()[-1] == "PASS",
            "Combined Q-anchor factory runtime import")
    result.pop("QBIAS_FACTORY_IMPORT", None)
    result["COMBINED_Q_ANCHOR_FACTORY_IMPORT"] = "PASS"
    return result


def production_defaults():
    """Resolve formal defaults from the exact production factory/signatures."""
    from serl_launcher.agents.continuous.sac import SACAgent
    from serl_launcher.utils.launcher import make_sac_agent

    create = inspect.signature(SACAgent.create).parameters
    factory = inspect.signature(make_sac_agent).parameters
    actor_opt = create["actor_optimizer_kwargs"].default
    critic_opt = create["critic_optimizer_kwargs"].default
    temp_opt = create["temperature_optimizer_kwargs"].default
    for value, name in ((actor_opt, "actor optimizer"), (critic_opt, "critic optimizer"),
                        (temp_opt, "temperature optimizer")):
        require(isinstance(value, dict), f"unknown production default: {name}")
    source = dedent(inspect.getsource(make_sac_agent))
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             getattr(node.func, "attr", None) == "create_states"]
    require(len(calls) == 1, "production factory create_states call not provable")
    explicit = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords
                if item.arg in ("backup_entropy", "critic_ensemble_size", "critic_subsample_size")}
    require(explicit == {"backup_entropy": False, "critic_ensemble_size": 10,
                         "critic_subsample_size": 2}, "production factory defaults not provable")
    return {
        "actor_learning_rate": actor_opt["learning_rate"],
        "critic_learning_rate": critic_opt["learning_rate"],
        "temperature_learning_rate": temp_opt["learning_rate"],
        "actor_warmup_steps": actor_opt["warmup_steps"],
        "critic_warmup_steps": critic_opt["warmup_steps"],
        "discount": factory["discount"].default,
        "soft_target_update_rate": create["soft_target_update_rate"].default,
        "backup_entropy": False,
        "critic_ensemble_size": 10,
        "critic_subsample_size": 2,
        "sources": {
            "optimizer": "SACAgent.create signature defaults",
            "discount": "serl_launcher.utils.launcher.make_sac_agent signature",
            "algorithm": "make_sac_agent explicit kwargs plus SACAgent.create signature",
        },
    }


def resolve_effective_config(cfg, defaults=None):
    defaults = production_defaults() if defaults is None else defaults
    required_defaults = ("actor_learning_rate", "critic_learning_rate", "temperature_learning_rate",
                         "actor_warmup_steps", "critic_warmup_steps", "discount",
                         "soft_target_update_rate", "backup_entropy", "critic_ensemble_size",
                         "critic_subsample_size")
    require(all(key in defaults for key in required_defaults), "unknown production default")
    optimizer = cfg.get("optimizer", {})
    algorithm = cfg.get("algorithm_defaults", {})
    training = cfg.get("training", {})
    require(all(key in training for key in ("online_batch_size", "demo_batch_size", "total_batch_size",
                                             "critic_actor_ratio", "utd_ratio", "training_starts",
                                             "random_steps", "replay_capacity")), "unknown training field")
    online, demo = int(training["online_batch_size"]), int(training["demo_batch_size"])
    require(online + demo == int(training["total_batch_size"]), "batch composition")
    return {
        "actor_lr": optimizer.get("actor_learning_rate", defaults["actor_learning_rate"]),
        "critic_lr": optimizer.get("critic_learning_rate", defaults["critic_learning_rate"]),
        "temperature_lr": optimizer.get("temperature_learning_rate", defaults["temperature_learning_rate"]),
        "actor_warmup": optimizer.get("actor_warmup_steps", defaults["actor_warmup_steps"]),
        "critic_warmup": optimizer.get("critic_warmup_steps", defaults["critic_warmup_steps"]),
        "random_steps": int(training["random_steps"]),
        "learning_starts": int(training["training_starts"]),
        "target_update_rate": algorithm.get("soft_target_update_rate", defaults["soft_target_update_rate"]),
        "discount": algorithm.get("discount", defaults["discount"]),
        "backup_entropy": algorithm.get("backup_entropy", defaults["backup_entropy"]),
        "batch": int(training["total_batch_size"]), "utd": int(training["utd_ratio"]),
        "demo_online": (demo, online), "critic_actor_ratio": int(training["critic_actor_ratio"]),
        "replay_capacity": int(training["replay_capacity"]),
        "critic_ensemble_size": algorithm.get("critic_ensemble_size", defaults["critic_ensemble_size"]),
        "critic_subsample_size": algorithm.get("critic_subsample_size", defaults["critic_subsample_size"]),
    }


def effective_config_regression(baseline, raw, defaults):
    expected = resolve_effective_config(baseline, defaults)
    explicit_same = copy.deepcopy(baseline)
    explicit_same["optimizer"] = {
        "actor_learning_rate": defaults["actor_learning_rate"],
        "critic_learning_rate": defaults["critic_learning_rate"],
        "temperature_learning_rate": defaults["temperature_learning_rate"],
        "actor_warmup_steps": defaults["actor_warmup_steps"],
        "critic_warmup_steps": defaults["critic_warmup_steps"],
    }
    require(resolve_effective_config(explicit_same, defaults) == expected,
            "explicit same-default representation regression")
    cases = (("critic_lr", "optimizer", "critic_learning_rate", .123),
             ("actor_lr", "optimizer", "actor_learning_rate", .123),
             ("temperature_lr", "optimizer", "temperature_learning_rate", .123),
             ("batch", "training", "total_batch_size", 65),
             ("utd", "training", "utd_ratio", 2),
             ("demo_online", "training", "demo_batch_size", 31))
    rejected = {}
    for name, section, key, value in cases:
        candidate = copy.deepcopy(raw)
        candidate[section][key] = value
        try:
            actual = resolve_effective_config(candidate, defaults)
            rejected[name] = actual != expected
        except (RuntimeError, KeyError):
            rejected[name] = True
        require(rejected[name], f"regression did not reject {name}")
    missing = dict(defaults)
    missing.pop("critic_learning_rate")
    try:
        resolve_effective_config(baseline, missing)
        unknown_rejected = False
    except RuntimeError:
        unknown_rejected = True
    require(unknown_rejected, "unknown default fail-closed regression")
    return {"EXPLICIT_SAME_DEFAULT_ACCEPTANCE": "PASS", "REAL_DRIFT_REJECTIONS": rejected,
            "UNKNOWN_DEFAULT_REJECTION": "PASS"}


def config_semantics_gate(raw, defaults=None):
    baseline = __import__("yaml").safe_load(
        (ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml").read_text())
    for section in ("environment", "algorithm", "seed", "fresh_initialization",
                    "load_legacy_checkpoint", "demo", "observation", "action", "contracts"):
        require(raw.get(section) == baseline.get(section), f"baseline semantic drift: {section}")
    defaults = production_defaults() if defaults is None else defaults
    formal_effective = resolve_effective_config(baseline, defaults)
    ablation_effective = resolve_effective_config(raw, defaults)
    require(formal_effective == ablation_effective, "effective baseline semantic drift")
    regression = effective_config_regression(baseline, raw, defaults)
    conservative = raw.get("conservative_q", {})
    require(conservative.get("enabled") is True and conservative.get("scope") == "demo_tail", "conservative scope")
    require(conservative.get("demo_batch_size") == 32 and conservative.get("selected_lambda") == 5.0,
            "conservative size/lambda")
    require(conservative.get("margin") == 0.0, "conservative margin")
    require(conservative.get("loss") == "mean_relu_q_policy_minus_q_demo_plus_margin", "conservative loss")
    anchor = raw.get("demo_actor_anchor", {})
    require(anchor.get("enabled") is True and anchor.get("scope") == "all_demo", "anchor scope")
    require(anchor.get("demo_batch_size") == 32 and anchor.get("selected_lambda") == .07,
            "anchor size/lambda")
    require(anchor.get("loss") == "mse_policy_mode_to_demo_action", "anchor loss")
    require(raw["ablation"]["semantic_diff_fields"] == ["conservative_q", "demo_actor_anchor"],
            "semantic diff fields")
    return {
        "CONFIG_SEMANTICS": "PASS", "CONFIG_DIFF_FIELDS": ["conservative_q", "demo_actor_anchor"],
        "PEG_F3_EFFECTIVE_CONFIG_EQUIVALENCE_GATE_FIX": "PASS",
        "FORMAL_DEFAULT_RESOLUTION": "PASS", "EFFECTIVE_CONFIG_RESOLVER": "PASS",
        "RAW_REPRESENTATION_DIFFERENCE": "YES",
        "EFFECTIVE_BASELINE_SEMANTICS_DIFFERENCE": "NO", "ALLOWED_ABLATION_DIFFS_ONLY": "PASS",
        "FORMAL_EFFECTIVE_CONFIG": formal_effective, "ABLATION_EFFECTIVE_CONFIG": ablation_effective,
        "DEFAULT_SOURCES": defaults["sources"], "REGRESSION_MATRIX": regression,
        "ACTOR_LR": ablation_effective["actor_lr"],
        "CRITIC_LR": ablation_effective["critic_lr"],
        "TEMPERATURE_LR": ablation_effective["temperature_lr"],
        "ACTOR_WARMUP": ablation_effective["actor_warmup"],
        "CRITIC_WARMUP": ablation_effective["critic_warmup"],
        "DEMO_ONLINE_RATIO": "32:32", "BATCH": 64, "UTD": 1,
        "CONSERVATIVE_Q_SCOPE": "DEMO_TAIL", "CONSERVATIVE_Q_LAMBDA": 5.0,
        "CONSERVATIVE_Q_MARGIN": 0.0, "DEMO_ANCHOR_ENABLED": "YES",
        "ANCHOR_SCOPE": "ALL_DEMO", "ANCHOR_LAMBDA": .07,
    }


def demo_tail_identity_gate(base, batches=8):
    import numpy as np
    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    from async_sac_state_sim import concat_rlpd_batches
    for index in range(batches):
        online = {
            "observations": np.full((32, 20), 1000 + index, np.float32),
            "actions": np.full((32, 7), 2000 + index, np.float32),
            "next_observations": np.full((32, 20), 3000 + index, np.float32),
            "rewards": np.full((32,), 4000 + index, np.float32),
            "masks": np.ones((32,), np.float32), "dones": np.zeros((32,), bool),
        }
        demo = {key: np.array(value, copy=True) for key, value in online.items()}
        for key in demo:
            if key not in ("masks", "dones"):
                demo[key] = demo[key] + 10000
        combined = concat_rlpd_batches(online, demo)
        for key in online:
            require(np.array_equal(np.asarray(combined[key])[:32], online[key]), f"online head {key}")
            require(np.array_equal(np.asarray(combined[key])[-32:], demo[key]), f"Demo tail {key}")
    return {"BATCH_SHAPE": 64, "ONLINE_ROWS": "batch[:32]", "DEMO_ROWS": "batch[-32:]",
            "DEMO_TAIL_BINDING": "PASS", "BATCHES_CHECKED": batches}


def install_bindings(role):
    import yaml
    import scripts.run_peg_f3_v2_rlpd_smoke as base
    from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "ablation config SHA256")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    frozen_production_defaults = production_defaults()
    config_semantics_gate(raw, frozen_production_defaults)

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_combined_q_anchor_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save, original_load = resume.save_bundle, resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {"identifier": "peg-f3-v2-combined-q-anchor-ablation-1k",
                  "profile": "isolated_combined_q_anchor_ablation", "fresh_initialization": True,
                  "resume": False, "demo_online_ratio": "32:32", "utd_ratio": 1,
                  "conservative_q_scope": "DEMO_TAIL", "conservative_q_lambda": 5.0,
                  "conservative_q_margin": 0.0,
                  "demo_actor_anchor_scope": "ALL_DEMO", "demo_actor_anchor_lambda": .07}
        return original_save(directory, state, replay, step, actor_steps, cfg, manifest,
                             RUN_ID, source, **kwargs)

    resume.save_bundle = save_bundle
    resume.load_bundle = lambda cp, cfg, manifest, run_id: original_load(cp, cfg, manifest, RUN_ID)
    base.CONFIG, base.OUTPUT, base.ACTIVE_OUTPUT = CONFIG_PATH, OUTPUT, OUTPUT
    base.RUN_UPDATES, base.CHECKPOINT_TARGETS = 1000, (500, 1000)
    base.FORMAL_5K = False
    base.SHORT_SHUTDOWN_ACCEPTANCE = False
    base.__file__ = str(Path(__file__).resolve())
    base.config = config

    original_runtime_env = base.runtime_env
    def runtime_env():
        env = original_runtime_env()
        required = [str(ROOT), str(ROOT / "external/serl/serl_launcher"),
                    str(ROOT / "external/serl/franka_sim"), str(base.ENV_PARENT)]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(required + ([existing] if existing else []))
        return env
    base.runtime_env = runtime_env
    return base, raw, frozen_production_defaults


def matched_audit(out, cfg):
    """Run the accepted 405-state/24-lateral audit without qbias-only LR telemetry."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent

    dataset = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([item[key] for item in dataset]) for key in dataset[0]}
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, types, metrics = [], [], []
    for entry in provenance:
        rows = [json.loads(line) for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open()]
        phases.extend(row["phase"] for row in rows)
        types.extend([entry["type"]] * len(rows))
        metrics.extend(row["after"]["metrics"] for row in rows)
    phases, types = np.asarray(phases), np.asarray(types)
    lateral = np.isin(types, ["lateral_plus", "lateral_minus"]) & (phases == "lateral_alignment") & (np.abs(arrays["actions"][:, 1]) >= .99)
    near = (np.asarray([x["lateral_error"] for x in metrics]) <= .010) & (np.asarray([x["axial_error"] for x in metrics]) <= .020)
    rotation = np.isin(types, ["rotation_roll", "rotation_pitch"]) & (phases == "orientation_correction")
    union = lateral | near | rotation
    require(int(union.sum()) == 405 and int(lateral.sum()) == 24, "matched state subsets")

    fresh = make_combined_q_anchor_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    state = restore_fresh_state(out / "checkpoints", fresh.state, 1000)
    agent = fresh.replace(state=state)
    obs, actions = arrays["observations"], arrays["actions"]
    policy = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
    overall_mae = float(np.mean(np.abs(policy - actions)))
    lateral_mae = float(np.mean(np.abs(policy[lateral] - actions[lateral])))
    action1_mae = float(np.mean(np.abs(policy[lateral, 1] - actions[lateral, 1])))

    def q(o, a):
        values = np.asarray(jax.device_get(agent.forward_critic(jnp.asarray(o), jnp.asarray(a), rng=None, train=False)))
        require(values.shape[0] == 10 and np.isfinite(values).all(), "critic output")
        return values, values.mean(axis=0)

    obs_u, demo_u, policy_u = obs[union], actions[union], policy[union]
    q_demo_members, q_demo = q(obs_u, demo_u)
    q_policy_members, q_policy = q(obs_u, policy_u)
    gap = q_policy - q_demo
    rng = np.random.default_rng(20260906)
    perturb = np.clip(demo_u[None] + rng.normal(0, .05, (8,) + demo_u.shape), -1, 1).astype(np.float32)
    _, q_perturb = q(np.repeat(obs_u, 8, axis=0), perturb.transpose(1, 0, 2).reshape(-1, 7))
    q_perturb = q_perturb.reshape(len(obs_u), 8)
    audit = {
        "ABLATION_Q_POLICY_MINUS_DEMO": float(gap.mean()),
        "BASELINE_Q_POLICY_MINUS_DEMO": BASE_Q_GAP,
        "DELTA_Q_BIAS": float(gap.mean() - BASE_Q_GAP),
        "ABLATION_DEMO_ACTION_MAE": overall_mae,
        "ABLATION_LATERAL_MAE": lateral_mae,
        "ABLATION_ACTION1_MAE": action1_mae,
        "Q_DEMO_GT_POLICY_RATE": float(np.mean(q_demo > q_policy)),
        "Q_POLICY_GT_DEMO_RATE": float(np.mean(q_policy > q_demo)),
        "Q_SCALE": {"demo": shared.q_stats(q_demo), "policy": shared.q_stats(q_policy),
                    "ensemble_policy_mean_std": float(q_policy_members.std(axis=0).mean())},
        "Q_PERTURBATION_RANKING": {"policy_above_all_demo_perturbations_rate": float(np.mean(q_policy > q_perturb.max(axis=1))),
                                   "perturbed_minus_demo": shared.q_stats((q_perturb - q_demo[:, None]).reshape(-1))},
    }
    (out / "matched_step_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return audit


def conservative_activity(out):
    import numpy as np
    rows = [json.loads(line) for line in (out / "updates.jsonl").open()]
    values = []
    anchor_values = []
    for row in rows:
        losses = row["losses"]
        def one(fragment):
            suffix = f"DictKey(key='{fragment}'))"
            hits = [float(v) for key, v in losses.items() if key.endswith(suffix)]
            require(len(hits) == 1, f"telemetry key {fragment}")
            return hits[0]
        values.append((int(row["update"]), one("base_critic_loss"),
                       one("conservative_q_loss"), one("weighted_conservative_q_loss"),
                       one("critic_loss")))
        anchor_values.append((one("sac_actor_loss"), one("demo_anchor_loss"),
                              one("weighted_demo_anchor_loss"), one("actor_loss")))
    a = np.asarray([item[1:] for item in values], np.float64)
    b = np.asarray(anchor_values, np.float64)
    require(len(values) == 1000 and np.isfinite(a).all(), "conservative activity trace")
    require(np.allclose(a[:, 2], 5.0 * a[:, 1], rtol=2e-5, atol=2e-6), "conservative lambda trace")
    require(np.allclose(a[:, 3], a[:, 0] + a[:, 2], rtol=2e-5, atol=2e-5), "total critic loss trace")
    require(np.isfinite(b).all(), "anchor activity trace")
    require(np.allclose(b[:, 2], .07 * b[:, 1], rtol=2e-5, atol=2e-7), "anchor lambda trace")
    require(np.allclose(b[:, 3], b[:, 0] + b[:, 2], rtol=2e-5, atol=2e-6), "total actor loss trace")
    return {"CONSERVATIVE_TERM_ACTIVE": "YES", "CONSERVATIVE_UPDATE_COUNT": len(values),
            "FIRST_CONSERVATIVE_UPDATE_STEP": values[0][0], "MEAN_BASE_CRITIC_LOSS": float(a[:, 0].mean()),
            "MEAN_CONSERVATIVE_RAW_LOSS": float(a[:, 1].mean()),
            "MEAN_CONSERVATIVE_WEIGHTED_LOSS": float(a[:, 2].mean()),
            "MEAN_TOTAL_CRITIC_LOSS": float(a[:, 3].mean()),
            "CONSERVATIVE_GRAD_RATIO": DESIGN_GRAD_RATIO,
            "CONSERVATIVE_GRAD_RATIO_SOURCE": "Stage A fixed-batch dry-run",
            "CONSERVATIVE_DOMINATES": "NO" if DESIGN_GRAD_RATIO < 0.5 else "YES",
            "ANCHOR_ACTIVE": "YES", "ANCHOR_UPDATE_COUNT": len(values),
            "MEAN_SAC_ACTOR_LOSS": float(b[:, 0].mean()),
            "MEAN_ANCHOR_RAW_LOSS": float(b[:, 1].mean()),
            "MEAN_ANCHOR_WEIGHTED_LOSS": float(b[:, 2].mean()),
            "MEAN_TOTAL_ACTOR_LOSS": float(b[:, 3].mean())}


def finalize(out, cfg, status, audit, activity):
    base_result = json.loads((out / "result.json").read_text())
    checks = base_result.get("checks", {})
    normal = all(checks.get(key, False) for key in
                 ("learner_stop", "actor_stop_observed", "trainerclient_stop", "broadcast_join",
                  "actor_natural_exit", "children_exited"))
    health = (status == 0 and base_result.get("LEARNER_UPDATES") == 1000 and
              base_result.get("GPU_BACKEND") == "gpu" and normal and
              checks.get("checkpoint_500") and checks.get("checkpoint_1000") and
              activity["CONSERVATIVE_TERM_ACTIVE"] == "YES" and
              activity["CONSERVATIVE_UPDATE_COUNT"] > 0 and
              activity["CONSERVATIVE_DOMINATES"] == "NO" and
              activity["ANCHOR_ACTIVE"] == "YES")
    qgap = audit["ABLATION_Q_POLICY_MINUS_DEMO"]
    overall = audit["ABLATION_DEMO_ACTION_MAE"]
    action1 = audit["ABLATION_ACTION1_MAE"]
    if health and overall < BASE_MAE and action1 < BASE_A1_MAE and qgap <= 0.5:
        result = "STRONG_PASS"
    elif health and overall < BASE_MAE and action1 < BASE_A1_MAE and qgap < BASE_Q_GAP:
        result = "PASS"
    else:
        result = "FAIL"
    gate = "PASS" if result in ("STRONG_PASS", "PASS") else "FAIL"
    final = {
        "PEG_F3_V2_COMBINED_Q_ANCHOR_ABLATION_1K": gate, "TRAINING_STARTED": True,
        "TRAINING_COMPLETED": status == 0, "UPDATES": base_result.get("LEARNER_UPDATES"),
        **activity, "GPU_BACKEND": base_result.get("GPU_BACKEND"), "DEMO_BUFFER": 1921,
        "ONLINE_REPLAY_FINAL": base_result.get("ONLINE_REPLAY", {}).get("final"),
        "DEMO_ONLINE_RATIO": "32:32",
        "LOSSES_FINITE": "PASS" if base_result.get("LOSSES_FINITE") else "FAIL",
        "ACTOR_LEARNER_COMM": "PASS" if checks.get("communication") else "FAIL",
        "CKPT_500": "PASS" if checks.get("checkpoint_500") else "FAIL",
        "CKPT_1000": "PASS" if checks.get("checkpoint_1000") else "FAIL",
        "Q_POLICY_MINUS_DEMO_BASELINE": BASE_Q_GAP,
        "Q_POLICY_MINUS_DEMO_COMBINED": qgap, "DELTA_Q_BIAS": qgap - BASE_Q_GAP,
        "DEMO_ACTION_MAE_BASELINE": BASE_MAE, "DEMO_ACTION_MAE_COMBINED": overall,
        "DELTA_DEMO_ACTION_MAE": overall - BASE_MAE,
        "LATERAL_MAE_BASELINE": BASE_LAT_MAE, "LATERAL_MAE_COMBINED": audit["ABLATION_LATERAL_MAE"],
        "DELTA_LATERAL_MAE": audit["ABLATION_LATERAL_MAE"] - BASE_LAT_MAE,
        "ACTION1_MAE_BASELINE": BASE_A1_MAE, "ACTION1_MAE_COMBINED": action1,
        "DELTA_ACTION1_MAE": action1 - BASE_A1_MAE,
        "Q_DEMO_GT_POLICY_RATE": audit["Q_DEMO_GT_POLICY_RATE"],
        "Q_POLICY_GT_DEMO_RATE": audit["Q_POLICY_GT_DEMO_RATE"], "Q_SCALE": audit["Q_SCALE"],
        "NORMAL_SHUTDOWN": "PASS" if normal else "FAIL",
        "RESIDUAL_PROCESSES": base_result.get("RESIDUAL_PROCESSES"), "ABLATION_RESULT": result,
        "COMBINED_RESULT": result,
        "EXTENDED_VALIDATION": "READY_FOR_STAGE_E" if gate == "PASS" else "BLOCKED",
        "FRESH_5K_R2": "NOT_YET",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "NEXT": "STAGE_E_BOUNDED_EXTENDED_VALIDATION" if gate == "PASS" else "STOP_COMBINED_FAIL",
        "CONFIG_SHA256": sha(CONFIG_PATH), "DATASET_SHA256": cfg["demo"]["sha256"],
    }
    (out / "final_result.json").write_text(json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "REPORT.md").write_text("# Peg F3 V2 Combined Q-Anchor Ablation 1K\n\n```json\n" +
                                        json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(final, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if gate == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base, cfg, frozen_production_defaults = install_bindings(args.role)
    require(args.output is None or args.output.resolve() == OUTPUT, "canonical output binding")
    if args.role == "preflight":
        result = base.preflight()
        result.update(runtime_binding_gate(base))
        result.update(config_semantics_gate(cfg, frozen_production_defaults))
        result.update(demo_tail_identity_gate(base))
        result["CONFIG_SHA256"] = sha(CONFIG_PATH)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.role == "supervise":
        require(not OUTPUT.exists(), "fresh isolated output required")
        binding = runtime_binding_gate(base)
        semantics = config_semantics_gate(cfg, frozen_production_defaults)
        tail = demo_tail_identity_gate(base)
        before = protected_snapshot()
        status = base.supervise()
        after = protected_snapshot()
        if OUTPUT.exists():
            for name, value in (("runtime_binding_gate.json", binding), ("config_semantics_gate.json", semantics),
                                ("demo_tail_identity_gate.json", tail), ("protected_before.json", before),
                                ("protected_after.json", after)):
                (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        require(before == after, "protected artifact changed")
        if status != 0:
            return status
        activity = conservative_activity(OUTPUT)
        audit = matched_audit(OUTPUT, cfg)
        return finalize(OUTPUT, cfg, status, audit, activity)
    require(args.output is not None and args.output.resolve() == OUTPUT, "child output binding")
    return {"learner": base.learner, "actor": base.actor, "verify": base.verify}[args.role](OUTPUT) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
