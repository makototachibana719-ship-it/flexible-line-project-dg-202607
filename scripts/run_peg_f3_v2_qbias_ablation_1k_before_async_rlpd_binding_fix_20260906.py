#!/usr/bin/env python3
"""Authorized fresh 1K Q-bias ablation using the proven F3 learner/actor harness."""
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
CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_qbias_ablation_1k_seed0.yaml"
EXPECTED_CONFIG = "290e10e1a18ba48ae8302a8eda63d994df26eb11fddcb446a77dff0b8368c9b9"
OUTPUT = ROOT / "results/ablation/peg_f3_v2_qbias_ablation_1k_seed0_retry_001"
BASE_Q_GAP = 1.578540
BASE_MAE = 0.713376
BASE_LAT_MAE = 0.734863
BASE_A1_MAE = 1.015701
ABLATION_RUN_ID = "wandb-disabled-peg-f3-v2-qbias-ablation-1k"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def python_binding_identity(invoked_path, expected_path, sys_prefix, expected_env):
    """Validate interpreter identity by real binary and Conda prefix, not spelling."""
    invoked = Path(invoked_path)
    expected = Path(expected_path)
    prefix = Path(sys_prefix)
    env = Path(expected_env)
    expected_exists = expected.is_file()
    invoked_resolved = invoked.resolve(strict=False)
    expected_resolved = expected.resolve(strict=False)
    env_resolved = env.resolve(strict=False)
    realpath_equivalent = expected_exists and invoked_resolved == expected_resolved
    env_identity = prefix.resolve(strict=False) == env_resolved
    binary_in_env = invoked_resolved == env_resolved or env_resolved in invoked_resolved.parents
    accepted = expected_exists and realpath_equivalent and env_identity and binary_in_env
    return {
        "PYTHON_INVOKED_PATH": str(invoked),
        "PYTHON_RESOLVED_PATH": str(invoked_resolved),
        "EXPECTED_PATH": str(expected),
        "EXPECTED_RESOLVED_PATH": str(expected_resolved),
        "SYS_PREFIX": str(prefix),
        "EXPECTED_INTERPRETER_EXISTS": "PASS" if expected_exists else "FAIL",
        "REALPATH_EQUIVALENCE": "PASS" if realpath_equivalent else "FAIL",
        "CONDA_ENV_IDENTITY": "PASS" if env_identity and binary_in_env else "FAIL",
        "accepted": accepted,
    }


def protected_snapshot():
    paths = [ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml",
             ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl",
             ROOT / "results/datasets/peg_v2_accepted_11_20260906/manifest.json",
             ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json"]
    formal = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run/checkpoints"
    for step in (1000, 2000, 3000, 4000, 5000):
        cp = formal / f"checkpoint_{step}"
        paths.extend((cp, cp.with_name(cp.name + "_online_replay.pkl"),
                      cp.with_name(cp.name + "_resume_metadata.json")))
    require(all(path.is_file() for path in paths), "protected artifact missing")
    return {str(path): sha(path) for path in paths}


def runtime_binding_gate(base):
    expected_python = Path("/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python")
    expected_env = Path("/home/makoto/miniconda3/envs/serl_peg_gpu")
    expected_dataset = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
    env = base.runtime_env()
    pythonpath = env.get("PYTHONPATH", "").split(os.pathsep)
    required = [ROOT, ROOT / "external/serl/serl_launcher", ROOT / "external/serl/franka_sim"]
    require(ROOT == Path("/home/makoto/flexible_line_project"), "project root binding")
    python_identity = python_binding_identity(sys.executable, expected_python, sys.prefix, expected_env)
    require(python_identity["accepted"], "python interpreter identity binding")
    require(all(str(path) in pythonpath for path in required), "formal-equivalent PYTHONPATH binding")
    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "ablation config SHA256")
    require(sha(expected_dataset) == "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878",
            "dataset SHA256")
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT / 'examples/async_rlpd')!r}); "
        "import franka_sim, async_sac_state_sim, training; "
        "print('FRANKA_SIM_IMPORT=PASS'); print('ASYNC_RLPD_IMPORT=PASS'); "
        "print('TRAINING_PACKAGE_IMPORT=PASS')"
    )
    check = subprocess.run([str(expected_python), "-c", code], cwd=ROOT, env=env,
                           text=True, capture_output=True, timeout=120)
    require(check.returncode == 0, "runtime import gate: " + check.stderr.strip())
    import jax
    require(jax.default_backend() == "gpu", "JAX backend")
    result = {
        "RUNTIME_BINDING": "PASS", "PROJECT_ROOT": str(ROOT), "PYTHON": str(expected_python),
        "PYTHONPATH_BINDING": [str(path) for path in required],
        "FRANKA_SIM_IMPORT": "PASS", "ASYNC_RLPD_IMPORT": "PASS",
        "TRAINING_PACKAGE_IMPORT": "PASS", "JAX_BACKEND": "gpu",
        "CONFIG_SHA256": sha(CONFIG_PATH), "DATASET_SHA256": sha(expected_dataset),
    }
    result.update({key: value for key, value in python_identity.items() if key != "accepted"})
    return result


class VirtualConfig(dict):
    """Expose harness-only limits without changing serialized config identity."""
    _smoke = {"learner_updates": 1000, "checkpoint_period": 500,
              "actor_max_steps": 1000000, "process_timeout_seconds": 7200,
              "update_timeout_seconds": 120}

    def __getitem__(self, key):
        if key == "smoke" and key not in self:
            return self._smoke
        return super().__getitem__(key)


def install_bindings(role):
    import yaml
    import scripts.run_peg_f3_v2_rlpd_smoke as base
    from training.peg_insertion.qbias_ablation import make_qbias_ablation_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "ablation config SHA256")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    require(raw["optimizer"]["critic_learning_rate"] == .0001 and
            raw["optimizer"]["critic_warmup_steps"] == 3000, "critic overrides")
    require(raw["optimizer"]["actor_learning_rate"] == .0003 and
            raw["optimizer"]["temperature_learning_rate"] == .0003 and
            raw["optimizer"]["actor_warmup_steps"] == 2000, "protected optimizer fields")

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_qbias_ablation_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save = resume.save_bundle
    original_load = resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {"identifier": "peg-f3-v2-qbias-ablation-1k", "profile": "config_only_qbias_ablation",
                  "fresh_initialization": True, "resume": False, "demo_online_ratio": "32:32",
                  "utd_ratio": 1, "critic_learning_rate": .0001, "critic_warmup_steps": 3000}
        return original_save(directory, state, replay, step, actor_steps, cfg, manifest,
                             ABLATION_RUN_ID, source, **kwargs)

    def load_bundle(cp, cfg, manifest, run_id):
        return original_load(cp, cfg, manifest, ABLATION_RUN_ID)

    resume.save_bundle = save_bundle
    resume.load_bundle = load_bundle
    base.CONFIG = CONFIG_PATH
    base.OUTPUT = OUTPUT
    base.ACTIVE_OUTPUT = OUTPUT
    base.RUN_UPDATES = 1000
    base.CHECKPOINT_TARGETS = (500, 1000)
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

    original_scalars = base.loss_scalars
    def loss_and_lr_scalars(jax, info):
        values = original_scalars(jax, info)
        for path, leaf in jax.tree_util.tree_flatten_with_path(info)[0]:
            key = str(path)
            array = __import__("numpy").asarray(jax.device_get(leaf))
            if "_lr" in key.lower() and array.size == 1:
                values[key] = float(array.reshape(()))
        return values
    base.loss_scalars = loss_and_lr_scalars
    return base, raw


def q_stats(values):
    import numpy as np
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def matched_audit(out, cfg):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.qbias_ablation import make_qbias_ablation_agent

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

    fresh = make_qbias_ablation_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    initial_critic = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(fresh.state.params["modules_critic"])]
    state = restore_fresh_state(out / "checkpoints", fresh.state, 1000)
    agent = fresh.replace(state=state)
    trained_critic = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(state.params["modules_critic"])]
    critic_delta = float(np.sqrt(sum(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)
                                     for a, b in zip(initial_critic, trained_critic))))
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
    updates = [json.loads(line) for line in (out / "updates.jsonl").open()]
    critic_lr = []
    critic_lr_steps = []
    for row in updates:
        values = [v for k, v in row["losses"].items() if "critic_lr" in k.lower()]
        if values:
            critic_lr.append(values[0])
            critic_lr_steps.append(int(row["update"]))
    require(len(updates) == 1000 and critic_lr, "critic LR telemetry")
    active_steps = [step for step, lr in zip(critic_lr_steps, critic_lr) if lr > 0]
    critic_update_count = len(active_steps)
    active = critic_delta > 0 and critic_update_count > 0
    active_class = "NO" if not active else ("YES" if critic_update_count == 1000 else "PARTIAL")
    interpretation = ("CRITIC_FROZEN_CONTROL" if not active else
                      "LOWER_LR_ACTIVE_CRITIC" if critic_update_count == 1000 else
                      "LOWER_LR_PARTIALLY_ACTIVE_CRITIC")
    audit = {
        "ABLATION_Q_POLICY_MINUS_DEMO": float(gap.mean()),
        "BASELINE_Q_POLICY_MINUS_DEMO": BASE_Q_GAP,
        "DELTA_Q_BIAS": float(gap.mean() - BASE_Q_GAP),
        "ABLATION_DEMO_ACTION_MAE": overall_mae,
        "ABLATION_LATERAL_MAE": lateral_mae,
        "ABLATION_ACTION1_MAE": action1_mae,
        "Q_DEMO_GT_POLICY_RATE": float(np.mean(q_demo > q_policy)),
        "Q_POLICY_GT_DEMO_RATE": float(np.mean(q_policy > q_demo)),
        "Q_SCALE": {"demo": q_stats(q_demo), "policy": q_stats(q_policy),
                    "ensemble_policy_mean_std": float(q_policy_members.std(axis=0).mean())},
        "Q_PERTURBATION_RANKING": {"policy_above_all_demo_perturbations_rate": float(np.mean(q_policy > q_perturb.max(axis=1))),
                                   "perturbed_minus_demo": q_stats((q_perturb - q_demo[:, None]).reshape(-1))},
        "CRITIC_PARAMETER_L2_CHANGE": critic_delta,
        "CRITIC_LR_FIRST": float(critic_lr[0]), "CRITIC_LR_LAST": float(critic_lr[-1]),
        "CRITIC_LR_MAX": float(max(critic_lr)), "CRITIC_UPDATE_ACTIVE_DURING_1K": active_class,
        "FIRST_CRITIC_UPDATE_STEP": active_steps[0] if active_steps else None,
        "CRITIC_UPDATE_COUNT": critic_update_count,
        "CRITIC_UPDATE_TRIGGER": "nonzero critic optimizer learning-rate schedule during learner update_high_utd",
        "ABLATION_INTERPRETATION": interpretation,
    }
    write = lambda path, value: path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write(out / "matched_step_audit.json", audit)
    return audit


def finalize(out, cfg, base_status, audit):
    base_result = json.loads((out / "result.json").read_text())
    cleanup = json.loads((out / "actor_cleanup_complete.json").read_text())
    checks = base_result.get("checks", {})
    normal_shutdown = all(checks.get(k, False) for k in ("learner_stop", "actor_stop_observed", "trainerclient_stop",
                                                           "broadcast_join", "actor_natural_exit", "children_exited"))
    qgap = audit["ABLATION_Q_POLICY_MINUS_DEMO"]
    alignment_ok = audit["ABLATION_DEMO_ACTION_MAE"] <= BASE_MAE * 1.05
    lateral_not_worse = audit["ABLATION_ACTION1_MAE"] <= BASE_A1_MAE
    if audit["CRITIC_UPDATE_ACTIVE_DURING_1K"] == "NO":
        scientific = "FROZEN_CRITIC_CONTROL"
    elif qgap <= .5 and alignment_ok and lateral_not_worse:
        scientific = "STRONG_PASS"
    elif qgap < BASE_Q_GAP and alignment_ok:
        scientific = "PASS"
    else:
        scientific = "FAIL"
    gate = "PASS" if base_status == 0 and scientific in ("STRONG_PASS", "PASS") and normal_shutdown else "FAIL"
    final = {
        "PEG_F3_V2_QBIAS_ABLATION_1K": gate, "TRAINING_COMPLETED": base_status == 0,
        "UPDATES": base_result.get("LEARNER_UPDATES"), "GPU_BACKEND": base_result.get("GPU_BACKEND"),
        "CRITIC_UPDATE_ACTIVE_DURING_1K": audit["CRITIC_UPDATE_ACTIVE_DURING_1K"],
        "FIRST_CRITIC_UPDATE_STEP": audit["FIRST_CRITIC_UPDATE_STEP"],
        "CRITIC_UPDATE_COUNT": audit["CRITIC_UPDATE_COUNT"],
        "CRITIC_UPDATE_TRIGGER": audit["CRITIC_UPDATE_TRIGGER"],
        "ABLATION_INTERPRETATION": audit["ABLATION_INTERPRETATION"],
        "CRITIC_LR_FIRST": audit["CRITIC_LR_FIRST"], "CRITIC_LR_LAST": audit["CRITIC_LR_LAST"],
        "CRITIC_LR_MAX": audit["CRITIC_LR_MAX"], "CKPT_500": checks.get("checkpoint_500"),
        "CKPT_1000": checks.get("checkpoint_1000"),
        "Q_POLICY_MINUS_DEMO_BASELINE": BASE_Q_GAP, "Q_POLICY_MINUS_DEMO_ABLATION": audit["ABLATION_Q_POLICY_MINUS_DEMO"],
        "DELTA_Q_BIAS": audit["DELTA_Q_BIAS"], "DEMO_ACTION_MAE_BASELINE": BASE_MAE,
        "DEMO_ACTION_MAE_ABLATION": audit["ABLATION_DEMO_ACTION_MAE"], "LATERAL_MAE_BASELINE": BASE_LAT_MAE,
        "LATERAL_MAE_ABLATION": audit["ABLATION_LATERAL_MAE"], "ACTION1_MAE_BASELINE": BASE_A1_MAE,
        "ACTION1_MAE_ABLATION": audit["ABLATION_ACTION1_MAE"], "Q_DEMO_GT_POLICY_RATE": audit["Q_DEMO_GT_POLICY_RATE"],
        "Q_POLICY_GT_DEMO_RATE": audit["Q_POLICY_GT_DEMO_RATE"], "Q_SCALE": audit["Q_SCALE"],
        "Q_PERTURBATION_RANKING": audit["Q_PERTURBATION_RANKING"], "NORMAL_SHUTDOWN": "PASS" if normal_shutdown else "FAIL",
        "RESIDUAL_PROCESSES": base_result.get("RESIDUAL_PROCESSES"), "ABLATION_RESULT": scientific,
        "FRESH_5K_R2": "READY" if scientific in ("STRONG_PASS", "PASS") and gate == "PASS" else ("NOT_YET" if scientific == "FROZEN_CRITIC_CONTROL" else "BLOCKED"),
        "NEXT": ("USER_AUTHORIZATION_REQUIRED_FOR_FRESH_5K_R2" if gate == "PASS" else
                 "EXTENDED_ABLATION_REQUIRED" if scientific == "FROZEN_CRITIC_CONTROL" else "MINIMAL_FIX_REDESIGN_REQUIRED"),
        "CONFIG_SHA256": sha(CONFIG_PATH), "DATASET_SHA256": cfg["demo"]["sha256"],
        "TRAINING_STARTED": True, "RESUME_STARTED": False,
        "actor_shutdown": cleanup.get("shutdown"),
    }
    (out / "final_result.json").write_text(json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "REPORT.md").write_text("# Peg F3 V2 Q-Bias Config-Only Ablation 1K\n\n```json\n" +
                                    json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(final, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if gate == "PASS" else 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    base, cfg = install_bindings(args.role)
    require(args.output is None or args.output.resolve() == OUTPUT, "canonical ablation output binding")
    if args.role == "preflight":
        binding = runtime_binding_gate(base)
        result = base.preflight(); result.update(binding); result.update(config_sha256=sha(CONFIG_PATH), critic_learning_rate=.0001,
                                                  critic_warmup_steps=3000, actor_learning_rate=.0003,
                                                  temperature_learning_rate=.0003, actor_warmup_steps=2000)
        print(json.dumps(result, indent=2)); return 0
    if args.role == "supervise":
        require(not OUTPUT.exists(), "fresh ablation output required")
        binding = runtime_binding_gate(base)
        protected_before = protected_snapshot()
        status = base.supervise()
        protected_after = protected_snapshot()
        if OUTPUT.exists():
            (OUTPUT / "runtime_binding_gate.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
            (OUTPUT / "protected_before.json").write_text(json.dumps(protected_before, indent=2, sort_keys=True) + "\n")
            (OUTPUT / "protected_after.json").write_text(json.dumps(protected_after, indent=2, sort_keys=True) + "\n")
        require(protected_before == protected_after, "protected artifact changed")
        if status != 0:
            return status
        audit = matched_audit(OUTPUT, cfg)
        return finalize(OUTPUT, cfg, status, audit)
    require(args.output is not None and args.output.resolve() == OUTPUT, "child output binding")
    return {"learner": base.learner, "actor": base.actor, "verify": base.verify}[args.role](OUTPUT) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
