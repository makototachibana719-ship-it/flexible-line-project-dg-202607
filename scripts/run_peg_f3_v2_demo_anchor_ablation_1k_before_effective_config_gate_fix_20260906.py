#!/usr/bin/env python3
"""Authorized fresh 1K Demo-anchor ablation on the proven F3 harness."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_peg_f3_v2_qbias_ablation_1k as shared

CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_demo_anchor_ablation_1k_seed0.yaml"
EXPECTED_CONFIG = "d93f9de0af827f1ac45f642da4ccee9bf41e90537f596c00701f66cdc3091d00"
OUTPUT = ROOT / "results/ablation/peg_f3_v2_demo_anchor_ablation_1k_seed0_run_001"
RUN_ID = "wandb-disabled-peg-f3-v2-demo-anchor-ablation-1k"
BASE_Q_GAP = 1.578540
BASE_MAE = 0.713376
BASE_LAT_MAE = 0.734863
BASE_A1_MAE = 1.015701
DESIGN_GRAD_RATIO = 0.0982066153


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
        ROOT / "training/peg_insertion/actor_demo_anchor_ablation.py",
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
        "from training.peg_insertion.actor_demo_anchor_ablation import make_demo_anchor_agent; "
        "print('PASS')"
    )
    check = subprocess.run([str(Path('/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python')), "-c", code],
                           cwd=ROOT, env=base.runtime_env(), text=True, capture_output=True, timeout=120)
    require(check.returncode == 0 and check.stdout.strip().splitlines()[-1] == "PASS",
            "Demo-anchor factory runtime import")
    result.pop("QBIAS_FACTORY_IMPORT", None)
    result["DEMO_ANCHOR_FACTORY_IMPORT"] = "PASS"
    return result


def config_semantics_gate(raw):
    baseline = __import__("yaml").safe_load(
        (ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml").read_text())
    for section in ("environment", "algorithm", "seed", "fresh_initialization",
                    "load_legacy_checkpoint", "demo", "observation", "action",
                    "optimizer", "training", "algorithm_defaults", "contracts"):
        require(raw.get(section) == baseline.get(section), f"baseline semantic drift: {section}")
    anchor = raw.get("demo_actor_anchor", {})
    require(anchor.get("enabled") is True and anchor.get("scope") == "all_demo", "anchor scope")
    require(anchor.get("demo_batch_size") == 32 and anchor.get("selected_lambda") == .07,
            "anchor size/lambda")
    require(anchor.get("loss") == "mse_policy_mode_to_demo_action", "anchor loss")
    require(raw["ablation"]["semantic_diff_fields"] == ["demo_actor_anchor"], "semantic diff fields")
    return {
        "CONFIG_SEMANTICS": "PASS", "CONFIG_DIFF_FIELDS": ["demo_actor_anchor"],
        "ACTOR_LR": raw["optimizer"]["actor_learning_rate"],
        "CRITIC_LR": raw["optimizer"]["critic_learning_rate"],
        "TEMPERATURE_LR": raw["optimizer"]["temperature_learning_rate"],
        "ACTOR_WARMUP": raw["optimizer"]["actor_warmup_steps"],
        "CRITIC_WARMUP": raw["optimizer"]["critic_warmup_steps"],
        "DEMO_ONLINE_RATIO": "32:32", "BATCH": 64, "UTD": 1,
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
    from training.peg_insertion.actor_demo_anchor_ablation import make_demo_anchor_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "ablation config SHA256")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    config_semantics_gate(raw)

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_demo_anchor_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save, original_load = resume.save_bundle, resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {"identifier": "peg-f3-v2-demo-anchor-ablation-1k",
                  "profile": "isolated_demo_anchor_ablation", "fresh_initialization": True,
                  "resume": False, "demo_online_ratio": "32:32", "utd_ratio": 1,
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
    return base, raw


def matched_audit(out, cfg):
    """Reuse the exact previously accepted 405-state/24-lateral metric implementation."""
    import training.peg_insertion.qbias_ablation as qmodule
    from training.peg_insertion.actor_demo_anchor_ablation import make_demo_anchor_agent
    old = qmodule.make_qbias_ablation_agent
    qmodule.make_qbias_ablation_agent = make_demo_anchor_agent
    try:
        return shared.matched_audit(out, cfg)
    finally:
        qmodule.make_qbias_ablation_agent = old


def anchor_activity(out):
    import numpy as np
    rows = [json.loads(line) for line in (out / "updates.jsonl").open()]
    values = []
    for row in rows:
        losses = row["losses"]
        def one(fragment):
            hits = [float(v) for key, v in losses.items() if fragment in key]
            require(len(hits) == 1, f"anchor telemetry key {fragment}")
            return hits[0]
        values.append((int(row["update"]), one("sac_actor_loss"),
                       one("demo_anchor_loss"), one("weighted_demo_anchor_loss"),
                       one("actor_loss")))
    a = np.asarray([item[1:] for item in values], np.float64)
    require(len(values) == 1000 and np.isfinite(a).all(), "anchor activity trace")
    require(np.allclose(a[:, 2], .07 * a[:, 1], rtol=2e-5, atol=2e-7), "anchor lambda trace")
    require(np.allclose(a[:, 3], a[:, 0] + a[:, 2], rtol=2e-5, atol=2e-6), "total actor loss trace")
    return {"ANCHOR_ACTIVE": "YES", "ANCHOR_UPDATE_COUNT": len(values),
            "FIRST_ANCHOR_UPDATE_STEP": values[0][0], "MEAN_SAC_ACTOR_LOSS": float(a[:, 0].mean()),
            "MEAN_ANCHOR_RAW_LOSS": float(a[:, 1].mean()),
            "MEAN_ANCHOR_WEIGHTED_LOSS": float(a[:, 2].mean()),
            "MEAN_TOTAL_ACTOR_LOSS": float(a[:, 3].mean()),
            "ANCHOR_GRAD_RATIO": DESIGN_GRAD_RATIO,
            "ANCHOR_GRAD_RATIO_SOURCE": "pre-registered fixed-batch design measurement",
            "ANCHOR_DOMINATES": "NO" if DESIGN_GRAD_RATIO < 1.0 else "YES"}


def finalize(out, cfg, status, audit, activity):
    base_result = json.loads((out / "result.json").read_text())
    checks = base_result.get("checks", {})
    normal = all(checks.get(key, False) for key in
                 ("learner_stop", "actor_stop_observed", "trainerclient_stop", "broadcast_join",
                  "actor_natural_exit", "children_exited"))
    health = (status == 0 and base_result.get("LEARNER_UPDATES") == 1000 and
              base_result.get("GPU_BACKEND") == "gpu" and normal and
              checks.get("checkpoint_500") and checks.get("checkpoint_1000") and
              activity["ANCHOR_ACTIVE"] == "YES" and activity["ANCHOR_DOMINATES"] == "NO")
    qgap = audit["ABLATION_Q_POLICY_MINUS_DEMO"]
    overall = audit["ABLATION_DEMO_ACTION_MAE"]
    action1 = audit["ABLATION_ACTION1_MAE"]
    if health and action1 < BASE_A1_MAE and overall < BASE_MAE and qgap <= 1.0:
        result = "STRONG_PASS"
    elif health and action1 < BASE_A1_MAE and overall < BASE_MAE and qgap <= BASE_Q_GAP * 1.05:
        result = "PASS"
    else:
        result = "FAIL"
    gate = "PASS" if result in ("STRONG_PASS", "PASS") else "FAIL"
    final = {
        "PEG_F3_V2_DEMO_ANCHOR_ABLATION_1K": gate, "TRAINING_STARTED": True,
        "TRAINING_COMPLETED": status == 0, "UPDATES": base_result.get("LEARNER_UPDATES"),
        **activity, "GPU_BACKEND": base_result.get("GPU_BACKEND"), "DEMO_BUFFER": 1921,
        "ONLINE_REPLAY_FINAL": base_result.get("ONLINE_REPLAY", {}).get("final"),
        "DEMO_ONLINE_RATIO": "32:32",
        "LOSSES_FINITE": "PASS" if base_result.get("LOSSES_FINITE") else "FAIL",
        "ACTOR_LEARNER_COMM": "PASS" if checks.get("communication") else "FAIL",
        "CKPT_500": "PASS" if checks.get("checkpoint_500") else "FAIL",
        "CKPT_1000": "PASS" if checks.get("checkpoint_1000") else "FAIL",
        "Q_POLICY_MINUS_DEMO_BASELINE": BASE_Q_GAP,
        "Q_POLICY_MINUS_DEMO_ANCHOR": qgap, "DELTA_Q_BIAS": qgap - BASE_Q_GAP,
        "DEMO_ACTION_MAE_BASELINE": BASE_MAE, "DEMO_ACTION_MAE_ANCHOR": overall,
        "DELTA_DEMO_ACTION_MAE": overall - BASE_MAE,
        "LATERAL_MAE_BASELINE": BASE_LAT_MAE, "LATERAL_MAE_ANCHOR": audit["ABLATION_LATERAL_MAE"],
        "DELTA_LATERAL_MAE": audit["ABLATION_LATERAL_MAE"] - BASE_LAT_MAE,
        "ACTION1_MAE_BASELINE": BASE_A1_MAE, "ACTION1_MAE_ANCHOR": action1,
        "DELTA_ACTION1_MAE": action1 - BASE_A1_MAE,
        "Q_DEMO_GT_POLICY_RATE": audit["Q_DEMO_GT_POLICY_RATE"],
        "Q_POLICY_GT_DEMO_RATE": audit["Q_POLICY_GT_DEMO_RATE"], "Q_SCALE": audit["Q_SCALE"],
        "NORMAL_SHUTDOWN": "PASS" if normal else "FAIL",
        "RESIDUAL_PROCESSES": base_result.get("RESIDUAL_PROCESSES"), "ABLATION_RESULT": result,
        "DEMO_ANCHOR_EXTENDED_ABLATION": "READY" if gate == "PASS" else "BLOCKED",
        "FRESH_5K_R2": "NOT_YET" if gate == "PASS" else "BLOCKED",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "NEXT": "USER_AUTHORIZATION_REQUIRED_FOR_EXTENDED_DEMO_ANCHOR_ABLATION" if gate == "PASS" else
                "ANCHOR_REDESIGN_OR_CONSERVATIVE_Q_DIAGNOSTIC",
        "CONFIG_SHA256": sha(CONFIG_PATH), "DATASET_SHA256": cfg["demo"]["sha256"],
    }
    (out / "final_result.json").write_text(json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "REPORT.md").write_text("# Peg F3 V2 Demo-Anchor Ablation 1K\n\n```json\n" +
                                        json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(final, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if gate == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base, cfg = install_bindings(args.role)
    require(args.output is None or args.output.resolve() == OUTPUT, "canonical output binding")
    if args.role == "preflight":
        result = base.preflight()
        result.update(runtime_binding_gate(base))
        result.update(config_semantics_gate(cfg))
        result.update(demo_tail_identity_gate(base))
        result["CONFIG_SHA256"] = sha(CONFIG_PATH)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.role == "supervise":
        require(not OUTPUT.exists(), "fresh isolated output required")
        binding = runtime_binding_gate(base)
        semantics = config_semantics_gate(cfg)
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
        activity = anchor_activity(OUTPUT)
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
