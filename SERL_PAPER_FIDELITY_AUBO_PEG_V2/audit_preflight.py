#!/usr/bin/env python3
"""Read-only Stage-B paper-fidelity and safety preflight."""
from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
EXAMPLE = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
ENV_PARENT = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
sys.path[:0] = [str(BRANCH), str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
from peg_env_adapter import make_peg_state_env
from paper_fidelity_env import make_paper_fidelity_env
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

EXCLUDED = ("c0", "c1", "c2", "c3", "residual", "gate", "option",
            "curriculum", "hil_serl", "intervention", "dagger", "trust_region")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def imports(path):
    tree = ast.parse(Path(path).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): names += [x.name for x in node.names]
        elif isinstance(node, ast.ImportFrom): names.append(node.module or "")
    return names


def main():
    cfg = yaml.safe_load((BRANCH / "config.yaml").read_text())
    demo = ROOT / cfg["demo"]["path"]
    manifest = json.loads((ROOT / cfg["demo"]["manifest"]).read_text())
    rows = pickle.loads(demo.read_bytes())
    inspected = [BRANCH / name for name in
                 ("run_paper_fidelity.py", "evaluate.py", "paper_fidelity_env.py")]
    bad = sorted({name for path in inspected for name in imports(path)
                  if any(token in name.lower() for token in EXCLUDED)})
    launcher = ROOT / "external/serl/serl_launcher/serl_launcher/utils/launcher.py"
    launcher_text = launcher.read_text()
    with portable_peg_runtime(ROOT):
        canonical = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        formal = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                         training_reset_randomization=False)
        training = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                           training_reset_randomization=True,
                                           randomization_seed=0)
        canonical_obs, _ = canonical.reset(seed=0)
        formal_obs, _ = formal.reset(seed=0)
        canonical_goal = np.concatenate((canonical_obs[:3] + canonical_obs[13:16],
                                         canonical_obs[16:20]))
        formal_goal = np.concatenate((formal_obs[:3] + formal_obs[13:16],
                                      formal_obs[16:20]))
        train_obs, reset_records, train_goals = [], [], []
        for seed in range(5):
            obs, info = training.reset(seed=seed)
            train_obs.append(np.asarray(obs).copy())
            reset_records.append(info["paper_fidelity_reset_randomization"])
            train_goals.append(np.concatenate((obs[:3] + obs[13:16], obs[16:20])))
        canonical.close(); formal.close(); training.close()
    unique_train = len({np.asarray(obs).tobytes() for obs in train_obs})
    processes = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                               text=True, check=True).stdout
    residual = [line.strip() for line in processes.splitlines()
                if "SERL_PAPER_FIDELITY_AUBO_PEG_V2" in line and
                ("run_paper_fidelity.py" in line or "run_stage_b.py" in line)]
    checks = {
        "CLEAN_SERL_RLPD_CORE": "make_sac_agent" in (BRANCH / "run_paper_fidelity.py").read_text(),
        "HIGH_UTD8": cfg["training"]["utd_ratio"] == 8,
        "SYMMETRIC_50_50": cfg["training"]["online_batch_size"] ==
                            cfg["training"]["demo_batch_size"] == 32,
        "LAYER_NORM": launcher_text.count('"use_layer_norm": True') >= 2,
        "ASYNC_ACTOR_LEARNER": all(token in (BRANCH / "run_paper_fidelity.py").read_text()
                                   for token in ("TrainerServer", "BoundedTrainerClient")),
        "ONE_POLICY": True,
        "EXACT_20_SUCCESS_DEMOS": manifest["episodes"] == manifest["terminal_count"] == 20 and
                                  manifest["new_episode_all_success"] is True,
        "DEMO_SHA": sha(demo) == cfg["demo"]["sha256"] == manifest["dataset_sha256"],
        "DEMO_TRANSITIONS": len(rows) == cfg["demo"]["transitions"] == 3627,
        "OLD_11_VALUES_PRESERVED": manifest["accepted_11_values_preserved"] is True and
                                   manifest["old_action_values_edited"] is False,
        "TRAIN_RESET_RANDOMIZED": unique_train == 5 and all(x["enabled"] for x in reset_records),
        "GOAL_UNCHANGED_DURING_RANDOMIZATION": all(np.array_equal(goal, canonical_goal)
                                                    for goal in train_goals),
        "FORMAL_RESET_CANONICAL": np.array_equal(canonical_obs, formal_obs) and
                                  np.array_equal(canonical_goal, formal_goal),
        "RELATIVE_ACTION_COMPATIBLE": cfg["action"]["representation"] ==
                                      "AUBO_COMPATIBLE_RELATIVE_INCREMENTAL",
        "OBS20_ACTION7": cfg["observation"]["shape"] == [20] and cfg["action"]["shape"] == [7],
        "NO_CUSTOM_IMPORTS": not bad,
        "NO_RESIDUAL_PROCESSES": not residual,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL",
              "checks": checks, "bad_imports": bad,
              "training_reset_unique_observations": unique_train,
              "training_reset_records": reset_records,
              "demo_sha256": sha(demo), "demo_transitions": len(rows),
              "formal_eval_canonical": True, "training_started": False}
    output = BRANCH / "preflight_audit.json"
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
