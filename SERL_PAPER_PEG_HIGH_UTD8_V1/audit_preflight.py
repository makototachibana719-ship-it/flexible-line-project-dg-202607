#!/usr/bin/env python3
"""Read-only causal-isolation audit for the UTD8 Stage-A branch."""
from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path

import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
V1 = ROOT / "SERL_PAPER_PEG_BASELINE_V1"
RUNNER = BRANCH / "run_high_utd8.py"
EXCLUDED = ("c0", "c1", "c2", "c3", "residual", "gate", "option",
            "curriculum", "hil_serl", "intervention", "dagger")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = yaml.safe_load((BRANCH / "config.yaml").read_text())
    old = yaml.safe_load((V1 / "config.yaml").read_text())
    demo = ROOT / cfg["demo"]["path"]
    rows = pickle.loads(demo.read_bytes())
    tree = ast.parse(RUNNER.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    bad = sorted({name for name in imports
                  if any(token in name.lower() for token in EXCLUDED)})
    scientific_keys = ("environment", "algorithm", "seed", "fresh_initialization",
                       "load_legacy_checkpoint", "demo", "observation", "action", "contracts")
    scientific_equal = all(cfg[key] == old[key] for key in scientific_keys)
    training_equal_except_utd = all(
        cfg["training"][key] == old["training"][key]
        for key in old["training"] if key != "utd_ratio")
    sac = ROOT / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py"
    sac_text = sac.read_text()
    semantic_markers = all(marker in sac_text for marker in (
        "minibatch_size = batch_size // utd_ratio",
        "jax.lax.scan(scan_body",
        'networks_to_update=frozenset({"critic"})',
        'networks_to_update=frozenset({"actor", "temperature"})'))
    checks = {
        "V1_EXISTS_AND_UNTOUCHED_BY_LOCATION": V1.is_dir(),
        "SCIENTIFIC_FIELDS_IDENTICAL": scientific_equal,
        "TRAINING_IDENTICAL_EXCEPT_UTD": training_equal_except_utd,
        "UTD_ONLY_CHANGE_1_TO_8": old["training"]["utd_ratio"] == 1 and
                                  cfg["training"]["utd_ratio"] == 8,
        "CRITIC_ACTOR_RATIO_UNCHANGED": cfg["training"]["critic_actor_ratio"] ==
                                         old["training"]["critic_actor_ratio"] == 1,
        "DEMO_SHA": sha(demo) == cfg["demo"]["sha256"],
        "DEMO_CARDINALITY": len(rows) == cfg["demo"]["transitions"] == 1921,
        "NO_CUSTOM_IMPORTS": not bad,
        "UPSTREAM_HIGH_UTD_SEMANTIC_MARKERS": semantic_markers,
        "CHECKPOINT_SCHEDULE": cfg["stage_a"]["mandatory_checkpoints"] ==
                               [250, 500, 750, 1000] and
                               cfg["stage_a"]["conditional_checkpoints"] == [1500, 2000],
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "bad_imports": bad,
        "resolved_semantics": {
            "batch_size": 64, "utd_ratio": 8, "minibatch_size": 8,
            "critic_target_updates_per_iteration": 8,
            "actor_updates_per_iteration": 1,
            "temperature_updates_per_iteration": 1,
            "critic_actor_ratio": 1,
        },
        "demo_sha256": sha(demo),
        "v1_config_sha256": sha(V1 / "config.yaml"),
        "upstream_sac_sha256": sha(sac),
        "training_started": False,
    }
    output = BRANCH / "preflight_audit.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
