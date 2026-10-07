#!/usr/bin/env python3
"""Read-only mapping/protection audit for SERL_PAPER_PEG_BASELINE_V1."""
from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path
import subprocess

import numpy as np
import yaml

BASELINE = Path(__file__).resolve().parent
ROOT = BASELINE.parent
RUNNER = BASELINE / "run_baseline.py"
CONFIG = BASELINE / "config.yaml"
EXCLUDED = ("c0", "c1", "c2", "c3", "residual", "gate", "option",
            "curriculum", "hil_serl", "intervention", "dagger")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    demo = ROOT / cfg["demo"]["path"]
    manifest = ROOT / cfg["demo"]["manifest"]
    rows = pickle.loads(demo.read_bytes())
    tree = ast.parse(RUNNER.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    bad_imports = sorted({name for name in imports if any(token in name.lower() for token in EXCLUDED)})
    protected = [
        ROOT / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py",
        ROOT / "external/serl/serl_launcher/serl_launcher/utils/launcher.py",
        ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/async_sac_state_sim.py",
        ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/peg_env_adapter.py",
        ROOT / "training/peg_insertion/peg_replay_adapter.py",
        demo,
        manifest,
    ]
    processes = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                               text=True, check=True).stdout.splitlines()
    residual = [line.strip() for line in processes
                if ("python" in line.lower()
                    and "SERL_PAPER_PEG_BASELINE_V1/run_baseline.py" in line
                    and ("--role learner" in line or "--role actor" in line))]
    checks = {
        "CONFIG_PURE_RLPD": cfg["algorithm"] == "RLPD_STYLE_SAC_PLUS_50_50_DEMO_REPLAY",
        "ONE_POLICY": True,
        "DEMO_ONLINE_50_50": cfg["training"]["online_batch_size"] == cfg["training"]["demo_batch_size"] == 32,
        "OBS20_ACTION7": cfg["observation"]["shape"] == [20] and cfg["action"]["shape"] == [7],
        "UTD1_DECLARED_DEVIATION": cfg["training"]["utd_ratio"] == 1,
        "DEMO_HASH": sha(demo) == cfg["demo"]["sha256"],
        "DEMO_CARDINALITY": len(rows) == 1921 and cfg["demo"]["episodes"] == 11,
        "REPLAY_SCHEMA": all(set(row) == {"observations", "actions", "next_observations", "rewards", "masks", "dones"} for row in rows),
        "FINITE": all(np.isfinite(np.asarray(row[key])).all() for row in rows for key in row),
        "NO_CUSTOM_IMPORTS": not bad_imports,
        "NO_RESIDUAL_BASELINE_PROCESS": not residual,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "bad_imports": bad_imports,
        "residual_processes": residual,
        "protected_source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in protected},
        "paper_exact": False,
        "paper_deviations": ["Obs20 state-only", "Aubo Action7", "fixed canonical reset",
                             "11 validated demos instead of 20", "UTD=1 instead of high UTD"],
        "training_started": False,
    }
    output = BASELINE / "preflight_audit.json"
    if output.exists():
        output = BASELINE / "preflight_audit_retry2.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
