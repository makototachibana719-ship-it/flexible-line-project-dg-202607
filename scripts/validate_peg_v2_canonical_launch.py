#!/usr/bin/env python3
"""Static/pre-workload validation of the canonical WSL Peg V2 launcher."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_peg_v2_canonical_wsl.sh"
EXPECTED_PYTHON = "/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python"
EXPECTED_ROOT = "/home/makoto/flexible_line_project"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cwd: str, *args: str) -> dict:
    result = subprocess.run([str(LAUNCHER), *args], cwd=cwd, capture_output=True,
                            text=True, timeout=120, check=False)
    return {"cwd": cwd, "args": list(args), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_peg_v2_canonical_launch.py OUTPUT")
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = LAUNCHER.read_text()
    checks = {
        "project_root_binding": f"PROJECT_ROOT={EXPECTED_ROOT}" in source,
        "python_executable_binding": f"PYTHON={EXPECTED_PYTHON}" in source,
        "pythonpath_binding": 'export PYTHONPATH="$PROJECT_ROOT:' in source,
        "runtime_binding": all(item in source for item in (
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false", "MUJOCO_GL=egl")),
        "no_conda_activation": "conda activate" not in source,
    }
    cwd_results = [run(cwd, "imports") for cwd in
                   (EXPECTED_ROOT, "/home/makoto", "/tmp")]
    checks["imports_all_cwd"] = all(item["returncode"] == 0 and
                                     "PEG_F3_CANONICAL_IMPORTS=PASS" in item["stdout"]
                                     for item in cwd_results)
    help_result = run("/tmp", "help")
    checks["help"] = help_result["returncode"] == 0 and "--role" in help_result["stdout"]
    proposed_output = ROOT / "results/validation/peg/f3_reqrep_normal_path_revalidation_20260906_002"
    checks["proposed_output_fresh"] = not proposed_output.exists()
    preflight_result = run("/tmp", "short-preflight", str(proposed_output))
    checks["pre_workload_command_validation"] = (
        preflight_result["returncode"] == 0 and '"updates": 20' in preflight_result["stdout"]
        and not proposed_output.exists())
    result = {
        "PEG_F3_CANONICAL_LAUNCH_BINDING": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "cwd_imports": cwd_results, "help": help_result,
        "preflight": preflight_result, "launcher": str(LAUNCHER),
        "launcher_sha256": sha(LAUNCHER), "workload_started": False,
        "next": "NORMAL_PATH_REVALIDATION_READY" if all(checks.values()) else "BLOCKED",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
