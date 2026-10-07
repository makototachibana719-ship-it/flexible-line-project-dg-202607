#!/usr/bin/env python3
"""CPU/static regression for the F3 runtime Python identity Gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_peg_f3_v2_qbias_ablation_1k import python_binding_identity

EXPECTED = Path("/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python")
EXPECTED_ENV = Path("/home/makoto/miniconda3/envs/serl_peg_gpu")


def check(invoked, prefix):
    return python_binding_identity(invoked, EXPECTED, prefix, EXPECTED_ENV)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("fresh validation output required")

    actual = check(sys.executable, sys.prefix)
    resolved = check(EXPECTED.resolve(), EXPECTED_ENV)
    with tempfile.TemporaryDirectory(prefix="peg-python-binding-") as tmp:
        alternate = Path(tmp) / "python-f3-equivalent"
        os.symlink(EXPECTED, alternate)
        alternate_result = check(alternate, EXPECTED_ENV)

    negative = {
        "system": check("/usr/bin/python3", "/usr"),
        "base_conda": check("/home/makoto/miniconda3/bin/python", "/home/makoto/miniconda3"),
        "other_conda": check("/home/makoto/miniconda3/envs/other/bin/python",
                             "/home/makoto/miniconda3/envs/other"),
        "windows": check("C:/Python/python.exe", "C:/Python"),
    }
    gates = {
        "EXPECTED_INTERPRETER_EXISTS": actual["EXPECTED_INTERPRETER_EXISTS"],
        "REALPATH_EQUIVALENCE": actual["REALPATH_EQUIVALENCE"],
        "CONDA_ENV_IDENTITY": actual["CONDA_ENV_IDENTITY"],
        "EXPECTED_SYMLINK_PATH_ACCEPTANCE": actual["accepted"],
        "RESOLVED_PATH_ACCEPTANCE": resolved["accepted"],
        "SAME_ENV_SYMLINK_ACCEPTANCE": alternate_result["accepted"],
        "WRONG_SYSTEM_PYTHON_REJECTION": not negative["system"]["accepted"],
        "WRONG_BASE_CONDA_REJECTION": not negative["base_conda"]["accepted"],
        "WRONG_CONDA_ENV_REJECTION": not negative["other_conda"]["accepted"],
        "WRONG_WINDOWS_PYTHON_REJECTION": not negative["windows"]["accepted"],
    }
    passed = all(value == "PASS" if isinstance(value, str) else value for value in gates.values())
    result = {
        "PEG_F3_RUNTIME_PYTHON_BINDING_GATE_FIX": "PASS" if passed else "FAIL",
        **{key: value for key, value in actual.items() if key != "accepted"},
        **{key: "PASS" if value else "FAIL" for key, value in gates.items() if isinstance(value, bool)},
        "WORKLOAD_STARTED": False,
        "ABLATION_RETRY": "READY" if passed else "BLOCKED",
        "matrix": {"actual": actual, "resolved": resolved, "alternate_symlink": alternate_result,
                   "negative": negative},
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "REPORT.md").write_text(
        "# Peg F3 Runtime Python Binding Gate Equivalence Fix\n\n```json\n"
        + json.dumps(result, indent=2, sort_keys=True) + "\n```\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
