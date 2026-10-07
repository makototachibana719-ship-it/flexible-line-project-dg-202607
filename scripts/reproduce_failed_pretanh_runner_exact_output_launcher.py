#!/usr/bin/env python3
"""Launch the exact failed pre-tanh runner with only OUTPUT rebound."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
for item in (
    ROOT,
    ROOT / "external/serl/franka_sim",
    ROOT / "external/packages/peg_insertion/peg_insertion/code/env",
):
    sys.path.insert(0, str(item))

import scripts.run_peg_f3_v2_early_pretanh_a0_anchor_ablation_1k as failed_runner

EXPECTED_RUNNER_SHA = "316759e2a2829183c7e944f60294fc80c46f566a2130b85625d80839b40698b7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_launcher_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repro-output", type=Path)
    known, rest = parser.parse_known_args(argv)
    output = known.repro_output
    if output is None:
        for index, item in enumerate(rest):
            if item == "--output" and index + 1 < len(rest):
                output = Path(rest[index + 1])
                break
    if output is None:
        raise RuntimeError("--repro-output required for supervise/preflight")
    output = output.resolve()
    if ROOT not in output.parents:
        raise RuntimeError("output outside project")
    return output, rest


def main():
    output, runner_argv = parse_launcher_args(sys.argv[1:])
    runner_path = ROOT / "scripts/run_peg_f3_v2_early_pretanh_a0_anchor_ablation_1k.py"
    if sha256(runner_path) != EXPECTED_RUNNER_SHA:
        raise RuntimeError("exact failed runner SHA mismatch")
    failed_runner.OUTPUT = output
    original_install = failed_runner.install_bindings

    def install_bindings(role):
        base, cfg, defaults = original_install(role)
        base.OUTPUT = output
        base.ACTIVE_OUTPUT = output
        base.CHECKPOINT_TARGETS = (500, 1000)
        base.__file__ = str(Path(__file__).resolve())
        return base, cfg, defaults

    failed_runner.install_bindings = install_bindings
    sys.argv = [sys.argv[0], *runner_argv]
    return failed_runner.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
