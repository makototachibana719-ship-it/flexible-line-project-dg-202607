"""Authorized one-shot entry point with a fresh exclusive output directory."""
from pathlib import Path

import run_peg_axial_bounded_diagnostic_retry as runner

runner.OUT = Path(runner.ROOT) / "results/diagnostics/peg/f3_axial_bounded_final_retry_20260905"

if __name__ == "__main__":
    runner.main()
