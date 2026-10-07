"""Import-safe corrected entry point for the Obs20 dynamics audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/makoto/flexible_line_project")
sys.path.insert(0, str(ROOT))

import scripts.run_peg_f3_v2_obs20_short_horizon_dynamics_model_feasibility as base

base.OUT = ROOT / "results/diagnostics/peg/f3_v2_obs20_short_horizon_dynamics_model_feasibility_20260909_002"


def corrected_main():
    original_ranges = base.basin.episode_ranges
    original_row_provenance = base.basin.row_provenance

    def indexed_ranges(demos):
        return [(index, start, end) for index, (start, end) in enumerate(original_ranges(demos))]

    def row_provenance(ranges, row):
        pairs = [(start, end) for _, start, end in ranges]
        return original_row_provenance(pairs, row)

    base.basin.episode_ranges = indexed_ranges
    base.basin.row_provenance = row_provenance
    base.main()


if __name__ == "__main__":
    corrected_main()
