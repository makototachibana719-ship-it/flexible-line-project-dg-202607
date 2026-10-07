"""Corrected entry point for the identical Obs20 dynamics feasibility audit."""
from __future__ import annotations

import scripts.run_peg_f3_v2_obs20_short_horizon_dynamics_model_feasibility as base

base.OUT = base.ROOT / "results/diagnostics/peg/f3_v2_obs20_short_horizon_dynamics_model_feasibility_20260909_002"


def corrected_main():
    original = base.basin.episode_ranges

    def indexed_ranges(demos):
        return [(index, start, end) for index, (start, end) in enumerate(original(demos))]

    # The base script uses the indexed form only for its held-out report and
    # passes the value to row_provenance.  Restore the original pair form for
    # every row_provenance call so scientific behavior remains unchanged.
    original_row_provenance = base.basin.row_provenance

    def row_provenance(ranges, row):
        pairs = [(start, end) for _, start, end in ranges]
        return original_row_provenance(pairs, row)

    base.basin.episode_ranges = indexed_ranges
    base.basin.row_provenance = row_provenance
    base.main()


if __name__ == "__main__":
    corrected_main()
