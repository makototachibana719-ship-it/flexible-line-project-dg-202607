#!/usr/bin/env python3
"""R2 wrapper for the fixed-prefill checkpoint recovery audit."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path("/home/makoto/flexible_line_project")
sys.path.insert(0, str(ROOT))

import scripts.recover_r1_fixed_prefill_checkpoint as recovery

recovery.RUN = ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica2"


def unique_out() -> Path:
    stem = "f3_v2_r2_fixed_prefill_checkpoint_recovery_20260907"
    for idx in range(1, 100):
        out = recovery.OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


recovery.unique_out = unique_out


if __name__ == "__main__":
    recovery.main()
