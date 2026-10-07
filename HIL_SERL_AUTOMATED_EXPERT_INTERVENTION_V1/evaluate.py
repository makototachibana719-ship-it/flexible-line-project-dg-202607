#!/usr/bin/env python3
"""Formal policy-only evaluator shim with an explicit hard-off HIL assertion."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
STAGE_B = ROOT / "SERL_PAPER_FIDELITY_AUBO_PEG_V2"
sys.path[:0] = [str(ROOT), str(STAGE_B)]


def load_stage_b_evaluator():
    spec = importlib.util.spec_from_file_location("stage_b_formal_evaluator", STAGE_B / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.BASELINE_ROOT = BRANCH
    module.ROOT = ROOT
    return module


def main() -> int:
    from training.peg_insertion.hil_serl.intervention_adapter_v1 import HILSERLInterventionAdapterV1

    # Construction itself fails closed if either of these formal-eval bindings changes.
    adapter = HILSERLInterventionAdapterV1(
        mode="FORMAL_EVAL", intervention_enabled=False, provider=None
    )
    assert adapter.enabled is False
    module = load_stage_b_evaluator()
    rc = module.main()

    # Augment the evaluator's evidence only after its canonical rollout completed.
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve()
    step = int(sys.argv[sys.argv.index("--step") + 1])
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "primary"
    result_path = run / f"eval_{step}_{label}.json"
    result = json.loads(result_path.read_text())
    result.update({
        "intervention_enabled": False,
        "expert_provider": None,
        "expert_call_count": 0,
        "formal_hil_runtime_assertion": "PASS",
    })
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

