"""Offline adjudication of persisted Link01 exclusion evidence; no simulator import."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_peg_axial_bounded_diagnostic import durable_save
from training.peg_insertion.formal_policy import bindings as formal_bindings
from training.peg_insertion.v2_recorder import bindings as v2_bindings, sha

OUT = ROOT / "results/validation/peg/f3_link01_exclusion_fix_20260905"
SOURCE_REL = "external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/assets/models/manipulators/Aubo_i5/Aubo_i5.xml"
BACKUP_REL = SOURCE_REL + ".before_link01_exclusion_20260905"
TARGET = ("base_link", "shoulder_Link")


def pairs(path):
    return [(node.get("body1"), node.get("body2")) for node in ET.parse(path).getroot().findall("./contact/exclude")]


def main():
    source = json.loads((OUT / "source_audit.json").read_text())
    generated = json.loads((OUT / "generation_audit.json").read_text())
    compiled = json.loads((OUT / "compile_only.json").read_text())
    reset = json.loads((OUT / "reset_contact_validation.json").read_text())
    snapshot = json.loads((OUT / "initial_snapshot.json").read_text())
    assert source["only_authorized_pair_changed"] and generated["only_authorized_pair_changed"]
    assert pairs(Path(source["source"])) == [TARGET] and pairs(Path(generated["generated"])) == [TARGET]
    assert compiled["status"] == "PASS" and not compiled["reset_called"] and not compiled["step_called"]
    assert reset["status"] == "PASS" and reset["link0_link1_contact_count"] == 0
    assert reset["peg_hole_reset_contact_count"] == 0 and reset["robot_fixture_reset_contact_count"] == 0
    assert not reset["initial_guard_triggered"] and reset["simulator_actions_executed"] == 0
    assert snapshot["pre_action"] and len(snapshot["contacts"]) == 0
    assert snapshot["initial"]["observation_contract_version"] == "PEG_OBS_CONTRACT_V2_SENSOR_FIXED"
    assert len(snapshot["initial"]["obs"]) == 20

    old_v2 = json.loads((ROOT / "results/demos/peg_v2_pilot/standard_20260905_001/episode.json").read_text())["hashes"]
    new_v2 = v2_bindings()
    changed_common = sorted(k for k, value in old_v2["files"].items() if new_v2["files"].get(k) != value)
    added = sorted(set(new_v2["files"]) - set(old_v2["files"]))
    assert changed_common == [SOURCE_REL] and added == [BACKUP_REL]

    historical_formal = json.loads((ROOT / "results/validation/peg/formal_readiness_v4.json").read_text())["bindings"]
    expected_formal = {
        "runtime_hash": old_v2["runtime_hash"], "config_hash": old_v2["config_hash"],
        "demo_hash": historical_formal["demo_hash"],
        "demo_manifest_hash": historical_formal["demo_manifest_hash"],
    }
    current_formal = formal_bindings()
    assert current_formal == expected_formal

    protected = json.loads((ROOT / "results/validation/peg/f3_stage2_20260904/checkpoint_9000/protected_after.json").read_text())
    mismatches = {path: {"historical": expected, "current": sha(Path(path))}
                  for path, expected in protected.items() if sha(Path(path)) != expected}
    stale_allowed = {
        str(ROOT / "configs/peg_insertion/peg_rlpd_baseline_seed0.yaml"),
        str(ROOT / "evaluation/peg_insertion/evaluate_policy.py"),
    }
    assert set(mismatches) == stale_allowed
    checkpoint_entries = {path: expected for path, expected in protected.items() if "/results/checkpoints/" in path}
    assert checkpoint_entries and all(sha(Path(path)) == expected for path, expected in checkpoint_entries.items())

    regression = {
        "status": "PASS", "adjudication": "OFFLINE_FROM_DURABLE_RESET_EVIDENCE_NO_NEW_RESET",
        "expected_environment_file_change": SOURCE_REL, "expected_backup_addition": BACKUP_REL,
        "unexpected_environment_changes": [], "stale_historical_manifest_entries": mismatches,
        "stale_entry_reason": "Both files changed in earlier SensorFix/evaluator work before this task; current V2/formal bindings are used as the task baseline.",
        "controller_unchanged": True, "ik_unchanged": True, "sensor_unchanged": True,
        "reward_unchanged": True, "success_unchanged": True, "action_unchanged": True,
        "v2_obs_schema_unchanged": True, "demo_unchanged": True, "checkpoint_unchanged": True,
        "formal_bindings": current_formal, "protected_checkpoint_file_count": len(checkpoint_entries),
        "additional_simulator_reset": False, "additional_simulator_action": False,
    }
    durable_save(OUT / "regression.json", regression)
    result = {
        "PEG_F3_LINK01_EXCLUSION_FIX": "PASS",
        "PEG_F3_LINK01_XML_COMPILE": "PASS",
        "PEG_F3_LINK01_RESET_CONTACT_VALIDATION": "PASS",
        "SOURCE_LINK01_EXCLUSION": "PASS", "GENERATED_LINK01_EXCLUSION": "PASS",
        "LINK0_LINK1_CONTACT_AFTER_FIX": "NO", "INITIAL_GUARD_TRIGGERED_AFTER_FIX": "NO",
        "AXIAL_DIAGNOSTIC": "READY", "demo_recording_started": False,
        "training_started": False, "simulator_actions_executed": 0,
        "reset_evidence_adjudication": "PASS_FROM_PERSISTED_RESET_NO_RERUN",
    }
    durable_save(OUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
