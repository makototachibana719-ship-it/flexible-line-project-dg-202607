"""One-pair source/generation/compile/reset validation; never dispatches an action."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_peg_axial_bounded_diagnostic import durable_save, persist_initial_evidence_and_guard
from training.peg_insertion.axial_telemetry import AxialTelemetry
from training.peg_insertion.formal_policy import bindings as formal_bindings
from training.peg_insertion.v2_recorder import PKG, bindings as v2_bindings, sha, snapshot, validate_snapshot

SOURCE_REL = "external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/assets/models/manipulators/Aubo_i5/Aubo_i5.xml"
SOURCE = ROOT / SOURCE_REL
SOURCE_BACKUP = SOURCE.with_name(SOURCE.name + ".before_link01_exclusion_20260905")
GENERATED = ROOT / "runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml"
GENERATED_BACKUP = GENERATED.with_name(GENERATED.name + ".before_link01_exclusion_20260905")
OUT = ROOT / "results/validation/peg/f3_link01_exclusion_fix_20260905"
TARGET = ("base_link", "shoulder_Link")
ROBOT_BODIES = {"base_link", "shoulder_Link", "upperArm_Link", "foreArm_Link", "wrist1_Link", "wrist2_Link", "wrist3_Link"}


def excludes(path: Path):
    return [(node.get("body1"), node.get("body2")) for node in ET.parse(path).getroot().findall("./contact/exclude")]


def without_target(path: Path) -> bytes:
    root = ET.parse(path).getroot()
    for contact in list(root.findall("contact")):
        for node in list(contact):
            if node.tag == "exclude" and (node.get("body1"), node.get("body2")) == TARGET:
                contact.remove(node)
        if len(contact) == 0:
            root.remove(contact)
    # ElementTree preserves indentation as text/tail nodes. Generation may
    # re-indent the closing root when a section is added, which is not a model
    # semantic change. Compare the parsed XML after removing blank layout only.
    for node in root.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        if node.tail is not None and not node.tail.strip():
            node.tail = None
    return ET.tostring(root, encoding="utf-8")


def geom_meta(model, geom_id: int):
    body_id = int(model.geom_bodyid[geom_id])
    data_id = int(model.geom_dataid[geom_id])
    return {
        "geom_id": geom_id,
        "geom_name": model.geom(geom_id).name,
        "body_id": body_id,
        "body_name": model.body(body_id).name,
        "mesh_name": model.mesh(data_id).name if int(model.geom_type[geom_id]) == 7 and data_id >= 0 else None,
    }


def enrich_contacts(model, contacts):
    rows = []
    for contact in contacts:
        row = dict(contact)
        row["geom1"] = geom_meta(model, int(contact["geom1_id"]))
        row["geom2"] = geom_meta(model, int(contact["geom2_id"]))
        rows.append(row)
    return rows


def is_link01(row):
    return {row["geom1"]["body_name"], row["geom2"]["body_name"]} == set(TARGET)


def is_peg_hole(row):
    text = " ".join(str(row[x][k]).lower() for x in ("geom1", "geom2") for k in ("geom_name", "body_name", "mesh_name"))
    return ("hole" in text) and ("peg" in text or "axle" in text)


def is_robot_fixture(row):
    bodies = {row["geom1"]["body_name"], row["geom2"]["body_name"]}
    one_robot = len(bodies & ROBOT_BODIES) == 1
    text = " ".join(str(row[x][k]).lower() for x in ("geom1", "geom2") for k in ("geom_name", "body_name", "mesh_name"))
    return one_robot and any(word in text for word in ("fixture", "pedestal", "table", "mount", "hole"))


def load_runtime():
    for line in (ROOT / "configs/peg_insertion/runtime_gpu_formal.env").read_text().splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            os.environ[key] = value


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    result = {
        "PEG_F3_LINK01_EXCLUSION_FIX": "FAIL",
        "PEG_F3_LINK01_XML_COMPILE": "FAIL",
        "PEG_F3_LINK01_RESET_CONTACT_VALIDATION": "FAIL",
        "SOURCE_LINK01_EXCLUSION": "FAIL",
        "GENERATED_LINK01_EXCLUSION": "FAIL",
        "LINK0_LINK1_CONTACT_AFTER_FIX": "UNRESOLVED",
        "INITIAL_GUARD_TRIGGERED_AFTER_FIX": "UNRESOLVED",
        "AXIAL_DIAGNOSTIC": "BLOCKED",
        "demo_recording_started": False,
        "training_started": False,
        "simulator_actions_executed": 0,
    }
    env = None
    observer = None
    try:
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        forbidden = ("scripts/record_peg_v2_demo.py", "scripts/run_peg_training.py", "scripts/run_peg_axial_bounded_diagnostic.py")
        conflicts = [line for line in processes.splitlines() if any(token in line for token in forbidden)]
        if conflicts:
            raise RuntimeError("conflicting process: " + conflicts[0])

        source_pairs = excludes(SOURCE)
        if source_pairs != [TARGET]:
            raise RuntimeError(f"source exclusions are not exactly the authorized pair: {source_pairs}")
        if without_target(SOURCE) != without_target(SOURCE_BACKUP):
            raise RuntimeError("source MJCF changed beyond the authorized exclusion")
        result["SOURCE_LINK01_EXCLUSION"] = "PASS"
        durable_save(OUT / "source_audit.json", {
            "source": str(SOURCE), "backup": str(SOURCE_BACKUP), "source_sha256": sha(SOURCE),
            "backup_sha256": sha(SOURCE_BACKUP), "exclusions": source_pairs,
            "only_authorized_pair_changed": True,
        })

        load_runtime()
        sys.path[:0] = [str(PKG / "code/example/async_rlpd_peg_sim"), str(PKG / "code/env")]
        from mujoco_env.mujoco_env.tasks.peg_insertion.aubo_i5_config import AuboI5Config
        from mujoco_env.mujoco_env.tasks.peg_insertion.peg_insertion import PegInsertionTask
        from peg_env_adapter import make_peg_state_env
        from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

        with portable_peg_runtime(ROOT):
            task = PegInsertionTask(
                robot_config=AuboI5Config.get_config(), scene_name="assemble_hole",
                include_image=False, image_size=(128, 128), include_depth=False,
                ee_site_name="attachment_site",
            )
        if Path(task.model_path).resolve() != GENERATED.resolve():
            raise RuntimeError(f"unexpected generated path: {task.model_path}")
        generated_pairs = excludes(GENERATED)
        if generated_pairs != [TARGET]:
            raise RuntimeError(f"generated exclusions are not exactly the authorized pair: {generated_pairs}")
        if without_target(GENERATED) != without_target(GENERATED_BACKUP):
            raise RuntimeError("generated XML changed beyond the authorized exclusion")
        result["GENERATED_LINK01_EXCLUSION"] = "PASS"
        durable_save(OUT / "generation_audit.json", {
            "generated": str(GENERATED), "backup": str(GENERATED_BACKUP),
            "generated_sha256": sha(GENERATED), "backup_sha256": sha(GENERATED_BACKUP),
            "exclusions": generated_pairs, "only_authorized_pair_changed": True,
        })

        import mujoco
        compiled = mujoco.MjModel.from_xml_path(str(GENERATED))
        durable_save(OUT / "compile_only.json", {
            "status": "PASS", "xml": str(GENERATED), "nq": int(compiled.nq),
            "nv": int(compiled.nv), "ngeom": int(compiled.ngeom), "nexclude": int(compiled.nexclude),
            "env_created": False, "reset_called": False, "step_called": False,
        })
        del compiled
        result["PEG_F3_LINK01_XML_COMPILE"] = "PASS"

        with portable_peg_runtime(ROOT):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        env.reset(seed=0)
        initial = snapshot(env)
        validate_snapshot(initial)
        # snapshot() intentionally normalizes arrays to JSON-safe lists.
        if len(initial["obs"]) != 20 or env.observation_contract_version != "PEG_OBS_CONTRACT_V2_SENSOR_FIXED":
            raise RuntimeError("V2 observation contract changed")
        observer = AxialTelemetry(env, OUT / "telemetry")
        guard_triggered = False
        try:
            persist_initial_evidence_and_guard(observer, initial, OUT)
        except RuntimeError as exc:
            if str(exc) != "dangerous initial penetration":
                raise
            guard_triggered = True
        persisted = json.loads((OUT / "initial_snapshot.json").read_text())
        contacts = enrich_contacts(env.env.model, persisted["contacts"])
        link01 = [row for row in contacts if is_link01(row)]
        peg_hole = [row for row in contacts if is_peg_hole(row)]
        robot_fixture = [row for row in contacts if is_robot_fixture(row)]
        if link01 or peg_hole or robot_fixture or guard_triggered or observer.count != 0:
            raise RuntimeError("reset-only contact validation failed")
        durable_save(OUT / "reset_contact_validation.json", {
            "status": "PASS", "contacts": contacts, "total_contact_count": len(contacts),
            "link0_link1_contact_count": len(link01), "peg_hole_reset_contact_count": len(peg_hole),
            "robot_fixture_reset_contact_count": len(robot_fixture),
            "initial_guard_triggered": guard_triggered, "guard_logic": "unchanged distance < -0.005 m",
            "simulator_actions_executed": 0, "observer_step_count": observer.count,
        })
        result["PEG_F3_LINK01_RESET_CONTACT_VALIDATION"] = "PASS"
        result["LINK0_LINK1_CONTACT_AFTER_FIX"] = "NO"
        result["INITIAL_GUARD_TRIGGERED_AFTER_FIX"] = "NO"

        old_v2 = json.loads((ROOT / "results/demos/peg_v2_pilot/standard_20260905_001/episode.json").read_text())["hashes"]
        new_v2 = v2_bindings()
        changed_common = sorted(k for k, value in old_v2["files"].items() if new_v2["files"].get(k) != value)
        added = sorted(set(new_v2["files"]) - set(old_v2["files"]))
        expected_backup_rel = SOURCE_REL + ".before_link01_exclusion_20260905"
        if changed_common != [SOURCE_REL] or added != [expected_backup_rel]:
            raise RuntimeError(f"unexpected environment binding differences: changed={changed_common}, added={added}")
        historical_formal = json.loads((ROOT / "results/validation/peg/formal_readiness_v4.json").read_text())["bindings"]
        expected_formal = {
            "runtime_hash": old_v2["runtime_hash"],
            "config_hash": old_v2["config_hash"],
            "demo_hash": historical_formal["demo_hash"],
            "demo_manifest_hash": historical_formal["demo_manifest_hash"],
        }
        new_formal = formal_bindings()
        if new_formal != expected_formal:
            raise RuntimeError("config/runtime/Demo formal bindings changed")
        protected = json.loads((ROOT / "results/validation/peg/f3_stage2_20260904/checkpoint_9000/protected_after.json").read_text())
        protected_mismatch = {path: [expected, sha(Path(path))] for path, expected in protected.items() if sha(Path(path)) != expected}
        if protected_mismatch:
            raise RuntimeError(f"checkpoint/replay protection mismatch: {protected_mismatch}")
        durable_save(OUT / "regression.json", {
            "status": "PASS", "expected_environment_file_change": SOURCE_REL,
            "expected_backup_addition": expected_backup_rel, "unexpected_changes": [],
            "controller_unchanged": True, "ik_unchanged": True, "sensor_unchanged": True,
            "reward_unchanged": True, "success_unchanged": True, "action_unchanged": True,
            "v2_obs_schema_unchanged": True, "demo_unchanged": True, "checkpoint_unchanged": True,
            "formal_bindings": new_formal, "protected_checkpoint_file_count": len(protected),
        })
        result["PEG_F3_LINK01_EXCLUSION_FIX"] = "PASS"
        result["AXIAL_DIAGNOSTIC"] = "READY"
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        if observer is not None:
            observer.close()
        if env is not None:
            env.close()
        durable_save(OUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
