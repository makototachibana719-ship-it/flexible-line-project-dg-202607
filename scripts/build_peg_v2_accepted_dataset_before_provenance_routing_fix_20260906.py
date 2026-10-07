"""Build the canonical Peg V2 dataset from the explicitly accepted 11 episodes."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serl_launcher.data.data_store import ReplayBufferDataStore
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.v2_recorder import bindings, validate_snapshot, validate_transition

OUTPUT = ROOT / "results/datasets/peg_v2_accepted_11_20260906"
DATASET_NAME = "peg_v2_accepted_11.pkl"
FIVE_MANIFEST = ROOT / "results/validation/peg/f3_v2_five_pilot_audit_20260906/manifest.json"

PLAN_PLUS = "results/validation/peg/f3_lateral_plus_auto_closure_20260905/command_plan.json"
PLAN_MINUS = "results/validation/peg/f3_lateral_minus_command_plan_20260905/command_plan.json"
PLAN_ROLL = "results/validation/peg/f3_rotation_roll_command_plan_20260905/command_plan.json"
PLAN_PITCH = "results/validation/peg/f3_rotation_pitch_revised_plan_20260906/command_plan_v2.json"

PLAN_HASHES = {
    PLAN_PLUS: "a7494bc04c469012dad7ca36f23f7e322ffcfd51d93f4f5627384440820aeea7",
    PLAN_MINUS: "dcf3a7981098935cc8fe919a99e82fcad2c7e81783dd15c9047305bc439fef81",
    PLAN_ROLL: "2c5e9f3ec9a165cab794a883b1a549e87248eb152d41bba0b29c25d698847e3d",
    PLAN_PITCH: "5e2aca80ac53564584faa2c3419e910f5c541d5351b31ac99fed258cb84e1415",
}

EPISODES = (
    ("standard", "standard", "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001", "results/diagnostics/peg/f3_v2_standard_rerecord_contact_20260905_001", None, 165, None),
    ("standard", "standard", "results/demos/peg_v2_pilot/standard_post_exclusion_20260906_002", "results/diagnostics/peg/f3_v2_formal_standard_contact_20260906_002", None, 165, "results/validation/peg/f3_v2_formal_standard_20260906_002/REPORT.md"),
    ("standard", "standard", "results/demos/peg_v2_pilot/standard_post_exclusion_20260906_003", "results/diagnostics/peg/f3_v2_formal_standard_contact_20260906_003", None, 165, "results/validation/peg/f3_v2_formal_standard_second_20260906_003/REPORT.md"),
    ("lateral_plus", "lateral_plus_y", "results/demos/peg_v2_pilot/lateral_plus_y_post_exclusion_20260905_003", "results/diagnostics/peg/f3_v2_lateral_plus_contact_20260905_003", PLAN_PLUS, 176, None),
    ("lateral_plus", "lateral_plus_y", "results/demos/peg_v2_pilot/lateral_plus_y_post_exclusion_20260906_004", "results/diagnostics/peg/f3_v2_formal_lateral_plus_contact_20260906_004", PLAN_PLUS, 176, "results/validation/peg/f3_v2_formal_lateral_plus_20260906_004/REPORT.md"),
    ("lateral_minus", "lateral_minus_y", "results/demos/peg_v2_pilot/lateral_minus_y_post_exclusion_20260905_001", "results/diagnostics/peg/f3_v2_lateral_minus_contact_20260905_001", PLAN_MINUS, 176, None),
    ("lateral_minus", "lateral_minus_y", "results/demos/peg_v2_pilot/lateral_minus_y_post_exclusion_20260906_002", "results/diagnostics/peg/f3_v2_formal_lateral_minus_contact_20260906_002", PLAN_MINUS, 176, "results/validation/peg/f3_v2_formal_lateral_minus_20260906_002/REPORT.md"),
    ("rotation_roll", "orientation_roll", "results/demos/peg_v2_pilot/rotation_roll_post_exclusion_20260905_001", "results/diagnostics/peg/f3_v2_rotation_roll_contact_20260905_001", PLAN_ROLL, 179, None),
    ("rotation_roll", "orientation_roll", "results/demos/peg_v2_pilot/rotation_roll_post_exclusion_20260906_002", "results/diagnostics/peg/f3_v2_formal_rotation_roll_contact_20260906_002", PLAN_ROLL, 179, "results/validation/peg/f3_v2_formal_rotation_roll_20260906_002/REPORT.md"),
    ("rotation_pitch", "orientation_pitch", "results/demos/peg_v2_pilot/rotation_pitch_post_exclusion_20260906_002", "results/diagnostics/peg/f3_v2_rotation_pitch_contact_20260906_002", PLAN_PITCH, 182, None),
    ("rotation_pitch", "orientation_pitch", "results/demos/peg_v2_pilot/rotation_pitch_post_exclusion_20260906_003", "results/diagnostics/peg/f3_v2_formal_rotation_pitch_contact_20260906_003", PLAN_PITCH, 182, "results/validation/peg/f3_v2_formal_rotation_pitch_20260906_003/REPORT.md"),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_commands(relative: str):
    path = ROOT / relative
    require(sha(path) == PLAN_HASHES[relative], f"canonical plan hash mismatch: {relative}")
    value = json.loads(path.read_text())
    return value["commands"] if isinstance(value, dict) else value


def validate_plan(rows, relative):
    commands = plan_commands(relative)
    require(len(commands) == len(rows), "canonical plan count mismatch")
    for index, (command, row) in enumerate(zip(commands, rows)):
        require(np.allclose(np.asarray(command["action"], dtype=np.float32),
                            np.asarray(row["action"], dtype=np.float32), rtol=0, atol=0),
                f"canonical action mismatch at {index}")
        phase_ok = row["phase"] == command["phase"]
        terminal_ok = index == len(rows) - 1 and row["phase"] == "success" and command["phase"] == "insertion"
        require(phase_ok or terminal_ok, f"canonical phase mismatch at {index}")


def training_transition(row):
    obs = np.asarray(row["obs"], dtype=np.float32)
    action = np.asarray(row["action"], dtype=np.float32)
    next_obs = np.asarray(row["next_obs"], dtype=np.float32)
    require(obs.shape == (20,) and next_obs.shape == (20,), "V2 observation shape")
    require(action.shape == (7,), "action shape")
    require(obs.dtype == action.dtype == next_obs.dtype == np.float32, "float32 contract")
    require(np.isfinite(obs).all() and np.isfinite(action).all() and np.isfinite(next_obs).all(), "nonfinite data")
    require(obs.astype(float).tolist() == row["obs"], "observation float32 provenance loss")
    require(action.astype(float).tolist() == row["action"], "action float32 provenance loss")
    require(next_obs.astype(float).tolist() == row["next_obs"], "next observation float32 provenance loss")
    terminated = bool(row["terminated"])
    truncated = bool(row["truncated"])
    return {
        "observations": obs,
        "actions": action,
        "next_observations": next_obs,
        "rewards": np.float32(row["reward"]),
        "masks": np.float32(0.0 if terminated else 1.0),
        "dones": bool(terminated or truncated),
    }


def validate_episode(spec, current_binding, five_manifest, dataset, provenance):
    label, episode_type, episode_relative, contact_relative, plan_relative, expected_count, report_relative = spec
    episode_path = ROOT / episode_relative
    contact_path = ROOT / contact_relative
    require(episode_path.is_dir() and contact_path.is_dir(), f"missing accepted evidence: {episode_relative}")
    status = json.loads((episode_path / "status.json").read_text())
    episode = json.loads((episode_path / "episode.json").read_text())
    rows = [json.loads(line) for line in (episode_path / "transitions.jsonl").open()]
    require(status.get("status") == "COMPLETED_SUCCESS", "episode not completed-success")
    require(status.get("transitions") == expected_count == len(rows), "episode transition count")
    transitions_hash = sha(episode_path / "transitions.jsonl")
    require(status.get("transitions_sha256") == transitions_hash, "episode stream hash")
    require(episode.get("episode_type") == episode_type, "episode type provenance")
    require(episode.get("hashes") == current_binding, "production hash binding")
    previous = episode["initial"]
    validate_snapshot(previous)
    for index, row in enumerate(rows, 1):
        validate_transition(row, previous, index, episode_type, current_binding)
        previous = row["after"]
    require(rows[-1]["success"] is True and rows[-1]["truncated"] is False, "terminal acceptance")
    require(sum(bool(row["success"]) for row in rows) == 1, "exactly one terminal success required")
    require(not any(bool(row["truncated"]) for row in rows), "truncated transition excluded")

    contact_summary = json.loads((contact_path / "summary.json").read_text())
    contact_rows = [json.loads(line) for line in (contact_path / "contacts.jsonl").open()]
    require(contact_summary.get("steps") == expected_count == len(contact_rows), "contact row count")
    require(not contact_summary.get("link01_contact_seen") and not contact_summary.get("unexpected_contact_seen"), "contact health")
    require(not any(row.get("link01_contact") or row.get("unexpected_contact") for row in contact_rows), "contact row health")

    if plan_relative:
        validate_plan(rows, plan_relative)
    else:
        pilot = five_manifest["pilots"]["standard"]
        require(pilot["episode"] == episode_relative and pilot["transitions_sha256"] == transitions_hash,
                "standard frozen-stream provenance")
    if report_relative:
        report = (ROOT / report_relative).read_text()
        gate = "PEG_F3_V2_FORMAL_" + label.upper()
        if label == "standard":
            gate = "PEG_F3_V2_FORMAL_STANDARD"
        require(f"{gate}=PASS" in report or (label == "standard" and "PEG_F3_V2_FORMAL_STANDARD_SECOND=PASS" in report),
                f"formal acceptance report mismatch: {report_relative}")
    else:
        pilot_key = label
        require(five_manifest["pilots"][pilot_key]["episode"] == episode_relative,
                f"five-pilot allowlist mismatch: {label}")

    start = len(dataset)
    dataset.extend(training_transition(row) for row in rows)
    provenance.append({
        "episode_index": len(provenance),
        "type": label,
        "recorder_episode_type": episode_type,
        "episode_path": episode_relative,
        "contact_path": contact_relative,
        "transition_count": expected_count,
        "dataset_start": start,
        "dataset_stop_exclusive": len(dataset),
        "transitions_sha256": transitions_hash,
        "episode_json_sha256": sha(episode_path / "episode.json"),
        "status_json_sha256": sha(episode_path / "status.json"),
        "contact_summary_sha256": sha(contact_path / "summary.json"),
        "canonical_plan": plan_relative or "STANDARD_FROZEN_PROTOCOL",
        "canonical_plan_sha256": PLAN_HASHES.get(plan_relative, transitions_hash),
        "acceptance_authority": report_relative or str(FIVE_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
    })


def replay_smoke(payload):
    loaded = pickle.loads(payload)
    require(isinstance(loaded, list) and len(loaded) == 1921, "loader transition count")
    observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
    with repaired_replay_datastore_insert():
        replay = ReplayBufferDataStore(observation_space, action_space, len(loaded) + 1)
        for transition in loaded:
            replay.insert(transition)
        sample = replay.sample(32)
    require(sample["observations"].shape == (32, 20), "sample observation shape")
    require(sample["next_observations"].shape == (32, 20), "sample next observation shape")
    require(sample["actions"].shape == (32, 7), "sample action shape")
    require(np.isfinite(sample["observations"]).all() and np.isfinite(sample["actions"]).all(), "sample finite")
    return len(loaded), int(sample["observations"].shape[0])


def write_exclusive(path: Path, data: bytes):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    require(not OUTPUT.exists(), "dataset output already exists")
    five = json.loads(FIVE_MANIFEST.read_text())
    require(five.get("status") == "PASS" and five.get("accepted_pilots") == 5, "five-pilot gate")
    current_binding = bindings(ROOT)
    dataset, provenance = [], []
    for spec in EPISODES:
        validate_episode(spec, current_binding, five, dataset, provenance)
    require(len(provenance) == 11 and len(dataset) == 1921, "accepted dataset cardinality")
    episode_counts = {name: sum(item["type"] == name for item in provenance)
                      for name in ("standard", "lateral_plus", "lateral_minus", "rotation_roll", "rotation_pitch")}
    require(episode_counts == {"standard": 3, "lateral_plus": 2, "lateral_minus": 2,
                               "rotation_roll": 2, "rotation_pitch": 2}, "episode type counts")
    transition_counts = {name: sum(item["transition_count"] for item in provenance if item["type"] == name)
                         for name in episode_counts}
    payload = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
    loaded_count, sample_size = replay_smoke(payload)
    dataset_hash = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "PEG_F3_V2_ACCEPTED_DATASET_V1",
        "status": "PASS",
        "v2_only": True,
        "dataset_file": DATASET_NAME,
        "dataset_sha256": dataset_hash,
        "episodes": len(provenance),
        "transitions": len(dataset),
        "episode_counts": episode_counts,
        "transition_counts": transition_counts,
        "production_hashes": {key: current_binding[key] for key in (
            "source_hash", "config_hash", "runtime_hash", "env_hash", "recorder_hash")},
        "five_pilot_manifest": str(FIVE_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "five_pilot_manifest_sha256": sha(FIVE_MANIFEST),
        "loader_test": "PASS",
        "replay_sample_test": "PASS",
        "sample_batch_size": sample_size,
        "training_started": False,
        "checkpoint_resumed": False,
    }
    summary = {
        "PEG_F3_V2_DATASET": "PASS",
        "episodes": 11,
        "transitions": len(dataset),
        **episode_counts,
        "V2_ONLY": "PASS",
        "DATASET_SCHEMA": "PASS",
        "PROVENANCE": "PASS",
        "HASH_BINDING": "PASS",
        "LOADER_TEST": "PASS",
        "REPLAY_SAMPLE_TEST": "PASS",
        "DATASET_PATH": str((OUTPUT / DATASET_NAME).relative_to(ROOT)).replace("\\", "/"),
        "DATASET_SHA256": dataset_hash,
        "TRAINING_SMOKE": "READY",
        "loaded_transitions": loaded_count,
        "training_started": False,
        "checkpoint_resumed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_exclusive(OUTPUT / DATASET_NAME, payload)
    write_exclusive(OUTPUT / "episode_provenance.json", json.dumps(provenance, indent=2).encode())
    write_exclusive(OUTPUT / "manifest.json", json.dumps(manifest, indent=2).encode())
    write_exclusive(OUTPUT / "dataset_summary.json", json.dumps(summary, indent=2).encode())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
