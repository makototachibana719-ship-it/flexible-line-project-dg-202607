"""CPU/static provenance-route regression; never builds a dataset."""
import copy
import hashlib
import json
from pathlib import Path

from build_peg_v2_accepted_dataset import (
    EPISODES, FIVE_MANIFEST, OUTPUT, ROOT, V2, validate_provenance_route,
)


def stream_hash(relative):
    path = ROOT / relative / "transitions.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route(spec, **changes):
    label, episode_type, episode_relative, contact_relative, plan_relative, count, report_relative, source_kind = spec
    episode = json.loads((ROOT / episode_relative / "episode.json").read_text())
    values = dict(source_kind=source_kind, label=label, episode_relative=episode_relative,
                  transitions_hash=stream_hash(episode_relative), report_relative=report_relative,
                  plan_relative=plan_relative, observation_contract=episode["observation_contract_version"],
                  five_manifest=json.loads(FIVE_MANIFEST.read_text()))
    values.update(changes)
    return validate_provenance_route(**values)


def rejected(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return True
    return False


def main():
    assert not OUTPUT.exists(), "dataset output must remain absent"
    for spec in EPISODES:
        route(spec)
    pilot_standard = EPISODES[0]
    formal_standard_2, formal_standard_3 = EPISODES[1], EPISODES[2]
    assert route(pilot_standard).endswith("manifest.json")
    assert route(formal_standard_2).endswith("REPORT.md")
    assert route(formal_standard_3).endswith("REPORT.md")
    assert rejected(lambda: route(pilot_standard, source_kind="unknown"))
    assert rejected(lambda: route(pilot_standard, manifest_path=Path("missing/manifest.json")))
    assert rejected(lambda: route(formal_standard_2, transitions_hash="0" * 64))
    assert rejected(lambda: route(pilot_standard, observation_contract="PEG_OBS_CONTRACT_V1"))
    assert not OUTPUT.exists(), "static test published dataset"
    print(json.dumps({
        "PEG_F3_V2_PROVENANCE_ROUTING_FIX": "PASS",
        "FIVE_PILOT_STANDARD_ROUTE": "PASS",
        "FORMAL_STANDARD_ROUTE": "PASS",
        "LATERAL_PLUS_ROUTE": "PASS",
        "LATERAL_MINUS_ROUTE": "PASS",
        "ROTATION_ROLL_ROUTE": "PASS",
        "ROTATION_PITCH_ROUTE": "PASS",
        "WRONG_SOURCE_REJECTION": "PASS",
        "WRONG_MANIFEST_REJECTION": "PASS",
        "WRONG_HASH_REJECTION": "PASS",
        "V1_REJECTION": "PASS",
        "DATASET_REBUILD": "READY",
        "dataset_published": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
