#!/usr/bin/env python3
"""Build the immutable-source local insertion supervision subset."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/home/makoto/flexible_line_project")
SOURCE_DIR = ROOT / "results/datasets/peg_v2_accepted_11_20260906"
SOURCE = SOURCE_DIR / "peg_v2_accepted_11.pkl"
PROVENANCE = SOURCE_DIR / "episode_provenance.json"
OUTPUT = ROOT / "results/datasets/peg_v2_local_insertion_supervision_20260908_001"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
V2 = "PEG_OBS_CONTRACT_V2_SENSOR_FIXED"


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_exclusive(path: Path, data: bytes):
    with path.open("xb") as f: f.write(data)


def transition_identity(row):
    h = hashlib.sha256()
    for key in ("observations", "actions", "next_observations", "rewards", "masks", "dones"):
        a = np.asarray(row[key]); h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main():
    require(not OUTPUT.exists(), "fresh local supervision output required")
    require(sha(SOURCE) == DEMO_SHA, "production Demo SHA")
    source_rows = pickle.loads(SOURCE.read_bytes())
    provenance = json.loads(PROVENANCE.read_text())
    require(len(source_rows) == 1921 and len(provenance) == 11, "source cardinality")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    before = {"production_demo": sha(SOURCE), "provenance": sha(PROVENANCE)}
    write_exclusive(OUTPUT / "protected_before.json", (json.dumps(before, indent=2, sort_keys=True) + "\n").encode())

    local, episode_records = [], []
    for entry in provenance:
        trace = ROOT / entry["episode_path"] / "transitions.jsonl"
        require(sha(trace) == entry["transitions_sha256"], "source trace hash")
        rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        require(len(rows) == entry["transition_count"], "trace count")
        require(rows[-1]["success"] and rows[-1]["terminated"] and not rows[-1]["truncated"], "canonical source success")
        require(all(r["observation_contract_version"] == V2 for r in rows), "V2 only")
        entry_index = next((i for i,r in enumerate(rows) if
            r["after"]["metrics"]["lateral_error"] <= .010 and
            r["after"]["metrics"]["axial_error"] <= .020), None)
        require(entry_index is not None, "near-hole entry")
        start = max(0, entry_index - 1)
        dataset_rows = source_rows[entry["dataset_start"]:entry["dataset_stop_exclusive"]]
        require(len(dataset_rows) == len(rows), "dataset/trace mapping")
        episode_id = f"episode_{entry['episode_index']:02d}:{entry['episode_path']}"
        selected = []
        for i in range(start, len(rows)):
            src, trace_row = dataset_rows[i], rows[i]
            obs = np.asarray(src["observations"]); act = np.asarray(src["actions"]); nxt = np.asarray(src["next_observations"])
            require(obs.shape == nxt.shape == (20,) and obs.dtype == nxt.dtype == np.float32, "Obs20 float32")
            require(act.shape == (7,) and act.dtype == np.float32, "Action7 float32")
            require(np.array_equal(obs, np.asarray(trace_row["obs"], np.float32)), "observation source equality")
            require(np.array_equal(act, np.asarray(trace_row["action"], np.float32)), "action source equality")
            require(np.array_equal(nxt, np.asarray(trace_row["next_obs"], np.float32)), "next observation source equality")
            require(all(np.isfinite(np.asarray(src[k])).all() for k in src), "finite transition")
            metrics = trace_row["after"]["metrics"]
            item = {k: np.asarray(src[k]).copy() for k in
                    ("observations","actions","next_observations","rewards","masks","dones")}
            item.update(source_episode=episode_id, source_dataset_sha=DEMO_SHA,
                        source_step=int(trace_row["step"]), phase=trace_row["phase"],
                        near_hole_entry_step=int(rows[entry_index]["step"]),
                        success=bool(trace_row["success"]),
                        lateral=float(metrics["lateral_error"]), axial=float(metrics["axial_error"]),
                        orientation=float(metrics["orientation_error"]),
                        transition_identity=transition_identity(src))
            local.append(item); selected.append(item["transition_identity"])
        episode_records.append({"episode":episode_id,"type":entry["type"],"source_hash":entry["transitions_sha256"],
            "near_hole_entry_step":int(rows[entry_index]["step"]),"selected_start_step":int(rows[start]["step"]),
            "transitions":len(selected),"transition_identities":selected})

    require(len(local) == 330 and all(x["transitions"] == 30 for x in episode_records), "local subset cardinality")
    actions = np.asarray([x["actions"] for x in local], np.float32)
    stats = {"mean":actions.mean(0).tolist(),"std":actions.std(0).tolist(),
             "saturation_fraction":(np.abs(actions)>=.999).mean(0).tolist(),
             "a6_excluded_from_bc_objective":True}

    # Deterministic episode-level ordering.  Path breaks ties where scripted
    # accepted episodes have identical transition streams.
    ordered = sorted(episode_records, key=lambda x: hashlib.sha256((x["source_hash"]+"|"+x["episode"]).encode()).hexdigest())
    train_count = min(len(ordered)-1, max(1, round(.8*len(ordered))))
    train_eps = [x["episode"] for x in ordered[:train_count]]; valid_eps = [x["episode"] for x in ordered[train_count:]]
    require(set(train_eps).isdisjoint(valid_eps) and train_eps and valid_eps, "episode-group split")
    split = {"method":"sha256(source_trace_sha|source_episode)","train_episodes":train_eps,"valid_episodes":valid_eps}
    split_bytes = (json.dumps(split, indent=2, sort_keys=True)+"\n").encode(); split_sha=hashlib.sha256(split_bytes).hexdigest()
    payload = pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL)
    write_exclusive(OUTPUT / "local_insertion_supervision.pkl", payload)
    write_exclusive(OUTPUT / "split.json", split_bytes)
    manifest = {"LOCAL_SUPERVISION_DATASET":str(OUTPUT/"local_insertion_supervision.pkl"),
        "LOCAL_SUPERVISION_EPISODES":len(episode_records),"LOCAL_SUPERVISION_TRANSITIONS":len(local),
        "LOCAL_SUPERVISION_SHA256":hashlib.sha256(payload).hexdigest(),"LOCAL_EXPERT_ACTION_STATS":stats,
        "LOCAL_BC_TRAIN_EPISODES":len(train_eps),"LOCAL_BC_VALID_EPISODES":len(valid_eps),
        "LOCAL_BC_SPLIT_SHA256":split_sha,"episodes":episode_records,
        "optional_exact_suffix_audit":{"C0_C1_C2":"INSPECTED","added_transitions":0,
            "reason":"near-hole-to-success suffix transitions already occur in accepted production episodes; no duplicate weighting"},
        "production_demo_sha256":DEMO_SHA,"observation_contract":V2}
    write_exclusive(OUTPUT / "manifest.json", (json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode())
    after={"production_demo":sha(SOURCE),"provenance":sha(PROVENANCE)}
    require(before==after,"protected source mutation")
    write_exclusive(OUTPUT / "protected_after.json", (json.dumps(after,indent=2,sort_keys=True)+"\n").encode())
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__ == "__main__": main()
