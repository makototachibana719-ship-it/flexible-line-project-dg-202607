#!/usr/bin/env python3
"""Read-only replay/timing variance audit for C+G N=2 replicas."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/home/makoto/flexible_line_project")
RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica1",
    "R2": ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica2_retry001",
}
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out() -> Path:
    stem = "f3_v2_async_variance_replay_audit_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [finite(x) for x in v]
    return v


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_replay(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("dataset_dict"), dict):
            return {k: np.asarray(v) for k, v in obj["dataset_dict"].items() if np.asarray(v).ndim >= 1}
        return {k: np.asarray(v) for k, v in obj.items() if np.asarray(v).ndim >= 1}
    attrs = {}
    for key in ("data", "dataset", "_data", "_dict"):
        if hasattr(obj, key):
            val = getattr(obj, key)
            if isinstance(val, dict):
                attrs.update({k: np.asarray(v) for k, v in val.items() if np.asarray(v).ndim >= 1})
    return attrs


def replay_summary(path: Path):
    obj = pickle.loads(path.read_bytes())
    data = flatten_replay(obj)
    if not data:
        return {"path": str(path), "sha256": sha(path), "fields": [], "size": None}
    n = int(obj.get("size", 0)) if isinstance(obj, dict) and obj.get("size") else max(int(v.shape[0]) for v in data.values())
    obs = data.get("observations")
    acts = data.get("actions")
    summary = {"path": str(path), "sha256": sha(path), "fields": sorted(data), "size": n}
    if obs is not None and obs.ndim >= 2:
        obs = obs[:n].astype(np.float64)
        summary["obs_mean"] = np.mean(obs, axis=0).tolist()
        summary["obs_std_mean"] = float(np.mean(np.std(obs, axis=0)))
        summary["early_like_fraction"] = float(np.mean(np.linalg.norm(obs[:, :7] - obs[0, :7], axis=1) < 0.25))
    if acts is not None and acts.ndim >= 2:
        acts = acts[:n].astype(np.float64)
        summary["action_mean"] = np.mean(acts, axis=0).tolist()
        summary["action_std"] = np.std(acts, axis=0).tolist()
        summary["action_saturated_fraction"] = float(np.mean(np.abs(acts) >= 0.95))
    return summary


def run_summary(run: Path):
    checkpoints = read_jsonl(run / "checkpoint_events.jsonl")
    updates = read_jsonl(run / "updates.jsonl")
    communication = read_jsonl(run / "communication.jsonl")
    result = read_json(run / "result.json")
    out = {
        "result_status": result.get("PEG_F3_V2_RLPD_SMOKE"),
        "online_replay": result.get("ONLINE_REPLAY"),
        "checkpoint_events": checkpoints,
        "update_count": len(updates),
        "communication_count": len(communication),
        "ckpt500": replay_summary(run / "checkpoints/checkpoint_500_online_replay.pkl"),
        "ckpt1000": replay_summary(run / "checkpoints/checkpoint_1000_online_replay.pkl"),
    }
    if updates:
        out["first_update"] = updates[0]
        out["last_update"] = updates[-1]
    return out


def l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    runs = {name: run_summary(path) for name, path in RUNS.items()}
    r1, r2 = runs["R1"], runs["R2"]
    diffs = {
        "replay_final_delta": abs(r1["online_replay"]["final"] - r2["online_replay"]["final"]),
        "ckpt500_size_delta": abs(r1["ckpt500"]["size"] - r2["ckpt500"]["size"]),
        "ckpt1000_size_delta": abs(r1["ckpt1000"]["size"] - r2["ckpt1000"]["size"]),
        "ckpt1000_obs_mean_l2": l2(r1["ckpt1000"].get("obs_mean", []), r2["ckpt1000"].get("obs_mean", [])),
        "ckpt1000_action_mean_l2": l2(r1["ckpt1000"].get("action_mean", []), r2["ckpt1000"].get("action_mean", [])),
        "update_count_delta": abs(r1["update_count"] - r2["update_count"]),
        "communication_count_delta": abs(r1["communication_count"] - r2["communication_count"]),
    }
    largest = max(diffs.items(), key=lambda item: item[1])
    match = "NO" if diffs["replay_final_delta"] > 0 or diffs["ckpt1000_obs_mean_l2"] > 0 else "YES"
    result = {
        "ASYNC_VARIANCE_STAGE": "A_REPLAY_DIFFERENCE_AUDIT",
        "REPLAY_DISTRIBUTION_MATCH": match,
        "LARGEST_REPLAY_DIFFERENCE": {"field": largest[0], "value": largest[1]},
        "DIFFS": diffs,
        "RUNS": runs,
        "TRAINING_STARTED": "false",
        "FORMAL_3K_STARTED": "false",
        "FORMAL_5K_STARTED": "false",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    (out / "result.json").write_text(json.dumps(finite(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(finite({
        "REPLAY_DISTRIBUTION_MATCH": match,
        "LARGEST_REPLAY_DIFFERENCE": result["LARGEST_REPLAY_DIFFERENCE"],
        "REPLAY_FINAL": {k: runs[k]["online_replay"]["final"] for k in sorted(runs)},
        "CKPT1000_SIZE": {k: runs[k]["ckpt1000"]["size"] for k in sorted(runs)},
        "CKPT1000_OBS_MEAN_L2": diffs["ckpt1000_obs_mean_l2"],
        "OUT": str(out / "result.json"),
    }), sort_keys=True))


if __name__ == "__main__":
    main()
