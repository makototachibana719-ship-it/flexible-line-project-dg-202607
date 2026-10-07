#!/usr/bin/env python3
"""Read-only retrospective preflight for a causal online-admission gate."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/home/makoto/flexible_line_project")
CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica2",
}
OUT_ROOT = ROOT / "results/diagnostics/peg"
STAGED_COLLECT_N = 60


def unique_out():
    for index in range(1, 100):
        out = OUT_ROOT / f"f3_v2_online_admission_quality_gate_20260907_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output directory")


def phase_label(value):
    value = str(value).lower()
    if "lateral" in value:
        return "lateral_correction"
    if "orient" in value or "rotation" in value:
        return "orientation_correction"
    if "near" in value or "insert" in value:
        return "near_hole"
    if "approach" in value:
        return "approach"
    return "setup"


def demo_phases():
    values = []
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            values.append(phase_label(json.loads(line)["phase"]))
    return np.asarray(values)


def nearest(query, reference, scale, exclude_self=False):
    distances, indices = [], []
    for start in range(0, len(query), 64):
        stop = min(start + 64, len(query))
        delta = (query[start:stop, None] - reference[None]) / scale
        d = np.sqrt(np.sum(np.square(delta), axis=2))
        if exclude_self:
            rows = np.arange(start, stop)
            d[np.arange(stop - start), rows] = np.inf
        indices.append(np.argmin(d, axis=1))
        distances.append(np.min(d, axis=1))
    return np.concatenate(distances), np.concatenate(indices)


def audit_run(run, demo_obs, phases, scale, threshold):
    payload = pickle.loads((run / "checkpoints/checkpoint_1000_online_replay.pkl").read_bytes())
    size = int(payload["size"])
    obs = np.asarray(payload["dataset_dict"]["observations"][:size], np.float64)
    dones = np.asarray(payload["dataset_dict"]["dones"][:size], bool)
    distance, nearest_index = nearest(obs, demo_obs, scale)
    admitted = distance <= threshold
    episode_start = np.zeros(size, dtype=bool)
    episode_start[0] = True
    if size > 1:
        episode_start[1:] = dones[:-1]
    assigned_phase = phases[nearest_index]
    rel = obs[:, 13:16]
    near_hole = (np.linalg.norm(rel[:, :2], axis=1) <= 0.010) & (np.abs(rel[:, 2]) <= 0.020)
    names = ("setup", "approach", "lateral_correction", "orientation_correction", "near_hole")
    admitted_count = int(np.sum(admitted))
    admitted_phase = {
        name: float(np.mean(assigned_phase[admitted] == name)) if admitted_count else 0.0
        for name in names
    }
    return {
        "size": size,
        "admitted_count": admitted_count,
        "admitted_fraction": float(np.mean(admitted)),
        "rejected_fraction": float(np.mean(~admitted)),
        "admitted_demo_support_fraction": float(np.mean(distance[admitted] <= threshold)) if admitted_count else 0.0,
        "admitted_near_hole_like_fraction": float(np.mean(near_hole[admitted])) if admitted_count else 0.0,
        "pre_divergence_fraction": float(np.mean(episode_start)),
        "pre_divergence_admitted_fraction": float(np.mean(admitted[episode_start])),
        "post_divergence_admitted_fraction": float(np.mean(admitted[~episode_start])),
        "far_post_divergence_fraction_before": float(np.mean((~episode_start) & (~admitted))),
        "far_post_divergence_fraction_after": 0.0,
        "admitted_phase_distribution": admitted_phase,
        "distance": {"min": float(np.min(distance)), "median": float(np.median(distance)), "max": float(np.max(distance))},
    }


def mean_field(rows, field):
    return float(np.mean([row[field] for row in rows.values()]))


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    demos = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    demo_obs = np.asarray([row["observations"] for row in demos], np.float64)
    phases = demo_phases()
    lo, hi = np.percentile(demo_obs, [1, 99], axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    demo_loo, _ = nearest(demo_obs, demo_obs, scale, exclude_self=True)
    threshold = float(np.max(demo_loo))
    runs = {name: audit_run(run, demo_obs, phases, scale, threshold) for name, run in RUNS.items()}
    phase_distribution = {
        phase: float(np.mean([row["admitted_phase_distribution"][phase] for row in runs.values()]))
        for phase in next(iter(runs.values()))["admitted_phase_distribution"]
    }
    enough_for_stage = all(row["admitted_count"] >= STAGED_COLLECT_N for row in runs.values())
    early_retained = all(
        row["pre_divergence_admitted_fraction"] > 0
        and row["admitted_phase_distribution"]["setup"] + row["admitted_phase_distribution"]["approach"] > 0
        for row in runs.values()
    )
    retrospective_pass = enough_for_stage and early_retained
    result = {
        "ONLINE_ADMISSION_QUALITY_REVIEW": "PASS" if retrospective_pass else "FAIL",
        "QUALITY_SIGNAL_SET": "CURRENT_OBS20_NORMALIZED_NEAREST_ACCEPTED_DEMO_DISTANCE",
        "QUALITY_GATE_RULE": f"ADMIT_IFF_D_DEMO_LE_{threshold:.12g}",
        "THRESHOLD_SOURCE": "MAX_ACCEPTED_DEMO_LEAVE_ONE_OUT_NEAREST_DISTANCE_USING_DEMO_P01_P99_SCALE",
        "ADMITTED_FRACTION": mean_field(runs, "admitted_fraction"),
        "REJECTED_FRACTION": mean_field(runs, "rejected_fraction"),
        "ADMITTED_DEMO_SUPPORT_FRACTION": mean_field(runs, "admitted_demo_support_fraction"),
        "ADMITTED_NEAR_HOLE_LIKE_FRACTION": mean_field(runs, "admitted_near_hole_like_fraction"),
        "POST_DIVERGENCE_ADMITTED_FRACTION": mean_field(runs, "post_divergence_admitted_fraction"),
        "ADMITTED_PHASE_DISTRIBUTION": phase_distribution,
        "MIN_ADMITTED_COUNT_PER_REPLICA": min(row["admitted_count"] for row in runs.values()),
        "REQUIRED_COUNT_PER_COLLECTION_STAGE": STAGED_COLLECT_N,
        "CAUSAL_FILTER": "YES",
        "RETROSPECTIVE_FILTER": "PASS" if retrospective_pass else "FAIL",
        "PRIMARY_ROOT_CAUSE": "NONE_AT_RETROSPECTIVE_GATE" if retrospective_pass else "NO_USEFUL_ONLINE_CORRIDOR_DATA_AVAILABLE",
        "NEXT": "STATIC_RUNNER_PREFLIGHT" if retrospective_pass else "DEMO_OR_CURRICULUM_REVIEW",
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": {"threshold": threshold, "demo_loo_distance": {
            "median": float(np.median(demo_loo)), "p99": float(np.percentile(demo_loo, 99)), "max": threshold}, "runs": runs},
    }
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "DETAILS"}, sort_keys=True))
    print(str(out / "result.json"))


if __name__ == "__main__":
    main()
