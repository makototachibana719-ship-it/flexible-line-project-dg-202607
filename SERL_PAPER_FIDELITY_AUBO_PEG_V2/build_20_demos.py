#!/usr/bin/env python3
"""Build a physical 20-episode V2 demo set; preserve the accepted 11 bit-exact."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
OUTPUT = BRANCH / "PAPER_FIDELITY_20_DEMO_V1"
SOURCE = ROOT / "results/datasets/peg_v2_accepted_11_20260906"
SOURCE_DATA = SOURCE / "peg_v2_accepted_11.pkl"
SOURCE_MANIFEST = SOURCE / "manifest.json"
SOURCE_PROVENANCE = SOURCE / "episode_provenance.json"
V2 = "PEG_OBS_CONTRACT_V2_SENSOR_FIXED"
FIELDS = {"observations", "actions", "next_observations", "rewards", "masks", "dones"}
EXAMPLE = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
ENV_PARENT = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
os.environ.setdefault("MUJOCO_GL", "egl")

from peg_env_adapter import make_peg_state_env
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


PROFILES = [
    # name, category, source accepted episode, reset prelude action, repeats, hold index, holds
    ("standard_plus_y_start", "standard", 0, [0, .22, .08, 0, 0, 0, -1], 2, 86, 2),
    ("standard_minus_y_start", "standard", 0, [0, -.22, -.08, 0, 0, 0, -1], 3, 104, 4),
    ("plus_y_early_correction", "lateral_plus", 3, [0, .18, .05, 0, 0, 0, -1], 2, 108, 2),
    ("plus_y_late_correction", "lateral_plus", 3, [0, -.16, .06, 0, 0, 0, -1], 3, 112, 5),
    ("minus_y_early_correction", "lateral_minus", 5, [0, -.18, .05, 0, 0, 0, -1], 2, 108, 3),
    ("minus_y_late_correction", "lateral_minus", 5, [0, .16, -.06, 0, 0, 0, -1], 3, 112, 6),
    ("roll_varied_start", "rotation_roll", 7, [0, .10, 0, .14, 0, 0, -1], 2, 114, 4),
    ("pitch_early_correction", "rotation_pitch", 9, [0, -.10, 0, 0, .14, 0, -1], 2, 116, 3),
    ("pitch_late_correction", "rotation_pitch", 9, [0, .10, 0, 0, -.14, 0, -1], 3, 121, 6),
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def raw_env(env):
    seen = set()
    while hasattr(env, "env") and id(env) not in seen:
        seen.add(id(env))
        env = env.env
    return env


def contact_flags(env):
    raw = raw_env(env)
    model, data = raw.model, raw.data
    link01 = unexpected = expected = False
    pairs = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        tokens, bodies = [], []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            body_id = int(model.geom_bodyid[geom_id])
            data_id = int(model.geom_dataid[geom_id])
            geom_name = model.geom(geom_id).name
            body_name = model.body(body_id).name
            mesh_name = (model.mesh(data_id).name
                         if int(model.geom_type[geom_id]) == 7 and data_id >= 0 else None)
            bodies.append(body_name)
            tokens.extend((geom_name, body_name, mesh_name))
        is_expected = any("hole" in str(token).lower() for token in tokens if token is not None)
        link01 |= frozenset(bodies) == frozenset(("base_link", "shoulder_Link"))
        unexpected |= not is_expected
        expected |= is_expected
        pairs.append(bodies)
    return bool(link01), bool(unexpected), bool(expected), pairs


def source_commands(provenance, episode_index):
    path = ROOT / provenance[episode_index]["episode_path"] / "transitions.jsonl"
    require(sha(path) == provenance[episode_index]["transitions_sha256"],
            f"source trace hash {episode_index}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return [(row["phase"], np.asarray(row["action"], np.float32)) for row in records], path


def episode(env, profile, provenance, episode_number):
    name, category, source_index, prelude_raw, repeats, hold_index, holds = profile
    commands, trace_path = source_commands(provenance, source_index)
    obs, _ = env.reset(seed=1000 + episode_number)
    prelude = np.asarray(prelude_raw, np.float32)
    link01 = unexpected = expected = False
    prelude_pairs = []
    for action in [prelude] * repeats + [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * 8:
        obs, _, terminated, truncated, _ = env.step(action)
        require(not terminated and not truncated, f"{name} reset prelude")
        lnk, bad, exp, pairs = contact_flags(env)
        link01 |= lnk; unexpected |= bad; expected |= exp; prelude_pairs.extend(pairs)
    initial_obs = np.asarray(obs, np.float32).copy()

    inverse = prelude.copy()
    inverse[:6] *= -1
    plan = [("initial_state_correction", inverse.copy()) for _ in range(repeats)]
    plan += [("initial_state_settle", np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
             for _ in range(8)]
    for index, command in enumerate(commands):
        if index == hold_index:
            plan += [("correction_timing_hold", np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
                     for _ in range(holds)]
        plan.append(command)

    rows, evidence = [], []
    success = truncated = False
    max_force = 0.0
    min_position = float("inf")
    for step, (phase, action) in enumerate(plan, 1):
        require(action.shape == (7,) and np.isfinite(action).all() and
                np.max(np.abs(action)) <= 1.0, f"{name} Action7 legal")
        current = np.asarray(obs, np.float32).copy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, np.float32)
        lnk, bad, exp, pairs = contact_flags(env)
        link01 |= lnk; unexpected |= bad; expected |= exp
        force = float(info.get("force_norm", 0.0))
        max_force = max(max_force, force) if np.isfinite(force) else float("inf")
        position = float(info.get("position_error", np.linalg.norm(next_obs[13:16])))
        min_position = min(min_position, position)
        done = bool(terminated or truncated)
        rows.append({
            "observations": current, "actions": action.copy(),
            "next_observations": next_obs.copy(), "rewards": np.float32(reward),
            "masks": np.float32(0 if terminated else 1), "dones": done,
        })
        evidence.append({
            "step": step, "phase": phase, "action": action.tolist(),
            "reward": float(reward), "terminated": bool(terminated),
            "truncated": bool(truncated), "success": bool(info.get("is_success", False)),
            "position_error": position,
            "lateral_error": float(info.get("lateral_error", np.nan)),
            "axial_error": float(info.get("axial_error", np.nan)),
            "orientation_error": float(info.get("orientation_error", np.nan)),
            "force_norm": force, "contact_pairs": pairs,
            "unexpected_contact": bad, "link01_contact": lnk,
        })
        obs = next_obs
        if done:
            success = bool(terminated and not truncated and info.get("is_success", True))
            break

    # Preserve the successful insertion direction if extra settling delayed the
    # base terminal event; this is still ordinary legal Action7 control.
    recovery = np.array([.30, 0, 0, 0, 0, 0, -1], np.float32)
    while not success and not truncated and len(rows) < 450:
        current = np.asarray(obs, np.float32).copy()
        next_obs, reward, terminated, truncated, info = env.step(recovery)
        next_obs = np.asarray(next_obs, np.float32)
        lnk, bad, exp, pairs = contact_flags(env)
        link01 |= lnk; unexpected |= bad; expected |= exp
        force = float(info.get("force_norm", 0.0))
        max_force = max(max_force, force) if np.isfinite(force) else float("inf")
        position = float(info.get("position_error", np.linalg.norm(next_obs[13:16])))
        min_position = min(min_position, position)
        done = bool(terminated or truncated)
        rows.append({"observations": current, "actions": recovery.copy(),
                     "next_observations": next_obs.copy(), "rewards": np.float32(reward),
                     "masks": np.float32(0 if terminated else 1), "dones": done})
        evidence.append({"step": len(rows), "phase": "bounded_insertion_completion",
                         "action": recovery.tolist(), "reward": float(reward),
                         "terminated": bool(terminated), "truncated": bool(truncated),
                         "success": bool(info.get("is_success", False)),
                         "position_error": position,
                         "lateral_error": float(info.get("lateral_error", np.nan)),
                         "axial_error": float(info.get("axial_error", np.nan)),
                         "orientation_error": float(info.get("orientation_error", np.nan)),
                         "force_norm": force, "contact_pairs": pairs,
                         "unexpected_contact": bad, "link01_contact": lnk})
        obs = next_obs
        success = bool(terminated and not truncated and info.get("is_success", True))

    continuity = all(np.array_equal(rows[i]["next_observations"], rows[i + 1]["observations"])
                     for i in range(len(rows) - 1))
    checks = {
        "success": success,
        "truncated_false": not truncated,
        "v2": all(row["observations"].shape == row["next_observations"].shape == (20,)
                  for row in rows),
        "action7_legal": all(row["actions"].shape == (7,) and
                             np.max(np.abs(row["actions"])) <= 1 for row in rows),
        "continuity": continuity,
        "contact_health": bool(not link01 and not unexpected and np.isfinite(max_force)),
        "terminal_contract": bool(rows and rows[-1]["dones"] and rows[-1]["masks"] == 0),
    }
    require(all(checks.values()), f"new demo {name}: {checks}")
    raw_dir = OUTPUT / "new_episodes" / f"{episode_number:02d}_{name}"
    raw_dir.mkdir(parents=True, exist_ok=False)
    with (raw_dir / "transitions.jsonl").open("x") as stream:
        for item in evidence:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "name": name, "category": category, "episode_index": episode_number,
        "source_accepted_episode": source_index, "source_trace": str(trace_path.relative_to(ROOT)),
        "initial_observation": initial_obs.tolist(), "prelude_action": prelude.tolist(),
        "prelude_repeats": repeats, "correction_hold_index": hold_index,
        "correction_hold_steps": holds, "transitions": len(rows),
        "success": True, "truncated": False, "observation_contract_version": V2,
        "max_force": max_force, "min_position_error": min_position,
        "expected_contact": expected, "prelude_contact_pairs": prelude_pairs,
        "checks": checks,
    }
    dump(raw_dir / "episode.json", summary)
    summary["transitions_sha256"] = sha(raw_dir / "transitions.jsonl")
    return rows, summary


def main():
    require(not OUTPUT.exists(), "PAPER_FIDELITY_20_DEMO_V1 already exists")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    require(source_manifest["status"] == "PASS" and source_manifest["episodes"] == 11 and
            source_manifest["dataset_sha256"] == sha(SOURCE_DATA), "accepted 11 binding")
    source_rows = pickle.loads(SOURCE_DATA.read_bytes())
    require(len(source_rows) == 1921 and all(set(row) == FIELDS for row in source_rows),
            "accepted replay schema")
    provenance = json.loads(SOURCE_PROVENANCE.read_text())
    require(len(provenance) == 11, "accepted provenance")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "new_episodes").mkdir()
    all_rows = list(source_rows)
    new_summaries = []
    boundaries = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        try:
            for offset, profile in enumerate(PROFILES):
                start = len(all_rows)
                rows, summary = episode(env, profile, provenance, 11 + offset)
                all_rows.extend(rows)
                boundaries.append({"episode_index": 11 + offset, "start": start,
                                   "stop_exclusive": len(all_rows), "name": profile[0]})
                new_summaries.append(summary)
        finally:
            env.close()

    require(len(new_summaries) == 9 and all(x["success"] and not x["truncated"]
                                            for x in new_summaries), "nine successful V2 demos")
    require(all(np.array_equal(all_rows[i][key], source_rows[i][key])
                if isinstance(source_rows[i][key], np.ndarray) else
                all_rows[i][key] == source_rows[i][key]
                for i in range(len(source_rows)) for key in FIELDS),
            "accepted 11 values preserved bit-exact")
    terminal_count = sum(bool(row["dones"]) for row in all_rows)
    require(terminal_count == 20, f"exactly 20 terminal episodes, got {terminal_count}")
    data_path = OUTPUT / "paper_fidelity_20_demo_v1.pkl"
    data_path.write_bytes(pickle.dumps(all_rows, protocol=pickle.HIGHEST_PROTOCOL))
    counts = dict(source_manifest["episode_counts"])
    for summary in new_summaries:
        counts[summary["category"]] = counts.get(summary["category"], 0) + 1
    manifest = {
        "schema": "PAPER_FIDELITY_20_DEMO_V1",
        "status": "PASS", "v2_only": True,
        "observation_contract_version": V2,
        "dataset_file": data_path.name, "dataset_sha256": sha(data_path),
        "episodes": 20, "transitions": len(all_rows), "terminal_count": terminal_count,
        "episode_counts": counts,
        "accepted_11_source": str(SOURCE.relative_to(ROOT)),
        "accepted_11_dataset_sha256": sha(SOURCE_DATA),
        "accepted_11_values_preserved": True,
        "new_episode_count": 9,
        "new_episode_all_success": True,
        "new_episode_all_truncated_false": True,
        "new_episode_all_v2": True,
        "new_episode_all_contact_health": True,
        "new_episode_all_action7_legal": True,
        "new_episode_all_continuity": True,
        "coverage": ["standard", "+Y", "-Y", "roll", "pitch",
                     "varied legitimate approach states", "varied correction timing"],
        "old_action_values_edited": False,
        "boundaries": boundaries,
    }
    dump(OUTPUT / "manifest.json", manifest)
    dump(OUTPUT / "new_episode_provenance.json", new_summaries)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
