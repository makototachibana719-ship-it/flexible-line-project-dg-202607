#!/usr/bin/env python3
"""Policy-only canonical evaluation for an isolated UTD8 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASELINE_ROOT = Path(__file__).resolve().parent
ROOT = BASELINE_ROOT.parent
sys.path.insert(0, str(ROOT))
os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE="false",
                  XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl")

import jax
import numpy as np
import yaml

from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_env(env):
    seen = set()
    while hasattr(env, "env") and id(env) not in seen:
        seen.add(id(env))
        env = env.env
    return env


def contact_flags(env):
    """Established project contact-health check; expected hole contact is legal."""
    raw = raw_env(env)
    model, data = raw.model, raw.data
    link01 = unexpected = False
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
            tokens.extend((geom_name, body_name, mesh_name))
            bodies.append(body_name)
        link01 |= frozenset(bodies) == frozenset(("base_link", "shoulder_Link"))
        expected = any("hole" in str(token).lower() for token in tokens if token is not None)
        unexpected |= not expected
        pairs.append(bodies)
    return bool(link01), bool(unexpected), pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True,
                        help="0 evaluates the deterministic fresh initialization")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--label", default="primary")
    args = parser.parse_args()
    run = args.run.resolve()
    require(BASELINE_ROOT in run.parents, "run must be inside isolated baseline")
    require(args.label.replace("_", "").isalnum(), "safe evaluation label")
    output = run / f"eval_{args.step}_{args.label}.json"
    require(not output.exists(), "evaluation output already exists")

    cfg = yaml.safe_load((BASELINE_ROOT / "config.yaml").read_text())
    dataset = ROOT / cfg["demo"]["path"]
    require(sha(dataset) == cfg["demo"]["sha256"], "Demo hash mismatch")
    checkpoint_dir = run / "checkpoints"
    checkpoint = checkpoint_dir / f"checkpoint_{args.step}"
    if args.step != 0:
        require(checkpoint.is_file(), "checkpoint missing")
    require(jax.default_backend() == "gpu", "GPU backend required")

    example = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
    env_parent = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
    sys.path[:0] = [str(example), str(env_parent)]
    from serl_launcher.utils.launcher import make_sac_agent
    from peg_env_adapter import make_peg_state_env

    rows = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_sac_agent(seed=cfg["seed"], sample_obs=np.zeros(20, np.float32),
                               sample_action=np.zeros(7, np.float32))
        agent = (fresh if args.step == 0 else
                 fresh.replace(state=restore_fresh_state(checkpoint_dir, fresh.state, args.step)))
        agent = jax.device_put(jax.tree_util.tree_map(jax.numpy.asarray, agent))
        for seed in range(args.seed_offset, args.seed_offset + args.episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            best = {"position": float("inf"), "lateral": float("inf"),
                    "axial": float("inf"), "orientation": float("inf")}
            max_force = 0.0
            near_hole = False
            near_hole_steps = 0
            expected_contact = False
            link01_contact = False
            unexpected_contact = False
            saturated_dimensions = 0
            action_dimensions = 0
            final_info = {}
            terminated = truncated = False
            for episode_step in range(500):
                action = agent.sample_actions(observations=np.asarray(obs, np.float32), argmax=True)
                action = np.asarray(jax.device_get(action), np.float32)
                require(action.shape == (7,) and np.isfinite(action).all(), "illegal policy action")
                saturated_dimensions += int(np.sum(np.abs(action[:6]) >= 0.999))
                action_dimensions += 6
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                final_info = dict(info)
                lateral = float(info.get("lateral_error", np.nan))
                axial = float(info.get("axial_error", np.nan))
                orientation = float(info.get("orientation_error", np.nan))
                position = float(np.hypot(lateral, axial))
                if np.isfinite(position): best["position"] = min(best["position"], position)
                if np.isfinite(lateral): best["lateral"] = min(best["lateral"], lateral)
                if np.isfinite(axial): best["axial"] = min(best["axial"], axial)
                if np.isfinite(orientation): best["orientation"] = min(best["orientation"], orientation)
                near = bool(lateral <= 0.010 and axial <= 0.020)
                near_hole |= near
                near_hole_steps += int(near)
                force = float(info.get("force_norm", 0.0))
                if np.isfinite(force): max_force = max(max_force, force)
                link01, unexpected, pairs = contact_flags(env)
                link01_contact |= link01
                unexpected_contact |= unexpected
                expected_contact |= bool(pairs) and not unexpected
                if terminated or truncated:
                    break
            rows.append({"seed": seed, "success": bool(final_info.get("is_success", terminated and not truncated)),
                         "steps": episode_step + 1, "return": total, "max_force": max_force,
                         "min_position_error": best["position"], "min_lateral_error": best["lateral"],
                         "min_axial_error": best["axial"], "min_orientation_error": best["orientation"],
                         "near_hole": near_hole, "near_hole_steps": near_hole_steps,
                         "expected_contact": expected_contact,
                         "link01_contact": link01_contact,
                         "unexpected_contact": unexpected_contact,
                         "contact_health": bool(not link01_contact and not unexpected_contact and
                                                np.isfinite(max_force)),
                         "action_saturation": (saturated_dimensions / action_dimensions
                                               if action_dimensions else 0.0),
                         "terminated": bool(terminated), "truncated": bool(truncated)})
        env.close()

    aggregate = {
        "episodes": len(rows),
        "success_count": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "near_hole_count": int(sum(row["near_hole"] for row in rows)),
        "near_hole_rate": float(np.mean([row["near_hole"] for row in rows])),
        "expected_contact_count": int(sum(row["expected_contact"] for row in rows)),
        "contact_health": bool(all(row["contact_health"] for row in rows)),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "median_min_position_error": float(np.median([row["min_position_error"] for row in rows])),
        "median_min_lateral_error": float(np.median([row["min_lateral_error"] for row in rows])),
        "median_min_axial_error": float(np.median([row["min_axial_error"] for row in rows])),
        "median_min_orientation_error": float(np.median([row["min_orientation_error"] for row in rows])),
        "max_force": float(max(row["max_force"] for row in rows)),
        "mean_action_saturation": float(np.mean([row["action_saturation"] for row in rows])),
    }
    result = {"status": "PASS", "policy_only": True, "intervention": False,
              "hierarchy": False, "checkpoint_step": args.step,
              "canonical_reset": True, "label": args.label,
              "seed_offset": args.seed_offset,
              "checkpoint_sha256": "FRESH_INITIALIZATION" if args.step == 0 else sha(checkpoint),
              "episodes_detail": rows,
              "aggregate": aggregate}
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
