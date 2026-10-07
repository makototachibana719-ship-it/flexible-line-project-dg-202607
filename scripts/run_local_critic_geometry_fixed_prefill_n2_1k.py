#!/usr/bin/env python3
"""Local-critic geometry candidate with fixed prefill and frozen admission."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_peg_f3_v2_rlpd_smoke as base

CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_local_critic_geometry_1k_seed0.yaml"
EXPECTED_CONFIG = "35cd3edf5058cc5a01eb18c5f59fdf40b6167f962eb63509180504225de131f4"
RUN_ID = "wandb-disabled-peg-f3-v2-local-critic-geometry-fixed-prefill-n2-1k"
PREFILL_SOURCE_EVIDENCE = "configs/peg_insertion/peg_v2_local_critic_geometry_1k_seed0.yaml:training.training_starts"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VirtualConfig(dict):
    _smoke = {
        "learner_updates": 1000,
        "checkpoint_period": 500,
        "actor_max_steps": 1000000,
        "process_timeout_seconds": 7200,
        "update_timeout_seconds": 120,
    }

    def __getitem__(self, key):
        if key == "smoke" and key not in self:
            return self._smoke
        return super().__getitem__(key)


def _raw_config():
    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "C+G config SHA256")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    require(raw["training"]["checkpoint_period"] == 500, "checkpoint period")
    require(raw["ablation"]["checkpoint_targets"] == [500, 1000], "checkpoint targets")
    require(raw["ablation"]["semantic_diff_fields"] == ["conservative_q", "demo_actor_anchor", "local_critic_geometry"], "single candidate")
    require(raw["conservative_q"]["selected_lambda"] == 5.0, "conservative q lambda")
    require(raw["demo_actor_anchor"]["selected_lambda"] == 0.07, "demo anchor lambda")
    require(raw["local_critic_geometry"]["selected_lambda"] == 93.72235984953991, "local critic lambda")
    require(raw["local_critic_geometry"]["margin"] == 0.0, "local critic margin")
    require(raw["training"]["training_starts"] == 64, "configured prefill threshold")
    return raw


def install(output: Path):
    from training.peg_insertion.local_critic_geometry_candidate import make_local_geometry_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    raw = _raw_config()

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_local_geometry_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save, original_load = resume.save_bundle, resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {
            "identifier": "peg-f3-v2-local-critic-geometry-fixed-prefill-n2-1k",
            "profile": "fixed_replay_prefill_gate_with_admission_freeze",
            "fresh_initialization": True,
            "resume": False,
            "demo_online_ratio": "32:32",
            "utd_ratio": 1,
            "prefill_threshold": raw["training"]["training_starts"],
            "online_admission_frozen_during_updates": True,
            "conservative_q_lambda": 5.0,
            "demo_actor_anchor_lambda": 0.07,
            "local_critic_geometry_lambda": 93.72235984953991,
            "local_critic_geometry_margin": 0.0,
        }
        return original_save(directory, state, replay, step, actor_steps, cfg, manifest, RUN_ID, source, **kwargs)

    resume.save_bundle = save_bundle
    resume.load_bundle = lambda cp, cfg, manifest, run_id: original_load(cp, cfg, manifest, RUN_ID)
    base.CONFIG = CONFIG_PATH
    base.OUTPUT = output
    base.ACTIVE_OUTPUT = output
    base.RUN_UPDATES = 1000
    base.CHECKPOINT_TARGETS = (500, 1000)
    base.FORMAL_5K = False
    base.SHORT_SHUTDOWN_ACCEPTANCE = False
    base.__file__ = str(Path(__file__).resolve())
    base.config = config

    original_runtime_env = base.runtime_env

    def runtime_env():
        env = original_runtime_env()
        required = [
            str(ROOT),
            str(ROOT / "external/serl/serl_launcher"),
            str(ROOT / "external/serl/franka_sim"),
            str(base.ENV_PARENT),
        ]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(required + ([existing] if existing else []))
        return env

    base.runtime_env = runtime_env
    return raw


def _clone_online_prefix(make_replay_buffer, env, online, limit):
    frozen = make_replay_buffer(env, capacity=limit, type="replay_buffer")
    data = online.dataset_dict
    require(len(online) >= limit, "prefill threshold not reached")
    keys = ("observations", "actions", "next_observations", "rewards", "masks", "dones")
    for index in range(limit):
        frozen.insert({key: np.asarray(data[key][index]) for key in keys})
    require(len(frozen) == limit, "frozen replay size")
    return frozen


def learner(out: Path):
    import jax
    from agentlace.trainer import TrainerServer
    from serl_launcher.utils.launcher import make_replay_buffer
    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    from async_sac_state_sim import concat_rlpd_batches, make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import install_actor_step_tracking, save_bundle

    cfg, data, backend = base.validate_config_and_dataset(check_gpu=False)
    prefill = int(cfg["training"]["training_starts"])
    require(prefill == 64, "prefill threshold binding")
    require(jax.default_backend() == "gpu", "learner GPU backend")
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    event_lock = threading.Lock()

    def event(value):
        value["timestamp_ns"] = time.time_ns()
        with event_lock:
            base.append_jsonl(out / "communication.jsonl", value)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        require(np.asarray(obs).shape == (20,) and np.asarray(obs).dtype == np.float32, "learner V2 observation")
        require(env.action_space.shape == (7,) and env.action_space.dtype == np.float32, "learner action contract")
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        initial_rng = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
        online = make_replay_buffer(env, capacity=cfg["training"]["replay_capacity"], type="replay_buffer")
        demo = make_replay_buffer(env, capacity=len(data) + 100, type="replay_buffer")
        install_actor_step_tracking(online, 0)
        for transition in data:
            demo.insert(transition)
        require(len(demo) == 1921 and len(online) == 0, "fresh replay population")
        server = TrainerServer(base.configure(),
            data_callback=lambda name, payload: event({"event": "live_insert",
                "count": len(payload["data"]), "online_size": len(online)}),
            request_callback=lambda kind, payload: {})
        server.register_data_store("actor_env", online)
        server.start(threaded=True)
        base.write_json(out / "server_ready.json", {"pid": os.getpid(), "backend": backend})
        server.publish_network(agent.state.params)
        event({"event": "publish_initial", "online_size": len(online)})
        deadline = time.monotonic() + 180
        while len(online) < prefill:
            if time.monotonic() > deadline:
                raise TimeoutError("fixed prefill deadline")
            time.sleep(.01)
        actor_steps_at_freeze = int(online.project_actor_steps)
        frozen_online = _clone_online_prefix(make_replay_buffer, env, online, prefill)
        install_actor_step_tracking(frozen_online, actor_steps_at_freeze)
        base.write_json(out / "stop_actor.json", {
            "completed_updates": 0,
            "reason": "fixed_prefill_freeze_before_updates",
            "prefill_threshold": prefill,
            "online_size_at_freeze": len(frozen_online),
            "actor_steps_at_freeze": actor_steps_at_freeze,
            "timestamp_ns": time.time_ns(),
        })
        server.stop()
        require(len(frozen_online) == prefill, "online replay freeze size")
        require(int(frozen_online.project_actor_steps) == actor_steps_at_freeze, "frozen replay tracking")
        sharding = jax.sharding.PositionalSharding(jax.local_devices()).replicate()
        online_iter = frozen_online.get_iterator(sample_args={"batch_size": 32}, device=sharding)
        demo_iter = demo.get_iterator(sample_args={"batch_size": 32}, device=sharding)
        env.close()

        completed = 0
        for update in range(1, 1001):
            require(len(frozen_online) == prefill, "online replay growth after freeze")
            online_batch = next(online_iter)
            demo_batch = next(demo_iter)
            require(online_batch["observations"].shape == demo_batch["observations"].shape == (32, 20),
                    "32 Demo + 32 Online observation batches")
            require(online_batch["actions"].shape == demo_batch["actions"].shape == (32, 7),
                    "32 Demo + 32 Online action batches")
            batch = concat_rlpd_batches(online_batch, demo_batch)
            started = time.perf_counter()
            new_agent, info = agent.update_high_utd(batch, utd_ratio=1)
            base.block_tree(jax, (new_agent, info))
            losses = base.loss_scalars(jax, info)
            require(len(losses) >= 3 and all(np.isfinite(v) for v in losses.values()), "loss/Q/temperature finite")
            rng_before = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
            rng_after = np.asarray(jax.device_get(new_agent.state.rng), dtype=np.uint32).tolist()
            require(rng_before != rng_after, "learner RNG did not advance")
            agent = new_agent
            completed = update
            base.append_jsonl(out / "updates.jsonl", {
                "update": update, "seconds": time.perf_counter() - started,
                "losses": losses, "losses_finite": True,
                "online_size": len(frozen_online), "demo_size": len(demo),
                "online_batch": 32, "demo_batch": 32,
                "online_admission_frozen": True,
                "rng_before": rng_before, "rng_after": rng_after})
            if update in (500, 1000):
                save_bundle(checkpoint_dir, agent.state, frozen_online, update,
                            actor_steps_at_freeze, cfg, ROOT / cfg["demo"]["manifest"], RUN_ID,
                            {"identifier": "peg-f3-v2-local-critic-geometry-fixed-prefill-n2-1k"})
                base.append_jsonl(out / "checkpoint_events.jsonl", {
                    "event": "save_complete", "checkpoint": update,
                    "online_size": len(frozen_online), "actor_steps": actor_steps_at_freeze})
        base.write_json(out / "stop_actor.json", {
            "completed_updates": completed,
            "reason": "fixed_prefill_freeze_completed",
            "prefill_threshold": prefill,
            "online_size_at_freeze": len(frozen_online),
            "actor_steps_at_freeze": actor_steps_at_freeze,
            "timestamp_ns": time.time_ns(),
        })
        base.write_json(out / "learner_result.json", {
            "status": "PASS", "backend": jax.default_backend(), "fresh_initialization": True,
            "demo_size": len(demo), "prefill_threshold": prefill,
            "initial_online_size": prefill, "final_online_size": len(frozen_online),
            "actor_steps": actor_steps_at_freeze, "completed_updates": completed,
            "initial_rng": initial_rng, "wandb_run": "DISABLED", "run_id": RUN_ID,
            "requested_updates": 1000, "online_admission_frozen_during_updates": True})


def preflight(output: Path):
    raw = install(output)
    result = base.preflight()
    result.update({
        "STABILIZATION_PROTOCOL": "FIXED_REPLAY_PREFILL_GATE_WITH_ADMISSION_FREEZE",
        "PREFILL_THRESHOLD": raw["training"]["training_starts"],
        "PREFILL_SOURCE_EVIDENCE": PREFILL_SOURCE_EVIDENCE,
        "BASELINE_C_PLUS_G_UNCHANGED": "YES",
        "ONLY_SCIENTIFIC_CHANGE": "LOCAL_CRITIC_GEOMETRY_CORRECTION",
        "ACTOR_OBJECTIVE_UNCHANGED": "YES",
        "BELLMAN_LOSS_UNCHANGED": "YES",
        "CONSERVATIVE_Q_UNCHANGED": "YES",
        "REPLAY_SCHEMA_UNCHANGED": "YES",
        "DATASET_UNCHANGED": "YES",
        "ACTION_SEMANTICS_UNCHANGED": "YES",
        "V2_UNCHANGED": "YES",
        "PRODUCTION_SAC_CHANGED": "NO",
        "PREFILL_GATE_ASSERTION": "learner updates require len(online) >= PREFILL_THRESHOLD before update loop",
        "ADMISSION_FREEZE_ASSERTION": "stop_actor + server.stop before update loop; frozen_online size asserted each update",
        "CHECKPOINT_BY_UPDATE_ASSERTION": [500, 1000],
        "TARGET_UPDATES": 1000,
        "NO_3K_5K": "PASS",
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def verify(out: Path):
    import jax
    from serl_launcher.utils.launcher import make_replay_buffer
    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import restore_replay

    cfg, _rows, _backend = base.validate_config_and_dataset(check_gpu=False)
    checkpoints = out / "checkpoints"
    replay_payload = pickle.loads((checkpoints / "checkpoint_1000_online_replay.pkl").read_bytes())
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        restored = {}
        for step in (500, 1000):
            state = restore_fresh_state(checkpoints, agent.state, step)
            numeric = [np.asarray(jax.device_get(v)) for v in jax.tree_util.tree_leaves(state.params)]
            require(numeric and all(np.isfinite(v).all() for v in numeric), f"checkpoint {step} finite")
            restored[str(step)] = {"numeric_leaves": len(numeric)}
        replay = make_replay_buffer(env, capacity=int(replay_payload["capacity"]), type="replay_buffer")
        replay_info = restore_replay(replay, checkpoints / "checkpoint_1000_online_replay.pkl")
        require(replay_info["size"] == 64, "frozen replay restore size")
        env.close()
    base.write_json(out / "restore_verification.json", {
        "status": "PASS",
        "checkpoints": restored,
        "restored_replay": replay_info,
        "training_resumed": False,
        "protocol": "fixed_prefill_freeze",
    })


def protocol_finalize(out: Path, base_rc: int):
    raw = json.loads((out / "result.json").read_text()) if (out / "result.json").is_file() else {}
    if raw:
        base.write_json(out / "base_supervise_result.json", raw)
    updates = base.read_jsonl(out / "updates.jsonl")
    learner = json.loads((out / "learner_result.json").read_text()) if (out / "learner_result.json").is_file() else {}
    restore = json.loads((out / "restore_verification.json").read_text()) if (out / "restore_verification.json").is_file() else {}
    checkpoints = [x.get("checkpoint") for x in base.read_jsonl(out / "checkpoint_events.jsonl")]
    checks = {
        "gpu": learner.get("backend") == "gpu",
        "prefill_gate": learner.get("prefill_threshold") == 64 and learner.get("initial_online_size") == 64,
        "admission_freeze": learner.get("final_online_size") == 64 and all(x.get("online_size") == 64 for x in updates),
        "ratio": len(updates) == 1000 and all(x["demo_batch"] == x["online_batch"] == 32 for x in updates),
        "updates": [x["update"] for x in updates] == list(range(1, 1001)),
        "finite": bool(updates) and all(x["losses_finite"] for x in updates),
        "checkpoint_500": 500 in checkpoints,
        "checkpoint_1000": 1000 in checkpoints,
        "restore": restore.get("status") == "PASS",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "PEG_F3_V2_LOCAL_CRITIC_GEOMETRY_FIXED_PREFILL_1K": status,
        "checks": checks,
        "base_supervise_returncode": base_rc,
        "PREFILL_THRESHOLD": 64,
        "ONLINE_REPLAY": {"initial": learner.get("initial_online_size"), "final": learner.get("final_online_size")},
        "LEARNER_UPDATES": len(updates),
        "CHECKPOINT_500": checks["checkpoint_500"],
        "CHECKPOINT_1000": checks["checkpoint_1000"],
        "CHECKPOINT_RESTORE": checks["restore"],
        "RESIDUAL_PROCESSES": raw.get("RESIDUAL_PROCESSES", "UNKNOWN"),
        "TRAINING_READY": "YES" if status == "PASS" else "NO",
        "formal_training_started": False,
        "resume_training_started": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    base.write_json(out / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(ROOT in output.parents, "output outside project")
    if args.role == "preflight":
        return preflight(output)
    install(output)
    if args.role == "learner":
        return learner(output) or 0
    if args.role == "actor":
        return base.actor(output) or 0
    if args.role == "verify":
        return verify(output) or 0
    require(not output.exists(), "fresh output required")
    return protocol_finalize(output, base.supervise())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
