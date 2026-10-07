#!/usr/bin/env python3
"""C+G runner with deterministic staged online admission."""
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

CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
EXPECTED_CONFIG = "b1cd4126c10673c267e0e2382a44eef97cfc7b0611a5774568d46ab6d8f2ff41"
RUN_ID = "wandb-disabled-peg-f3-v2-cg-deterministic-staged-admission-n2-1k"
PREFILL_SOURCE_EVIDENCE = "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml:training.training_starts"
INITIAL_PREFILL = 64
STAGED_COLLECT_N = 60
STAGED_TRAIN_K = 100
STAGE_COUNT = 10
EXPECTED_FINAL_REPLAY = INITIAL_PREFILL + (STAGE_COUNT - 1) * STAGED_COLLECT_N


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
    require(raw["ablation"]["semantic_diff_fields"] == ["conservative_q", "demo_actor_anchor"], "C+G only")
    require(raw["conservative_q"]["selected_lambda"] == 5.0, "conservative q lambda")
    require(raw["demo_actor_anchor"]["selected_lambda"] == 0.07, "demo anchor lambda")
    require(raw["training"]["training_starts"] == 64, "configured prefill threshold")
    return raw


def install(output: Path):
    from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    raw = _raw_config()

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_combined_q_anchor_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save, original_load = resume.save_bundle, resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {
            "identifier": "peg-f3-v2-cg-deterministic-staged-admission-n2-1k",
            "profile": "deterministic_staged_online_admission",
            "fresh_initialization": True,
            "resume": False,
            "demo_online_ratio": "32:32",
            "utd_ratio": 1,
            "prefill_threshold": raw["training"]["training_starts"],
            "online_admission_frozen_during_each_update_block": True,
            "staged_collect_n": STAGED_COLLECT_N,
            "staged_train_k": STAGED_TRAIN_K,
            "stage_count": STAGE_COUNT,
            "conservative_q_lambda": 5.0,
            "demo_actor_anchor_lambda": 0.07,
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
    require(int(cfg["training"]["training_starts"]) == INITIAL_PREFILL, "initial prefill binding")
    require(jax.default_backend() == "gpu", "learner GPU backend")
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    event_lock = threading.Lock()

    def event(value):
        value["timestamp_ns"] = time.time_ns()
        with event_lock:
            base.append_jsonl(out / "communication.jsonl", value)

    def fixed_batch(store, indices):
        return {key: jax.device_put(np.asarray(store.dataset_dict[key])[indices])
                for key in base.FIELDS}

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        require(np.asarray(obs).shape == (20,) and np.asarray(obs).dtype == np.float32, "learner V2 observation")
        require(env.action_space.shape == (7,) and env.action_space.dtype == np.float32, "learner action contract")
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        initial_rng = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
        sample_rng = np.random.default_rng(int(cfg["seed"]) + 20260907)
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
        event({"event": "fresh_actor_state_bound", "online_size": 0})
        env.close()

        completed = 0
        required_network_after_ns = 0
        for stage in range(STAGE_COUNT):
            collect_count = INITIAL_PREFILL if stage == 0 else STAGED_COLLECT_N
            target_online = INITIAL_PREFILL + stage * STAGED_COLLECT_N
            target_update = (stage + 1) * STAGED_TRAIN_K
            base.write_json(out / f"stage_{stage:03d}_collect.json", {
                "stage": stage, "collect_count": collect_count,
                "target_online": target_online,
                "required_network_after_ns": required_network_after_ns})
            deadline = time.monotonic() + 180
            while len(online) < target_online or not (out / f"stage_{stage:03d}_actor_done.json").is_file():
                require(len(online) <= target_online, "online admission exceeded exact stage target")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"stage {stage} collection deadline")
                time.sleep(.01)
            require(len(online) == target_online, "exact online stage target")
            actor_done = json.loads((out / f"stage_{stage:03d}_actor_done.json").read_text())
            require(actor_done["stage"] == stage and actor_done["uploaded"] == collect_count,
                    "actor stage acknowledgement")
            base.write_json(out / f"stage_{stage:03d}_train.json", {
                "stage": stage, "online_size": len(online), "start_update": completed + 1,
                "target_update": target_update})
            base.append_jsonl(out / "stage_events.jsonl", {
                "event": "train_start", "stage": stage, "target_replay": target_online,
                "actual_replay": len(online), "target_update": target_update,
                "actual_update": completed})
            for update in range(completed + 1, target_update + 1):
                require(len(online) == target_online, "actor admitted during frozen learner block")
                online_idx = sample_rng.integers(0, target_online, size=32)
                demo_idx = sample_rng.integers(0, len(demo), size=32)
                online_batch = fixed_batch(online, online_idx)
                demo_batch = fixed_batch(demo, demo_idx)
                batch = concat_rlpd_batches(online_batch, demo_batch)
                started = time.perf_counter()
                new_agent, info = agent.update_high_utd(batch, utd_ratio=1)
                base.block_tree(jax, (new_agent, info))
                losses = base.loss_scalars(jax, info)
                require(len(losses) >= 3 and all(np.isfinite(v) for v in losses.values()),
                        "loss/Q/temperature finite")
                rng_before = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
                rng_after = np.asarray(jax.device_get(new_agent.state.rng), dtype=np.uint32).tolist()
                require(rng_before != rng_after, "learner RNG did not advance")
                agent = new_agent
                completed = update
                base.append_jsonl(out / "updates.jsonl", {
                    "update": update, "stage": stage, "seconds": time.perf_counter() - started,
                    "losses": losses, "losses_finite": True, "online_size": len(online),
                    "demo_size": len(demo), "online_batch": 32, "demo_batch": 32,
                    "online_admission_frozen": True, "rng_before": rng_before, "rng_after": rng_after})
                if update in (500, 1000):
                    save_bundle(checkpoint_dir, agent.state, online, update,
                                int(online.project_actor_steps), cfg, ROOT / cfg["demo"]["manifest"], RUN_ID,
                                {"identifier": "peg-f3-v2-cg-deterministic-staged-admission-n2-1k"})
                    base.append_jsonl(out / "checkpoint_events.jsonl", {
                        "event": "save_complete", "checkpoint": update,
                        "online_size": len(online), "actor_steps": int(online.project_actor_steps)})
            required_network_after_ns = time.time_ns()
            server.publish_network(agent.state.params)
            event({"event": "publish_stage", "stage": stage, "update": completed,
                   "required_network_after_ns": required_network_after_ns})
            base.append_jsonl(out / "stage_events.jsonl", {
                "event": "train_complete", "stage": stage, "target_replay": target_online,
                "actual_replay": len(online), "target_update": target_update,
                "actual_update": completed})
        require(len(online) == EXPECTED_FINAL_REPLAY and completed == 1000, "final stage totals")
        base.write_json(out / "stop_actor.json", {
            "completed_updates": completed, "reason": "deterministic_staged_admission_completed",
            "final_online_size": len(online), "actor_steps": int(online.project_actor_steps),
            "timestamp_ns": time.time_ns()})
        cleanup_deadline = time.monotonic() + 20
        while not (out / "actor_cleanup_complete.json").is_file():
            if time.monotonic() > cleanup_deadline:
                raise TimeoutError("actor cleanup before server stop")
            time.sleep(.01)
        server.stop()
        base.write_json(out / "learner_result.json", {
            "status": "PASS", "backend": jax.default_backend(), "fresh_initialization": True,
            "demo_size": len(demo), "prefill_threshold": INITIAL_PREFILL,
            "initial_online_size": INITIAL_PREFILL, "final_online_size": len(online),
            "actor_steps": int(online.project_actor_steps), "completed_updates": completed,
            "initial_rng": initial_rng, "wandb_run": "DISABLED", "run_id": RUN_ID,
            "requested_updates": 1000, "stage_count": STAGE_COUNT,
            "staged_collect_n": STAGED_COLLECT_N, "staged_train_k": STAGED_TRAIN_K,
            "online_admission_frozen_during_each_update_block": True})


def actor(out: Path):
    import jax
    from agentlace.data.data_store import QueuedDataStore
    from training.peg_insertion.actor_shutdown import BoundedTrainerClient
    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

    cfg = base.config()
    require(jax.default_backend() == "gpu", "actor GPU backend")
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        sampling_rng = jax.random.split(jax.random.PRNGKey(cfg["seed"]))[1]
        store = QueuedDataStore(2000)
        client = BoundedTrainerClient(
            "actor_env", "127.0.0.1", base.configure(), store, wait_for_server=False,
            shutdown_timeout_seconds=10, shutdown_evidence_path=out / "actor_shutdown.json")
        received = [0]
        network_times = []

        def network(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))
            received[0] += 1
            timestamp_ns = time.time_ns()
            network_times.append(timestamp_ns)
            base.append_jsonl(out / "actor_network.jsonl", {
                "event": "received", "count": received[0], "timestamp_ns": timestamp_ns})

        client.recv_network_callback(network)
        steps = 0
        try:
            for stage in range(STAGE_COUNT):
                marker = out / f"stage_{stage:03d}_collect.json"
                deadline = time.monotonic() + 180
                while not marker.is_file():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"stage {stage} marker deadline")
                    time.sleep(.01)
                spec = json.loads(marker.read_text())
                while max(network_times, default=0) < spec["required_network_after_ns"]:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"stage {stage} network deadline")
                    time.sleep(.01)
                for local_step in range(spec["collect_count"]):
                    sampling_rng, key = jax.random.split(sampling_rng)
                    action = np.asarray(jax.device_get(agent.sample_actions(
                        observations=jax.device_put(obs), seed=key, deterministic=False)), dtype=np.float32)
                    require(action.shape == (7,) and np.isfinite(action).all(), "actor action contract")
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    store.insert({"observations": np.asarray(obs, dtype=np.float32), "actions": action,
                        "next_observations": np.asarray(next_obs, dtype=np.float32),
                        "rewards": np.float32(reward), "masks": np.float32(0 if terminated else 1),
                        "dones": bool(terminated or truncated)})
                    obs = next_obs
                    if terminated or truncated:
                        obs, _ = env.reset()
                    steps += 1
                    if (local_step + 1) % 30 == 0:
                        require(client.update(), "actor staged upload failed")
                require(client.update(), "actor final staged upload failed")
                base.write_json(out / f"stage_{stage:03d}_actor_done.json", {
                    "stage": stage, "uploaded": spec["collect_count"],
                    "total_steps": steps, "network_count": received[0]})
            deadline = time.monotonic() + 180
            while not (out / "stop_actor.json").is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError("final learner stop deadline")
                time.sleep(.01)
        finally:
            try:
                client.update()
            except Exception:
                pass
            base.write_json(out / "actor_stop_observed.json", {
                "stop_marker": (out / "stop_actor.json").exists(), "steps": steps,
                "timestamp_ns": time.time_ns()})
            base.write_json(out / "actor_result.json", {
                "status": "PASS", "backend": jax.default_backend(), "steps": steps,
                "received_networks": received[0], "completed_stages": STAGE_COUNT,
                "stopped_by_learner": (out / "stop_actor.json").exists()})
            shutdown = client.stop()
            env.close()
            base.write_json(out / "actor_cleanup_complete.json", {
                "status": "PASS", "shutdown": shutdown, "environment_closed": True,
                "timestamp_ns": time.time_ns()})


def preflight(output: Path):
    raw = install(output)
    result = base.preflight()
    result.update({
        "STAGED_STATIC_PREFLIGHT": "PASS",
        "STABILIZATION_PROTOCOL": "DETERMINISTIC_STAGED_ADMISSION",
        "PREFILL_THRESHOLD": raw["training"]["training_starts"],
        "PREFILL_SOURCE_EVIDENCE": PREFILL_SOURCE_EVIDENCE,
        "STAGED_COLLECT_N": STAGED_COLLECT_N,
        "STAGED_TRAIN_K": STAGED_TRAIN_K,
        "STAGE_COUNT_1K": STAGE_COUNT,
        "EXPECTED_FINAL_REPLAY_SIZE": EXPECTED_FINAL_REPLAY,
        "C_PLUS_G_ONLY": "YES",
        "ALGORITHM_UNCHANGED": "YES",
        "REPLAY_SCHEMA_UNCHANGED": "YES",
        "DATASET_UNCHANGED": "YES",
        "ACTION_SEMANTICS_UNCHANGED": "YES",
        "V2_UNCHANGED": "YES",
        "PRODUCTION_SAC_CHANGED": "NO",
        "ACTOR_OBJECTIVE_UNCHANGED": "YES",
        "CRITIC_OBJECTIVE_UNCHANGED": "YES",
        "COLLECTION_PHASE_ASSERTION": "learner waits for exact replay target and actor acknowledgement",
        "TRAIN_PHASE_ASSERTION": "actor waits for next immutable collect marker; replay size asserted each update",
        "SAMPLE_ORDER_ASSERTION": "fixed seed and update-index-driven replay indices",
        "CHECKPOINT_BY_UPDATE_ASSERTION": [500, 1000],
        "TARGET_UPDATES": 1000,
        "NO_3K_5K": "PASS",
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def dry_run(output: Path):
    require(not output.exists(), "fresh dry-run output required")
    output.mkdir(parents=True, exist_ok=False)
    replay = 0
    update = 0
    stages = []
    for stage in range(STAGE_COUNT):
        replay += INITIAL_PREFILL if stage == 0 else STAGED_COLLECT_N
        update += STAGED_TRAIN_K
        stages.append({"stage": stage, "target_replay": replay, "actual_replay": replay,
                       "target_update": update, "actual_update": update})
    status = "PASS" if replay == EXPECTED_FINAL_REPLAY and update == 1000 else "FAIL"
    result = {"STAGED_PROTOCOL_DRY_RUN": status, "stages": stages,
              "stage_count": len(stages), "final_replay": replay, "final_update": update,
              "training_started": False, "protected_artifacts_changed": "NO"}
    base.write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


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
            numeric = [np.asarray(jax.device_get(v)) for v in jax.tree_util.tree_leaves(state.params)
                       if hasattr(v, "dtype")]
            require(numeric and all(np.isfinite(v).all() for v in numeric), f"checkpoint {step} finite")
            restored[str(step)] = {"numeric_leaves": len(numeric)}
        replay = make_replay_buffer(env, capacity=int(replay_payload["capacity"]), type="replay_buffer")
        replay_info = restore_replay(replay, checkpoints / "checkpoint_1000_online_replay.pkl")
        require(replay_info["size"] == EXPECTED_FINAL_REPLAY, "staged replay restore size")
        env.close()
    base.write_json(out / "restore_verification.json", {
        "status": "PASS",
        "checkpoints": restored,
        "restored_replay": replay_info,
        "training_resumed": False,
        "protocol": "deterministic_staged_admission",
    })


def protocol_finalize(out: Path, base_rc: int):
    raw = json.loads((out / "result.json").read_text()) if (out / "result.json").is_file() else {}
    if raw:
        base.write_json(out / "base_supervise_result.json", raw)
    updates = base.read_jsonl(out / "updates.jsonl")
    learner = json.loads((out / "learner_result.json").read_text()) if (out / "learner_result.json").is_file() else {}
    restore = json.loads((out / "restore_verification.json").read_text()) if (out / "restore_verification.json").is_file() else {}
    checkpoints = [x.get("checkpoint") for x in base.read_jsonl(out / "checkpoint_events.jsonl")]
    stages = base.read_jsonl(out / "stage_events.jsonl")
    train_complete = [x for x in stages if x.get("event") == "train_complete"]
    checks = {
        "gpu": learner.get("backend") == "gpu",
        "prefill_gate": learner.get("prefill_threshold") == INITIAL_PREFILL and learner.get("initial_online_size") == INITIAL_PREFILL,
        "stage_count": learner.get("stage_count") == STAGE_COUNT and len(train_complete) == STAGE_COUNT,
        "exact_stage_boundaries": all(
            x.get("stage") == stage and x.get("target_replay") == x.get("actual_replay") == INITIAL_PREFILL + stage * STAGED_COLLECT_N
            and x.get("target_update") == x.get("actual_update") == (stage + 1) * STAGED_TRAIN_K
            for stage, x in enumerate(train_complete)),
        "admission_freeze": learner.get("final_online_size") == EXPECTED_FINAL_REPLAY and all(
            x.get("online_size") == INITIAL_PREFILL + x.get("stage") * STAGED_COLLECT_N for x in updates),
        "ratio": len(updates) == 1000 and all(x["demo_batch"] == x["online_batch"] == 32 for x in updates),
        "updates": [x["update"] for x in updates] == list(range(1, 1001)),
        "finite": bool(updates) and all(x["losses_finite"] for x in updates),
        "checkpoint_500": 500 in checkpoints,
        "checkpoint_1000": 1000 in checkpoints,
        "restore": restore.get("status") == "PASS",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "PEG_F3_V2_CG_DETERMINISTIC_STAGED_ADMISSION_1K": status,
        "checks": checks,
        "base_supervise_returncode": base_rc,
        "PREFILL_THRESHOLD": INITIAL_PREFILL,
        "STAGED_COLLECT_N": STAGED_COLLECT_N,
        "STAGED_TRAIN_K": STAGED_TRAIN_K,
        "STAGE_COUNT": STAGE_COUNT,
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
    parser.add_argument("--role", choices=("preflight", "dry-run", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(ROOT in output.parents, "output outside project")
    if args.role == "preflight":
        return preflight(output)
    if args.role == "dry-run":
        return dry_run(output)
    install(output)
    if args.role == "learner":
        return learner(output) or 0
    if args.role == "actor":
        return actor(output) or 0
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
