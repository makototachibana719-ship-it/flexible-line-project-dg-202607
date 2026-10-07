#!/usr/bin/env python3
"""One-shot F3 V2 RLPD 1000-update learner/actor smoke; never formal training."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import faulthandler
import hashlib
import json
import multiprocessing
import os
import pickle
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/peg_insertion/peg_v2_rlpd_smoke_1000_seed0.yaml"
OUTPUT = ROOT / "results/validation/peg/f3_v2_rlpd_smoke_1000_20260906"
SHORT_OUTPUT = ROOT / "results/validation/peg/f3_v2_short_shutdown_acceptance_20260906"
EXAMPLE = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
ENV_PARENT = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
FIELDS = ("observations", "actions", "next_observations", "rewards", "masks", "dones")
PORTS = (5595, 5596)
ACTIVE_OUTPUT = OUTPUT
RUN_UPDATES = 1000
SHORT_SHUTDOWN_ACCEPTANCE = False
POST_CLEANUP_EXIT_DIAGNOSTICS = False
FORMAL_5K = False
CHECKPOINT_TARGETS = (500, 1000)


def set_run_mode(short_shutdown_acceptance: bool, output: Path | None = None,
                 formal_config: Path | None = None) -> None:
    global ACTIVE_OUTPUT, RUN_UPDATES, SHORT_SHUTDOWN_ACCEPTANCE, CONFIG
    global FORMAL_5K, CHECKPOINT_TARGETS
    require(not (short_shutdown_acceptance and formal_config), "short/formal modes are exclusive")
    FORMAL_5K = formal_config is not None
    if FORMAL_5K:
        CONFIG = formal_config.resolve()
        require(ROOT in CONFIG.parents and CONFIG.is_file(), "formal config binding")
    SHORT_SHUTDOWN_ACCEPTANCE = short_shutdown_acceptance
    default_output = SHORT_OUTPUT if short_shutdown_acceptance else OUTPUT
    ACTIVE_OUTPUT = output.resolve() if output is not None else default_output
    require(ROOT in ACTIVE_OUTPUT.parents, "output outside project")
    if FORMAL_5K:
        cfg = config()
        RUN_UPDATES = int(cfg["formal"]["target_updates"])
        period = int(cfg["training"]["checkpoint_period"])
        CHECKPOINT_TARGETS = tuple(range(period, RUN_UPDATES + 1, period))
    else:
        RUN_UPDATES = 20 if short_shutdown_acceptance else 1000
        CHECKPOINT_TARGETS = (500, 1000)


def start_post_cleanup_diagnostics(out: Path) -> None:
    """Sample interpreter-visible state after actor cleanup without blocking exit."""
    first_snapshot = threading.Event()

    def sample():
        started = time.monotonic()
        for delay in (0, 2, 5, 10):
            remaining = delay - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            frames = sys._current_frames()
            threads = []
            for thread in threading.enumerate():
                frame = frames.get(thread.ident)
                threads.append({
                    "name": thread.name,
                    "ident": thread.ident,
                    "daemon": thread.daemon,
                    "alive": thread.is_alive(),
                    "stack": traceback.format_stack(frame) if frame is not None else [],
                })
            children = [{"pid": child.pid, "name": child.name, "alive": child.is_alive()}
                        for child in multiprocessing.active_children()]
            try:
                loop = asyncio.get_running_loop()
                pending = [repr(task) for task in asyncio.all_tasks(loop) if not task.done()]
                asyncio_state = {"running_loop": True, "pending": pending}
            except RuntimeError:
                asyncio_state = {"running_loop": False, "pending": []}
            payload = {
                "delay_seconds": delay,
                "timestamp_ns": time.time_ns(),
                "threads": threads,
                "non_daemon_threads": [item for item in threads if not item["daemon"]],
                "executor_threads": [item for item in threads
                                     if "executor" in item["name"].lower()],
                "multiprocessing_children": children,
                "asyncio": asyncio_state,
            }
            write_json(out / f"python_state_t{delay}.json", payload)
            with (out / f"faulthandler_t{delay}.txt").open("w") as stream:
                faulthandler.dump_traceback(file=stream, all_threads=True)
            if delay == 0:
                first_snapshot.set()

    sampler = threading.Thread(target=sample, name="post-cleanup-diagnostic-sampler",
                               daemon=True)
    sampler.start()
    require(first_snapshot.wait(timeout=2), "post-cleanup T+0 snapshot")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def config():
    return yaml.safe_load(CONFIG.read_text())


def configure():
    from serl_launcher.utils.launcher import make_trainer_config
    return dataclasses.replace(make_trainer_config(), port_number=PORTS[0], broadcast_port=PORTS[1])


def runtime_env():
    return dict(
        os.environ,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false",
        MUJOCO_GL="egl",
        WANDB_MODE="disabled",
        PYTHONDONTWRITEBYTECODE="1",
    )


def validate_config_and_dataset(check_gpu=True):
    from training.peg_insertion.observation_contract import V2, validate_config
    cfg = config()
    validate_config(cfg, for_training=True)
    require(cfg["fresh_initialization"] is True and cfg["load_legacy_checkpoint"] is False,
            "fresh/V1 checkpoint policy")
    require(cfg["observation"] == {"contract_version": V2, "shape": [20], "dtype": "float32"},
            "V2 observation config")
    require(cfg["action"] == {"shape": [7], "dtype": "float32", "low": -1.0, "high": 1.0},
            "action config")
    require(cfg["training"]["online_batch_size"] == cfg["training"]["demo_batch_size"] == 32,
            "32/32 RLPD ratio")
    require(cfg["training"]["total_batch_size"] == 64 and cfg["training"]["utd_ratio"] == 1,
            "batch/UTD config")
    if FORMAL_5K:
        require(cfg["formal"]["target_updates"] == 5000 and
                cfg["training"]["checkpoint_period"] == 1000,
                "formal update/checkpoint config")
    else:
        require(cfg["smoke"]["learner_updates"] == 1000 and
                cfg["smoke"]["checkpoint_period"] == 500,
                "bounded update/checkpoint config")
    dataset_path = ROOT / cfg["demo"]["path"]
    manifest_path = ROOT / cfg["demo"]["manifest"]
    require(dataset_path.is_file() and manifest_path.is_file(), "dataset/manifest missing")
    require(sha256(dataset_path) == cfg["demo"]["sha256"], "dataset SHA256")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "PASS" and manifest.get("v2_only") is True,
            "dataset manifest V2 gate")
    require(manifest.get("episodes") == 11 and manifest.get("transitions") == 1921,
            "dataset manifest cardinality")
    require(manifest.get("dataset_sha256") == cfg["demo"]["sha256"], "manifest hash binding")
    rows = pickle.loads(dataset_path.read_bytes())
    require(isinstance(rows, list) and len(rows) == 1921, "dataset load/cardinality")
    for row in rows:
        require(set(row) == set(FIELDS), "dataset field schema")
        require(row["observations"].shape == row["next_observations"].shape == (20,), "obs20")
        require(row["actions"].shape == (7,), "action7")
        require(row["observations"].dtype == row["next_observations"].dtype == np.float32,
                "observation float32")
        require(row["actions"].dtype == np.float32, "action float32")
        require(all(np.isfinite(np.asarray(row[key])).all() for key in FIELDS), "dataset finite")
    backend = "NOT_CHECKED"
    if check_gpu:
        probe = subprocess.run(
            [sys.executable, "-c", "import jax; print(jax.default_backend())"],
            env=runtime_env(), capture_output=True, text=True, timeout=60, check=True)
        backend = probe.stdout.strip().splitlines()[-1]
        require(backend == "gpu", "JAX GPU backend required")
    return cfg, rows, backend


def block_tree(jax, tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def loss_scalars(jax, info):
    result = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(info)[0]:
        array = np.asarray(jax.device_get(leaf))
        if "loss" in str(path).lower() and array.size == 1:
            result[str(path)] = float(array.reshape(()))
    return result


def learner(out: Path):
    import jax
    from agentlace.trainer import TrainerServer
    from serl_launcher.utils.launcher import make_replay_buffer
    sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
    from async_sac_state_sim import concat_rlpd_batches, make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import install_actor_step_tracking, save_bundle

    cfg, data, backend = validate_config_and_dataset(check_gpu=False)
    require(jax.default_backend() == "gpu", "learner GPU backend")
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    run_id = "wandb-disabled-f3-v2-formal-5k" if FORMAL_5K else "wandb-disabled-f3-v2-smoke-1000"
    state = {"live": 0}
    event_lock = threading.Lock()

    def event(value):
        value["timestamp_ns"] = time.time_ns()
        with event_lock:
            append_jsonl(out / "communication.jsonl", value)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        require(np.asarray(obs).shape == (20,) and np.asarray(obs).dtype == np.float32,
                "learner V2 env observation")
        require(env.action_space.shape == (7,) and env.action_space.dtype == np.float32,
                "learner action contract")
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        initial_rng = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
        online = make_replay_buffer(env, capacity=cfg["training"]["replay_capacity"],
                                    type="replay_buffer")
        demo = make_replay_buffer(env, capacity=len(data) + 100, type="replay_buffer")
        install_actor_step_tracking(online, 0)
        for transition in data:
            demo.insert(transition)
        require(len(demo) == 1921 and len(online) == 0, "fresh replay population")
        server = TrainerServer(configure(),
            data_callback=lambda name, payload: event({"event": "live_insert",
                "count": len(payload["data"]), "online_size": len(online)}),
            request_callback=lambda kind, payload: {})
        server.register_data_store("actor_env", online)
        server.start(threaded=True)
        write_json(out / "server_ready.json", {"pid": os.getpid(), "backend": backend})
        deadline = time.monotonic() + 180
        while len(online) < cfg["training"]["training_starts"]:
            if time.monotonic() > deadline:
                raise TimeoutError("online replay startup deadline")
            time.sleep(.05)
        state["live"] = len(online)
        server.publish_network(agent.state.params)
        event({"event": "publish_initial", "online_size": len(online)})
        sharding = jax.sharding.PositionalSharding(jax.local_devices()).replicate()
        online_iter = online.get_iterator(sample_args={"batch_size": 32}, device=sharding)
        demo_iter = demo.get_iterator(sample_args={"batch_size": 32}, device=sharding)
        env.close()

        completed = 0
        try:
            for update in range(1, RUN_UPDATES + 1):
                online_batch = next(online_iter)
                demo_batch = next(demo_iter)
                require(online_batch["observations"].shape == demo_batch["observations"].shape == (32, 20),
                        "32 Demo + 32 Online observation batches")
                require(online_batch["actions"].shape == demo_batch["actions"].shape == (32, 7),
                        "32 Demo + 32 Online action batches")
                batch = concat_rlpd_batches(online_batch, demo_batch)
                require(batch["observations"].shape == (64, 20) and batch["actions"].shape == (64, 7),
                        "combined batch contract")
                timed_out = {"value": False}
                def timeout():
                    timed_out["value"] = True
                    write_json(out / "timeout.json", {"update": update, "limit": 120})
                    os._exit(124)
                watchdog = threading.Timer(cfg["smoke"]["update_timeout_seconds"], timeout)
                watchdog.daemon = True
                watchdog.start()
                started = time.perf_counter()
                try:
                    new_agent, info = agent.update_high_utd(batch, utd_ratio=1)
                    block_tree(jax, (new_agent, info))
                finally:
                    watchdog.cancel()
                losses = loss_scalars(jax, info)
                require(len(losses) >= 3 and all(np.isfinite(v) for v in losses.values()),
                        "loss/Q/temperature finite")
                require(any("critic" in key.lower() for key in losses), "Q/critic metric missing")
                require(any("temperature" in key.lower() for key in losses), "temperature metric missing")
                rng_before = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
                rng_after = np.asarray(jax.device_get(new_agent.state.rng), dtype=np.uint32).tolist()
                require(rng_before != rng_after, "learner RNG did not advance")
                agent = new_agent
                completed = update
                append_jsonl(out / "updates.jsonl", {
                    "update": update, "seconds": time.perf_counter() - started,
                    "losses": losses, "losses_finite": True,
                    "online_size": len(online), "demo_size": len(demo),
                    "online_batch": 32, "demo_batch": 32,
                    "rng_before": rng_before, "rng_after": rng_after})
                server.publish_network(agent.state.params)
                event({"event": "publish_update", "update": update})
                if update in CHECKPOINT_TARGETS:
                    source = {"identifier": "peg-f3-v2-formal-5k" if FORMAL_5K else "peg-f3-v2-rlpd-smoke-1000",
                              "profile": "formal_5k" if FORMAL_5K else "bounded_smoke",
                              "fresh_initialization": True, "resume": False,
                              "demo_online_ratio": "32:32", "utd_ratio": 1}
                    save_bundle(checkpoint_dir, agent.state, online, update,
                                int(online.project_actor_steps), cfg,
                                ROOT / cfg["demo"]["manifest"], run_id, source)
                    append_jsonl(out / "checkpoint_events.jsonl", {
                        "event": "save_complete", "checkpoint": update,
                        "online_size": len(online), "actor_steps": int(online.project_actor_steps)})
        finally:
            write_json(out / "stop_actor.json", {
                "completed_updates": completed, "timestamp_ns": time.time_ns()})
            server.stop()
        write_json(out / "learner_result.json", {
            "status": "PASS", "backend": jax.default_backend(), "fresh_initialization": True,
            "demo_size": len(demo), "initial_online_size": state["live"],
            "final_online_size": len(online), "actor_steps": int(online.project_actor_steps),
            "completed_updates": completed, "initial_rng": initial_rng,
            "wandb_run": "DISABLED", "run_id": run_id,
            "requested_updates": RUN_UPDATES})


def actor(out: Path):
    import jax
    from agentlace.data.data_store import QueuedDataStore
    from training.peg_insertion.actor_shutdown import BoundedTrainerClient
    sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    cfg = config()
    require(jax.default_backend() == "gpu", "actor GPU backend")
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        require(np.asarray(obs).shape == (20,), "actor V2 observation")
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        sampling_rng = jax.random.split(jax.random.PRNGKey(cfg["seed"]))[1]
        store = QueuedDataStore(2000)
        client = BoundedTrainerClient(
            "actor_env", "127.0.0.1", configure(), store, wait_for_server=False,
            shutdown_timeout_seconds=10,
            shutdown_evidence_path=out / "actor_shutdown.json")
        received = [0]
        def network(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))
            received[0] += 1
            append_jsonl(out / "actor_network.jsonl", {"event": "received", "count": received[0]})
        client.recv_network_callback(network)
        steps = 0
        try:
            while not (out / "stop_actor.json").exists():
                require(steps < cfg["smoke"]["actor_max_steps"], "actor bounded step limit")
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
                if steps % 30 == 0:
                    require(client.update(), "actor upload failed")
                steps += 1
        finally:
            try:
                client.update()
            except Exception:
                pass
            write_json(out / "actor_stop_observed.json", {
                "stop_marker": (out / "stop_actor.json").exists(), "steps": steps,
                "timestamp_ns": time.time_ns()})
            write_json(out / "actor_result.json", {"status": "PASS", "backend": jax.default_backend(),
                "steps": steps, "received_networks": received[0],
                "stopped_by_learner": (out / "stop_actor.json").exists()})
            shutdown = client.stop()
            env.close()
            write_json(out / "actor_cleanup_complete.json", {
                "status": "PASS", "shutdown": shutdown, "environment_closed": True,
                "timestamp_ns": time.time_ns()})
            if POST_CLEANUP_EXIT_DIAGNOSTICS:
                start_post_cleanup_diagnostics(out)


def verify(out: Path):
    import jax
    from serl_launcher.utils.launcher import make_replay_buffer
    sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import load_bundle, restore_replay
    from training.peg_insertion.checkpoint import restore_fresh_state
    cfg = config()
    run_id = "wandb-disabled-f3-v2-formal-5k" if FORMAL_5K else "wandb-disabled-f3-v2-smoke-1000"
    checkpoints = out / "checkpoints"
    bundle_checks = {}
    for step in CHECKPOINT_TARGETS:
        cp = checkpoints / f"checkpoint_{step}"
        meta = load_bundle(cp, cfg, ROOT / cfg["demo"]["manifest"], run_id)
        bundle_checks[str(step)] = {
            "metadata": meta["checkpoint_step"] == meta["learner_update_steps"] == step,
            "demo": meta["demo_sha256"] == cfg["demo"]["sha256"] and
                    meta["demo_transition_count"] == 1921,
            "V2": meta["observation_contract_version"] == cfg["observation"]["contract_version"],
            "replay_size": meta["online_replay_size"] > 0}
        require(all(bundle_checks[str(step)].values()), f"checkpoint_{step} bundle")
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        restored = restore_fresh_state(checkpoints, fresh.state, CHECKPOINT_TARGETS[-1])
        block_tree(jax, restored)
        numeric = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(restored)
                   if hasattr(x, "dtype")]
        require(numeric and all(np.isfinite(x).all() for x in numeric), "restored state finite")
        replay = make_replay_buffer(env, capacity=cfg["training"]["replay_capacity"], type="replay_buffer")
        replay_info = restore_replay(replay, checkpoints / f"checkpoint_{CHECKPOINT_TARGETS[-1]}_online_replay.pkl")
        require(replay_info["size"] > 0 and replay_info["actor_steps"] > 0, "restored replay")
        sample = replay.sample(32)
        require(sample["observations"].shape == (32, 20) and sample["actions"].shape == (32, 7),
                "restored replay sample")
        env.close()
    write_json(out / "restore_verification.json", {"status": "PASS", "checkpoints": bundle_checks,
        "restored_numeric_leaves": len(numeric), "restored_replay": replay_info,
        "training_resumed": False})


def preflight():
    require(not ACTIVE_OUTPUT.exists(), "smoke output already exists")
    for port in PORTS:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    cfg, rows, backend = validate_config_and_dataset(check_gpu=True)
    result = {"status": "PASS", "backend": backend, "dataset_sha256": cfg["demo"]["sha256"],
              "episodes": cfg["demo"]["episodes"], "transitions": len(rows),
              "demo_batch": 32, "online_batch": 32, "utd": 1,
              "updates": RUN_UPDATES,
              "checkpoint_period": ("NOT_RUN" if SHORT_SHUTDOWN_ACCEPTANCE else
                                    cfg["training"]["checkpoint_period"] if FORMAL_5K else 500),
              "checkpoint_targets": list(CHECKPOINT_TARGETS),
              "formal_5k": FORMAL_5K,
              "short_shutdown_acceptance": SHORT_SHUTDOWN_ACCEPTANCE,
              "fresh_initialization": True, "load_legacy_checkpoint": False}
    print(json.dumps(result, indent=2))
    return result


def supervise():
    pre = preflight()
    out = ACTIVE_OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "preflight.json", pre)
    env = runtime_env()
    children, streams = [], []
    error = None
    def launch(role):
        stream = (out / f"{role}.log").open("w")
        streams.append(stream)
        mode_args = (["--short-shutdown-acceptance"] if SHORT_SHUTDOWN_ACCEPTANCE else [])
        if FORMAL_5K:
            mode_args += ["--formal-5k-config", str(CONFIG), "--authorize-formal-5k"]
        child = subprocess.Popen([sys.executable, "-u", __file__, "--role", role,
                                  "--output", str(out)] +
                                 mode_args,
                                 env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    monitor_stream = (out / "gpu_dmon.log").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "dmon", "-d", "1"], stdout=monitor_stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        learner_process = launch("learner")
        ready_deadline = time.monotonic() + 300
        while not (out / "server_ready.json").exists():
            if learner_process.poll() is not None or time.monotonic() > ready_deadline:
                raise RuntimeError("learner failed before server ready")
            time.sleep(.2)
        actor_process = launch("actor")
        deadline = time.monotonic() + config()["smoke"]["process_timeout_seconds"]
        while learner_process.poll() is None:
            if actor_process.poll() is not None and not (out / "stop_actor.json").exists():
                raise RuntimeError("actor exited before learner")
            if time.monotonic() > deadline:
                raise TimeoutError("bounded smoke process timeout")
            time.sleep(1)
        actor_process.wait(timeout=30)
        require(learner_process.returncode == actor_process.returncode == 0, "learner/actor exit")
        actor_exit_timestamp_ns = time.time_ns()
        if not SHORT_SHUTDOWN_ACCEPTANCE:
            verify_process = launch("verify")
            verify_process.wait(timeout=180)
            require(verify_process.returncode == 0, "checkpoint restore verification")
    except Exception as exc:
        error = repr(exc)
    finally:
        for process in children + [monitor]:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for stream in streams + [monitor_stream]:
            stream.close()
    updates = read_jsonl(out / "updates.jsonl")
    learner_result = json.loads((out / "learner_result.json").read_text()) if (out / "learner_result.json").is_file() else {}
    actor_result = json.loads((out / "actor_result.json").read_text()) if (out / "actor_result.json").is_file() else {}
    restore_result = json.loads((out / "restore_verification.json").read_text()) if (out / "restore_verification.json").is_file() else {}
    cleanup_result = json.loads((out / "actor_cleanup_complete.json").read_text()) if (out / "actor_cleanup_complete.json").is_file() else {}
    stop_result = json.loads((out / "stop_actor.json").read_text()) if (out / "stop_actor.json").is_file() else {}
    stop_observed = json.loads((out / "actor_stop_observed.json").read_text()) if (out / "actor_stop_observed.json").is_file() else {}
    communication = read_jsonl(out / "communication.jsonl")
    checkpoints = [x.get("checkpoint") for x in read_jsonl(out / "checkpoint_events.jsonl")]
    actor_exit_seconds = ((actor_exit_timestamp_ns - stop_result["timestamp_ns"]) / 1e9
                          if "actor_exit_timestamp_ns" in locals() and stop_result.get("timestamp_ns") else None)
    shutdown = cleanup_result.get("shutdown", {})
    checks = {
        "gpu": learner_result.get("backend") == actor_result.get("backend") == "gpu",
        "demo": learner_result.get("demo_size") == 1921,
        "online_growth": learner_result.get("final_online_size", 0) > learner_result.get("initial_online_size", 0) >= 64,
        "ratio": len(updates) == RUN_UPDATES and all(x["demo_batch"] == x["online_batch"] == 32 for x in updates),
        "updates": [x["update"] for x in updates] == list(range(1, RUN_UPDATES + 1)),
        "finite": bool(updates) and all(x["losses_finite"] for x in updates),
        "communication": actor_result.get("received_networks", 0) > 0 and actor_result.get("steps", 0) > 0,
        "checkpoint_500": SHORT_SHUTDOWN_ACCEPTANCE or 500 in checkpoints,
        "checkpoint_1000": SHORT_SHUTDOWN_ACCEPTANCE or 1000 in checkpoints,
        "restore": SHORT_SHUTDOWN_ACCEPTANCE or restore_result.get("status") == "PASS",
        "children_exited": all(process.poll() is not None for process in children),
        "learner_stop": stop_result.get("completed_updates") == RUN_UPDATES,
        "actor_stop_observed": stop_observed.get("stop_marker") is True,
        "trainerclient_stop": cleanup_result.get("status") == "PASS",
        "broadcast_join": shutdown.get("status") == "PASS" and not shutdown.get("timed_out_threads"),
        "actor_natural_exit": actor_exit_seconds is not None and actor_exit_seconds <= 30,
        "real_communication": any(x.get("event") == "live_insert" for x in communication),
    }
    status = "PASS" if error is None and all(checks.values()) else "FAIL"
    if SHORT_SHUTDOWN_ACCEPTANCE:
        result = {
            "PEG_F3_SHORT_SHUTDOWN_ACCEPTANCE": status,
            "LEARNER_UPDATES": len(updates),
            "ACTOR_STEPS": actor_result.get("steps", 0),
            "NETWORK_UPDATES_RECEIVED": actor_result.get("received_networks", 0),
            "ONLINE_REPLAY_FINAL": learner_result.get("final_online_size", 0),
            "LEARNER_STOP_SIGNAL": "PASS" if checks["learner_stop"] else "FAIL",
            "ACTOR_STOP_SIGNAL_OBSERVED": "PASS" if checks["actor_stop_observed"] else "FAIL",
            "TRAINERCLIENT_STOP": "PASS" if checks["trainerclient_stop"] else "FAIL",
            "BROADCAST_THREAD_JOIN": "PASS" if checks["broadcast_join"] else "FAIL",
            "ACTOR_NATURAL_EXIT": "PASS" if checks["actor_natural_exit"] else "FAIL",
            "ACTOR_EXIT_TIME_SEC": actor_exit_seconds,
            "RESIDUAL_PROCESSES": "NONE" if checks["children_exited"] else "PRESENT",
            "REAL_COMMUNICATION": "PASS" if checks["real_communication"] else "FAIL",
            "LEARNER_UPDATE": "PASS" if checks["updates"] else "FAIL",
            "ACTOR_NETWORK_UPDATE_RECEIVED": "PASS" if actor_result.get("received_networks", 0) > 0 else "FAIL",
            "PEG_F3_V2_RLPD_SMOKE_FINAL": status,
            "TRAINING_READY": "YES" if status == "PASS" else "NO",
            "error": error,
            "formal_training_started": False,
            "resume_training_started": False,
            "checkpoint_written": False,
        }
        write_json(out / "result.json", result)
        print(json.dumps(result, indent=2))
        return 0 if status == "PASS" else 1
    result = {"PEG_F3_V2_RLPD_SMOKE": status, "checks": checks, "error": error,
              "GPU_BACKEND": learner_result.get("backend", "UNAVAILABLE"),
              "DATASET_HASH": "PASS" if pre["dataset_sha256"] else "FAIL",
              "DEMO_BUFFER": learner_result.get("demo_size", "UNAVAILABLE"),
              "ONLINE_REPLAY": {"initial": learner_result.get("initial_online_size"),
                                "final": learner_result.get("final_online_size")},
              "DEMO_ONLINE_RATIO": "32:32" if checks["ratio"] else "FAIL",
              "LEARNER_UPDATES": len(updates), "LOSSES_FINITE": checks["finite"],
              "ACTOR_LEARNER_COMM": checks["communication"],
              "CHECKPOINT_500": checks["checkpoint_500"],
              "CHECKPOINT_1000": checks["checkpoint_1000"],
              "CHECKPOINT_RESTORE": checks["restore"],
              "RESIDUAL_PROCESSES": "NONE" if checks["children_exited"] else "PRESENT",
              "wandb_run": "DISABLED", "TRAINING_READY": "YES" if status == "PASS" else "NO",
              "formal_training_started": False, "resume_training_started": False,
              "retry": False}
    write_json(out / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--short-shutdown-acceptance", action="store_true")
    parser.add_argument("--formal-5k-config", type=Path)
    parser.add_argument("--authorize-formal-5k", action="store_true")
    parser.add_argument("--post-cleanup-exit-diagnostics", action="store_true")
    parser.add_argument("--interpreter-exit-diagnostics", action="store_true")
    parser.add_argument("--interpreter-exit-path", choices=("exception", "normal"))
    args = parser.parse_args()
    global POST_CLEANUP_EXIT_DIAGNOSTICS
    POST_CLEANUP_EXIT_DIAGNOSTICS = args.post_cleanup_exit_diagnostics
    set_run_mode(args.short_shutdown_acceptance, args.output, args.formal_5k_config)
    if args.role == "preflight":
        preflight()
        return 0
    if args.role == "supervise":
        require(not FORMAL_5K or args.authorize_formal_5k,
                "formal 5K requires explicit --authorize-formal-5k")
        return supervise()
    require(args.output is not None and ROOT in args.output.resolve().parents, "output outside project")
    probe = None
    if args.role == "actor" and args.interpreter_exit_diagnostics:
        require(args.interpreter_exit_path is not None, "interpreter exit path label required")
        from scripts.peg_actor_interpreter_exit_probe import InterpreterExitProbe
        probe = InterpreterExitProbe(args.output.resolve(), args.interpreter_exit_path)
        probe.install()
    try:
        {"learner": learner, "actor": actor, "verify": verify}[args.role](args.output.resolve())
    finally:
        if probe is not None:
            probe.top_level_finally()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
