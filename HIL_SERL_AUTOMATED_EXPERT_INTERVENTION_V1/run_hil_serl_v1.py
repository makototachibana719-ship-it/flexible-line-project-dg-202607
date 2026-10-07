#!/usr/bin/env python3
"""Isolated 500-update HIL-SERL causal pilot for paper-fidelity Aubo Peg."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np
import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
STAGE_B = ROOT / "SERL_PAPER_FIDELITY_AUBO_PEG_V2"
CONFIG = BRANCH / "config.yaml"
OUTPUT = BRANCH / "runs/pilot_seed0"
EXAMPLE = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
ENV_PARENT = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
FIELDS = ("observations", "actions", "next_observations", "rewards", "masks", "dones")
PORTS = (5695, 5696)
TARGETS = (100, 250, 500)
RUN_UPDATES = 500

sys.path[:0] = [str(ROOT), str(STAGE_B), str(EXAMPLE), str(ENV_PARENT)]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cfg():
    return yaml.safe_load(CONFIG.read_text())


def load_base():
    spec = importlib.util.spec_from_file_location("stage_b_runtime", STAGE_B / "run_paper_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.CONFIG = CONFIG
    module.CHECKPOINT_TARGETS = TARGETS
    module.RUN_UPDATES = RUN_UPDATES
    return module


def configure():
    from serl_launcher.utils.launcher import make_trainer_config
    return dataclasses.replace(make_trainer_config(), port_number=PORTS[0], broadcast_port=PORTS[1])


def runtime_env():
    return dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl",
                WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")


def validate(check_gpu=True):
    c = cfg()
    require(c["fresh_initialization"] and not c["load_legacy_checkpoint"], "fresh initialization")
    require(c["training"]["online_batch_size"] == c["training"]["demo_batch_size"] == 32,
            "symmetric 32/32 replay")
    require(c["training"]["total_batch_size"] == 64 and c["training"]["utd_ratio"] == 8,
            "batch/UTD frozen")
    require(c["training"]["critic_actor_ratio"] == 1 and c["training"]["reset_randomization"],
            "actor/update/reset frozen")
    demo_path, manifest_path = ROOT / c["demo"]["path"], ROOT / c["demo"]["manifest"]
    require(sha(demo_path) == c["demo"]["sha256"], "20-demo SHA")
    manifest = json.loads(manifest_path.read_text())
    data = pickle.loads(demo_path.read_bytes())
    require(manifest["status"] == "PASS" and manifest["episodes"] == 20 and len(data) == 3627,
            "20-demo manifest/cardinality")
    for row in data:
        require(set(row) == set(FIELDS) and row["observations"].shape == (20,) and
                row["actions"].shape == (7,), "canonical replay schema")
    expert_path = ROOT / c["production_expert"]["path"]
    require(sha(expert_path) == c["production_expert"]["sha256"], "production expert SHA")
    expert = pickle.loads(expert_path.read_bytes())
    require(len(expert) == c["production_expert"]["transitions"] == 1921,
            "validated production expert cardinality")
    backend = "NOT_CHECKED"
    if check_gpu:
        import subprocess
        probe = subprocess.run([sys.executable, "-c", "import jax; print(jax.default_backend())"],
                               env=runtime_env(), capture_output=True, text=True, check=True, timeout=60)
        backend = probe.stdout.strip().splitlines()[-1]
        require(backend == "gpu", "GPU backend")
    return c, data, expert, backend


class ValidatedExactLookaheadProvider:
    """Existing nearest-demo state-feedback expert plus preregistered support proxy."""

    def __init__(self, env, oracle_env, expert_rows, threshold):
        from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
        self.env, self.oracle_env = env, oracle_env
        self.capture, self.restore = capture, restore
        self.observations = np.asarray([row["observations"] for row in expert_rows], np.float64)
        self.actions = np.asarray([row["actions"] for row in expert_rows], np.float32)
        lo, hi = np.percentile(self.observations, [1, 99], axis=0)
        self.scale = np.maximum(hi - lo, 1e-6)
        self.threshold = float(threshold)
        self.active = False
        self.call_count = 0
        self.lookahead_count = 0

    def nearest(self, obs):
        d = np.sqrt(np.sum(np.square((self.observations - np.asarray(obs, np.float64)) /
                                     self.scale), axis=1))
        index = int(np.argmin(d))
        return float(d[index]), index

    @staticmethod
    def state_hash(snapshot):
        h = hashlib.sha256()
        h.update(np.asarray(snapshot["observation"], np.float32).tobytes())
        h.update(np.asarray(snapshot["mujoco"], np.float64).tobytes())
        return h.hexdigest()

    def decide(self, *, obs20, policy_action):
        from scripts.screen_peg_f3_v2_formal_5k import contact_flags
        from training.peg_insertion.hil_serl.intervention_adapter_v1 import InterventionDecision
        self.call_count += 1
        current_d, current_index = self.nearest(obs20)
        snapshot = self.capture(self.env, obs20)
        state_sha = self.state_hash(snapshot)
        common = {"state_hash": state_sha, "current_D_DEMO": current_d,
                  "support_threshold": self.threshold, "expert_source_row": current_index,
                  "provider": "validated_nearest_normalized_Obs20_production_Demo"}

        if self.active and current_d <= self.threshold:
            self.active = False
            return InterventionDecision(False, provenance={**common, "recovery_restored": True,
                "trigger_reason": "RELEASE_ON_VALIDATED_SUPPORT_RETURN"})
        if self.active:
            return InterventionDecision(True, self.actions[current_index], {**common,
                "recovery_restored": False, "trigger_reason": "CONTINUE_UNTIL_SUPPORT_RETURN"})

        self.lookahead_count += 1
        self.restore(self.oracle_env, snapshot)
        next_obs, _, terminated, truncated, info = self.oracle_env.step(policy_action)
        next_d, _ = self.nearest(next_obs)
        link, unexpected, _ = contact_flags(self.oracle_env)
        healthy = bool(not link and not unexpected and np.isfinite(float(info.get("force_norm", 0.0))))
        crossing = bool(next_d > self.threshold or not healthy)
        lookahead = {**common, "next_D_DEMO_under_policy": next_d,
                     "policy_successor_contact_healthy": healthy,
                     "policy_successor_terminal": bool(terminated or truncated)}
        if crossing:
            self.active = True
            reason = "POLICY_ONE_STEP_EXITS_VALIDATED_SUPPORT" if next_d > self.threshold else "POLICY_ONE_STEP_UNHEALTHY_CONTACT"
            return InterventionDecision(True, self.actions[current_index], {**lookahead,
                "recovery_restored": False, "trigger_reason": reason})
        return InterventionDecision(False, provenance={**lookahead,
            "recovery_restored": True, "trigger_reason": "POLICY_SUCCESSOR_REMAINS_RECOVERABLE"})


def preflight(out: Path | None = None):
    from paper_fidelity_env import make_paper_fidelity_env
    from training.peg_insertion.hil_serl.intervention_adapter_v1 import HILSERLInterventionAdapterV1
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    for port in PORTS:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    c, _, expert, backend = validate(check_gpu=True)
    with portable_peg_runtime(ROOT):
        env = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                      training_reset_randomization=True, randomization_seed=1)
        oracle = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                         training_reset_randomization=True, randomization_seed=2)
        obs, _ = env.reset(seed=0)
        oracle.reset(seed=0)
        provider = ValidatedExactLookaheadProvider(env, oracle, expert,
                                                   c["production_expert"]["support_threshold"])
        decision = provider.decide(obs20=obs, policy_action=np.zeros(7, np.float32))
        require(provider.call_count == provider.lookahead_count == 1, "one exact lookahead")
        require(decision.expert_action is None or (decision.expert_action.shape == (7,) and
                np.isfinite(decision.expert_action).all() and np.max(np.abs(decision.expert_action)) <= 1),
                "legal expert Action7")
        formal = HILSERLInterventionAdapterV1(mode="FORMAL_EVAL", intervention_enabled=False, provider=None)
        require(not formal.enabled, "formal hard-off")
        oracle.close(); env.close()
    result = {"HIL_SERL_V1_PREFLIGHT": "PASS", "GPU_BACKEND": backend,
              "TRIGGER": c["production_expert"]["trigger"],
              "SUPPORT_THRESHOLD": c["production_expert"]["support_threshold"],
              "EXPERT_SHA": c["production_expert"]["sha256"],
              "FORMAL_INTERVENTION_HARD_OFF": "PASS", "EXACT_LOOKAHEAD": "PASS",
              "LEGAL_ACTION7": "PASS"}
    if out is not None:
        dump(out / "preflight.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def learner(out: Path):
    import jax
    from agentlace.trainer import TrainerServer
    from paper_fidelity_env import make_paper_fidelity_env
    from serl_launcher.utils.launcher import make_replay_buffer, make_sac_agent
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import install_actor_step_tracking, save_bundle
    base = load_base()
    c, data, _, backend = validate(check_gpu=False)
    require(jax.default_backend() == "gpu", "learner GPU")
    checkpoint_dir = out / "checkpoints"; checkpoint_dir.mkdir(exist_ok=False)
    run_id = "wandb-disabled-hil-serl-aubo-peg-v1"
    event_lock = threading.Lock()
    def event(value):
        value["timestamp_ns"] = time.time_ns()
        with event_lock: append(out / "communication.jsonl", value)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                      training_reset_randomization=True, randomization_seed=c["seed"])
        agent = make_sac_agent(seed=c["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        online = make_replay_buffer(env, capacity=c["training"]["replay_capacity"], type="replay_buffer")
        demo = make_replay_buffer(env, capacity=c["training"]["demo_replay_capacity"], type="replay_buffer")
        install_actor_step_tracking(online, 0)
        for row in data: demo.insert(row)
        base_demo_size = len(demo)
        server = TrainerServer(configure(), data_callback=lambda name, payload: event({
            "event": "insert", "store": name, "count": len(payload["data"]),
            "online_size": len(online), "demo_size": len(demo)}), request_callback=lambda kind, payload: {})
        server.register_data_store("actor_env", online)
        server.register_data_store("actor_env_intvn", demo)
        server.start(threaded=True)
        dump(out / "server_ready.json", {"pid": os.getpid(), "backend": backend})
        deadline = time.monotonic() + 180
        while len(online) < c["training"]["training_starts"]:
            require(time.monotonic() < deadline, "online startup timeout"); time.sleep(.05)
        initial_online = len(online)
        server.publish_network(agent.state.params)
        sharding = jax.sharding.PositionalSharding(jax.local_devices()).replicate()
        online_iter = online.get_iterator(sample_args={"batch_size": 32}, device=sharding)
        demo_iter = demo.get_iterator(sample_args={"batch_size": 32}, device=sharding)
        env.close(); completed = 0
        try:
            for update in range(1, RUN_UPDATES + 1):
                ob, db = next(online_iter), next(demo_iter)
                batch = base.concat_rlpd_batches(ob, db)
                require(batch["observations"].shape == (64, 20) and batch["actions"].shape == (64, 7),
                        "32 online + 32 augmented-demo")
                started = time.perf_counter()
                agent, info = agent.update_high_utd(batch, utd_ratio=8)
                base.block_tree(jax, (agent, info))
                losses = base.loss_scalars(jax, info)
                require(losses and all(np.isfinite(v) for v in losses.values()), "finite losses")
                completed = update
                append(out / "updates.jsonl", {"update": update, "seconds": time.perf_counter()-started,
                    "losses": losses, "losses_finite": True, "online_size": len(online),
                    "demo_size": len(demo), "intervention_buffer_size": len(demo)-base_demo_size,
                    "online_batch": 32, "demo_batch": 32})
                server.publish_network(agent.state.params)
                if update in TARGETS:
                    dump(out / f"pause_actor_{update}.request.json", {"checkpoint": update, "timestamp_ns": time.time_ns()})
                    pause_deadline = time.monotonic() + 120
                    while not (out / f"pause_actor_{update}.ack.json").exists():
                        require(time.monotonic() < pause_deadline, "actor pause timeout"); time.sleep(.05)
                    time.sleep(.2)
                    source = {"identifier": "HIL_SERL_AUTOMATED_EXPERT_INTERVENTION_V1",
                              "profile": "fresh_hil_serl_v1", "fresh_initialization": True,
                              "resume": False, "demo_online_ratio": "32:32", "utd_ratio": 8,
                              "critic_actor_ratio": 1, "demo_episodes": 20,
                              "training_reset_randomization": True,
                              "training_only_intervention": True,
                              "intervention_buffer_size": len(demo)-base_demo_size,
                              "critic_updates": update*8, "actor_updates": update}
                    save_bundle(checkpoint_dir, agent.state, online, update, int(online.project_actor_steps),
                                c, ROOT / c["demo"]["manifest"], run_id, source)
                    dump(out / f"checkpoint_{update}.ready.json", {"checkpoint": update,
                         "actor_paused": True, "critic_updates": update*8, "actor_updates": update,
                         "intervention_buffer_size": len(demo)-base_demo_size})
                    decision_path = out / f"decision_{update}.json"
                    decision_deadline = time.monotonic() + c["pilot"]["decision_timeout_seconds"]
                    while not decision_path.exists():
                        require(time.monotonic() < decision_deadline, "decision timeout"); time.sleep(.1)
                    if json.loads(decision_path.read_text())["action"] == "stop": break
        finally:
            dump(out / "stop_actor.json", {"completed_updates": completed, "timestamp_ns": time.time_ns()})
            server.stop()
        dump(out / "learner_result.json", {"status": "PASS", "backend": jax.default_backend(),
             "base_demo_size": base_demo_size, "demo_size": len(demo),
             "intervention_buffer_size": len(demo)-base_demo_size,
             "initial_online_size": initial_online, "final_online_size": len(online),
             "actor_steps": int(online.project_actor_steps), "completed_updates": completed,
             "utd_ratio": 8, "critic_updates": completed*8})


def actor(out: Path):
    import jax
    from agentlace.data.data_store import QueuedDataStore
    from paper_fidelity_env import make_paper_fidelity_env
    from serl_launcher.utils.launcher import make_sac_agent
    from training.peg_insertion.actor_shutdown import BoundedTrainerClient
    from training.peg_insertion.hil_serl.intervention_adapter_v1 import (
        HILSERLInterventionAdapterV1, execute_intervention_control_step)
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    c, _, expert, _ = validate(check_gpu=False)
    require(jax.default_backend() == "gpu", "actor GPU")
    with portable_peg_runtime(ROOT):
        env = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                      training_reset_randomization=True, randomization_seed=c["seed"]+1)
        oracle = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                         training_reset_randomization=True, randomization_seed=c["seed"]+2)
        obs, _ = env.reset(seed=c["seed"]); oracle.reset(seed=c["seed"])
        agent = make_sac_agent(seed=c["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        rng = jax.random.split(jax.random.PRNGKey(c["seed"]))[1]
        online_store, int_store = QueuedDataStore(4000), QueuedDataStore(4000)
        client = BoundedTrainerClient("hil_actor", "127.0.0.1", configure(),
            data_stores={"actor_env": online_store, "actor_env_intvn": int_store},
            wait_for_server=False, shutdown_timeout_seconds=10,
            shutdown_evidence_path=out / "actor_shutdown.json")
        received = [0]
        def network(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params)); received[0] += 1
        client.recv_network_callback(network)
        provider = ValidatedExactLookaheadProvider(env, oracle, expert,
                                                   c["production_expert"]["support_threshold"])
        adapter = HILSERLInterventionAdapterV1(mode="TRAINING", intervention_enabled=True,
            provider=provider, provenance_path=out / "intervention_transitions.jsonl",
            expert_identity={"sha256": c["production_expert"]["sha256"],
                             "binding": c["production_expert"]["binding"]})
        steps = episode = episode_step = 0
        episodes = []; active_event = None; episode_events = []
        episode_interventions = 0; episode_return = 0.0
        try:
            while not (out / "stop_actor.json").exists():
                paused = False
                for checkpoint in TARGETS:
                    request, ack = out / f"pause_actor_{checkpoint}.request.json", out / f"pause_actor_{checkpoint}.ack.json"
                    if request.exists() and not ack.exists():
                        require(client.update(), "checkpoint upload")
                        dump(ack, {"checkpoint": checkpoint, "steps": steps, "interventions": adapter.intervention_count})
                        while not (out / f"decision_{checkpoint}.json").exists() and not (out / "stop_actor.json").exists(): time.sleep(.05)
                        if (out / f"decision_{checkpoint}.json").exists() and json.loads(
                                (out / f"decision_{checkpoint}.json").read_text())["action"] == "stop":
                            # Do not create transitions between the final pause/upload
                            # and the learner's durable stop marker.
                            while not (out / "stop_actor.json").exists():
                                time.sleep(.01)
                        paused = True; break
                if paused: continue
                require(steps < c["pilot"]["actor_max_steps"], "actor step bound")
                rng, key = jax.random.split(rng)
                policy = np.asarray(jax.device_get(agent.sample_actions(
                    observations=jax.device_put(obs), seed=key, deterministic=False)), np.float32)
                result = execute_intervention_control_step(env=env, online_store=online_store,
                    intervention_store=int_store, adapter=adapter, obs20=obs, policy_action=policy,
                    c3_has_authority=True, episode=episode, step=episode_step)
                if result.decision.intervened:
                    episode_interventions += 1
                    if active_event is None:
                        p = dict(result.decision.provenance)
                        active_event = {"episode": episode, "start_step": episode_step,
                            "end_step": episode_step, "duration": 0,
                            "policy_action_at_trigger": policy.tolist(),
                            "expert_action_at_trigger": result.executed_action.tolist(),
                            "trigger_reason": p.get("trigger_reason"), "state_hash": p.get("state_hash"),
                            "support_metric": {"current_D_DEMO": p.get("current_D_DEMO"),
                                "next_D_DEMO_under_policy": p.get("next_D_DEMO_under_policy"),
                                "threshold": p.get("support_threshold")},
                            "recovery_restored": False, "policy_resumed": False}
                    active_event["end_step"] = episode_step; active_event["duration"] += 1
                elif active_event is not None:
                    active_event.update(recovery_restored=True, policy_resumed=True)
                    episode_events.append(active_event); active_event = None
                obs = result.next_obs; episode_return += float(result.reward)
                steps += 1; episode_step += 1
                # The learner writes the durable stop marker immediately before
                # closing Agentlace.  Do not turn that intentional shutdown race
                # into a failed periodic upload.
                if steps % 10 == 0 and not (out / "stop_actor.json").exists():
                    require(client.update(), "actor upload")
                if result.terminated or result.truncated:
                    success = bool(result.info.get("is_success", result.terminated and not result.truncated))
                    if active_event is not None:
                        active_event.update(recovery_restored=provider.nearest(obs)[0] <= provider.threshold,
                                            policy_resumed=False)
                        episode_events.append(active_event); active_event = None
                    for event in episode_events:
                        event["episode_eventually_succeeded"] = success
                        append(out / "intervention_events.jsonl", event)
                    episodes.append({"episode": episode, "steps": episode_step, "return": episode_return,
                        "success": success, "interventions": episode_interventions,
                        "had_intervention": episode_interventions > 0})
                    append(out / "training_episodes.jsonl", episodes[-1])
                    episode += 1; episode_step = 0; episode_return = 0.0
                    episode_interventions = 0; episode_events = []; provider.active = False
                    obs, _ = env.reset()
        finally:
            try: client.update()
            except Exception: pass
            if active_event is not None:
                active_event.update(recovery_restored=False, policy_resumed=False,
                                    episode_eventually_succeeded=False)
                append(out / "intervention_events.jsonl", active_event)
            dump(out / "actor_result.json", {"status": "PASS", "steps": steps,
                 "episodes_completed": len(episodes), "interventions": adapter.intervention_count,
                 "provider_calls": provider.call_count, "lookaheads": provider.lookahead_count,
                 "received_networks": received[0]})
            shutdown = client.stop(); oracle.close(); env.close()
            dump(out / "actor_cleanup_complete.json", {"status": "PASS", "shutdown": shutdown,
                 "environment_closed": True, "oracle_closed": True})


def verify(out: Path):
    base = load_base()
    # The base verifier is algorithm/checkpoint-semantic code; bind only this branch's identities.
    original = base.load_base if hasattr(base, "load_base") else None
    del original
    # Its hard-coded run id differs, so perform direct fresh-state and replay restoration here.
    import jax
    from paper_fidelity_env import make_paper_fidelity_env
    from serl_launcher.utils.launcher import make_replay_buffer, make_sac_agent
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import restore_replay
    c = cfg(); checks = {}
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                      training_reset_randomization=False)
        fresh = make_sac_agent(seed=c["seed"], sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        for step in TARGETS:
            restored = restore_fresh_state(out / "checkpoints", fresh.state, step)
            base.block_tree(jax, restored)
            leaves = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(restored) if hasattr(x, "dtype")]
            checks[str(step)] = bool(leaves and all(np.isfinite(x).all() for x in leaves))
        replay = make_replay_buffer(env, capacity=c["training"]["replay_capacity"], type="replay_buffer")
        info = restore_replay(replay, out / "checkpoints/checkpoint_500_online_replay.pkl")
        require(info["size"] > 0 and info["actor_steps"] > 0, "restored online replay")
        env.close()
    require(all(checks.values()), "finite checkpoint restores")
    dump(out / "restore_verification.json", {"status": "PASS", "checkpoints": checks,
         "restored_replay": info, "training_resumed": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("preflight", "learner", "actor", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(); out = args.output.resolve()
    require(ROOT in out.parents, "isolated project output")
    if args.role == "preflight": preflight(None); return 0
    {"learner": learner, "actor": actor, "verify": verify}[args.role](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
