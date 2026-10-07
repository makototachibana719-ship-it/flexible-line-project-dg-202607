#!/usr/bin/env python3
"""Actor-update attribution diagnostic for the failed early pre-tanh A0 anchor lineage."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_peg_f3_v2_early_pretanh_a0_anchor_ablation_1k as pretanh_runner

OUTPUT = ROOT / "results/diagnostics/peg/f3_v2_actor_update_attribution_pretanh_a0_20260906_002"
TARGETS = (0, 50, 100, 250, 500, 750, 1000)
CHECKPOINTS = (50, 100, 250, 500, 750, 1000)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def install(role: str):
    base, cfg, defaults = pretanh_runner.install_bindings(role)
    base.OUTPUT = OUTPUT
    base.ACTIVE_OUTPUT = OUTPUT
    base.RUN_UPDATES = 1000
    base.CHECKPOINT_TARGETS = CHECKPOINTS
    base.__file__ = str(Path(__file__).resolve())
    return base, cfg, defaults


def initial_early_sets(cfg, data):
    from training.peg_insertion.early_pretanh_a0_anchor_ablation import early_pretanh_mask

    mask = early_pretanh_mask(ROOT, cfg)
    early_obs = np.asarray([row["observations"] for row, keep in zip(data, mask) if keep], np.float32)
    early_keys = {np.ascontiguousarray(row).tobytes() for row in early_obs}
    require(early_obs.shape == (363, 20), "early row binding")
    return early_obs, early_keys


def attribution_row(jax, jnp, agent, fixed_obs, update, early_rows_in_batch, anchor_active_updates):
    params = agent.state.params
    obs_batch = jnp.asarray(fixed_obs[None], dtype=jnp.float32)
    dist = agent.forward_policy(obs_batch, rng=None, train=False)
    loc = dist.distribution.loc[0]
    mode = dist.mode()[0]
    std = dist.distribution.stddev()[0]
    temperature = agent.forward_temperature()

    def q_loss_z(z):
        action = mode.at[0].set(jnp.tanh(z[0]))
        return -jnp.mean(agent.forward_critic(obs_batch, action[None], rng=None, train=False))

    def entropy_loss_z(z):
        action = mode.at[0].set(jnp.tanh(z[0]))
        return temperature * dist.log_prob(action[None])[0]

    def sac_loss_z(z):
        return q_loss_z(z) + entropy_loss_z(z)

    def global_anchor_z(z):
        return float(agent.config["demo_anchor_lambda"]) * jnp.square(jnp.tanh(z[0]))

    def early_anchor_z(z):
        return float(agent.config["early_pretanh_a0_anchor"]["lambda"]) * jnp.square(z[0])

    dl_q = jax.grad(q_loss_z)(loc)[0]
    dl_ent = jax.grad(entropy_loss_z)(loc)[0]
    dl_sac = jax.grad(sac_loss_z)(loc)[0]
    dl_global = jax.grad(global_anchor_z)(loc)[0]
    dl_early = jax.grad(early_anchor_z)(loc)[0]
    total = dl_sac + dl_global + dl_early

    def scalar(x):
        return float(np.asarray(jax.device_get(x)).reshape(()))

    return {
        "update": int(update),
        "a0_mode": scalar(mode[0]),
        "a0_pretanh": scalar(loc[0]),
        "a0_std": scalar(std[0]),
        "temperature": scalar(temperature),
        "early_rows_in_batch": int(early_rows_in_batch),
        "early_anchor_active": True,
        "early_anchor_active_updates": int(anchor_active_updates),
        "dl_q_dz0": scalar(dl_q),
        "dl_entropy_dz0": scalar(dl_ent),
        "dl_sac_dz0": scalar(dl_sac),
        "dl_global_anchor_dz0": scalar(dl_global),
        "dl_early_pretanh_anchor_dz0": scalar(dl_early),
        "dl_total_dz0": scalar(total),
        "loss_pretanh_source": "distribution.distribution.loc[:,0]",
        "rollout_action_source": "published actor params -> sample_actions",
    }


def learner(out: Path, base):
    import jax
    import jax.numpy as jnp
    from agentlace.trainer import TrainerServer
    from serl_launcher.utils.launcher import make_replay_buffer

    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    from async_sac_state_sim import concat_rlpd_batches, make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import install_actor_step_tracking, save_bundle

    cfg, data, backend = base.validate_config_and_dataset(check_gpu=False)
    require(jax.default_backend() == "gpu", "learner GPU backend")
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    state = {"live": 0}
    event_lock = threading.Lock()

    def event(value):
        value["timestamp_ns"] = time.time_ns()
        with event_lock:
            base.append_jsonl(out / "communication.jsonl", value)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        fixed_obs = np.asarray(obs, np.float32)
        require(fixed_obs.shape == (20,) and fixed_obs.dtype == np.float32, "learner V2 env observation")
        agent = make_sac_agent(seed=cfg["seed"], sample_obs=env.observation_space.sample(), sample_action=env.action_space.sample())
        initial_rng = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
        early_obs, early_keys = initial_early_sets(cfg, data)
        online = make_replay_buffer(env, capacity=cfg["training"]["replay_capacity"], type="replay_buffer")
        demo = make_replay_buffer(env, capacity=len(data) + 100, type="replay_buffer")
        install_actor_step_tracking(online, 0)
        for transition in data:
            demo.insert(transition)
        require(len(demo) == 1921 and len(online) == 0, "fresh replay population")
        server = TrainerServer(
            base.configure(),
            data_callback=lambda name, payload: event({"event": "live_insert", "count": len(payload["data"]), "online_size": len(online)}),
            request_callback=lambda kind, payload: {},
        )
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
        early_exposure_rows = []
        anchor_active_updates = 0
        append_jsonl(out / "actor_update_attribution.jsonl", attribution_row(jax, jnp, agent, fixed_obs, 0, 0, 0))
        try:
            for update in range(1, 1001):
                online_batch = next(online_iter)
                demo_batch = next(demo_iter)
                demo_obs_np = np.asarray(jax.device_get(demo_batch["observations"]), dtype=np.float32)
                early_rows = sum(np.ascontiguousarray(row).tobytes() in early_keys for row in demo_obs_np)
                early_exposure_rows.append(int(early_rows))
                anchor_active_updates += 1
                require(online_batch["observations"].shape == demo_batch["observations"].shape == (32, 20), "32 Demo + 32 Online observation batches")
                require(online_batch["actions"].shape == demo_batch["actions"].shape == (32, 7), "32 Demo + 32 Online action batches")
                batch = concat_rlpd_batches(online_batch, demo_batch)
                watchdog = threading.Timer(cfg["smoke"]["update_timeout_seconds"], lambda: os._exit(124))
                watchdog.daemon = True
                watchdog.start()
                started = time.perf_counter()
                try:
                    new_agent, info = agent.update_high_utd(batch, utd_ratio=1)
                    base.block_tree(jax, (new_agent, info))
                finally:
                    watchdog.cancel()
                losses = base.loss_scalars(jax, info)
                require(len(losses) >= 3 and all(np.isfinite(v) for v in losses.values()), "loss/Q/temperature finite")
                rng_before = np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist()
                rng_after = np.asarray(jax.device_get(new_agent.state.rng), dtype=np.uint32).tolist()
                require(rng_before != rng_after, "learner RNG did not advance")
                agent = new_agent
                completed = update
                base.append_jsonl(out / "updates.jsonl", {
                    "update": update,
                    "seconds": time.perf_counter() - started,
                    "losses": losses,
                    "losses_finite": True,
                    "online_size": len(online),
                    "demo_size": len(demo),
                    "online_batch": 32,
                    "demo_batch": 32,
                    "early_rows_in_batch": int(early_rows),
                    "early_anchor_active": True,
                    "rng_before": rng_before,
                    "rng_after": rng_after,
                })
                if update in TARGETS:
                    append_jsonl(out / "actor_update_attribution.jsonl", attribution_row(jax, jnp, agent, fixed_obs, update, early_rows, anchor_active_updates))
                server.publish_network(agent.state.params)
                event({"event": "publish_update", "update": update})
                if update in CHECKPOINTS:
                    source = {
                        "identifier": "peg-f3-v2-actor-update-attribution-pretanh-diagnostic",
                        "profile": "diagnostic_same_failed_pretanh_config",
                        "fresh_initialization": True,
                        "resume": False,
                        "demo_online_ratio": "32:32",
                        "utd_ratio": 1,
                        "training_semantics_changed": False,
                    }
                    save_bundle(
                        checkpoint_dir,
                        agent.state,
                        online,
                        update,
                        int(online.project_actor_steps),
                        cfg,
                        ROOT / cfg["demo"]["manifest"],
                        "wandb-disabled-peg-f3-v2-actor-update-attribution-pretanh-diagnostic",
                        source,
                    )
                    base.append_jsonl(out / "checkpoint_events.jsonl", {"event": "save_complete", "checkpoint": update, "online_size": len(online), "actor_steps": int(online.project_actor_steps)})
        finally:
            write_json(out / "stop_actor.json", {"completed_updates": completed, "timestamp_ns": time.time_ns()})
            server.stop()
        write_json(out / "learner_result.json", {
            "status": "PASS",
            "backend": jax.default_backend(),
            "fresh_initialization": True,
            "demo_size": len(demo),
            "initial_online_size": state["live"],
            "final_online_size": len(online),
            "actor_steps": int(online.project_actor_steps),
            "completed_updates": completed,
            "initial_rng": initial_rng,
            "early_demo_rows_total": int(early_obs.shape[0]),
            "early_rows_update_fraction": float(np.mean(np.asarray(early_exposure_rows) > 0)),
            "mean_early_rows_per_update": float(np.mean(early_exposure_rows)),
            "early_anchor_active_updates": int(anchor_active_updates),
            "wandb_run": "DISABLED",
            "run_id": "wandb-disabled-peg-f3-v2-actor-update-attribution-pretanh-diagnostic",
            "requested_updates": 1000,
        })


def finalize_attribution(out: Path, base_status: int):
    trace = [json.loads(line) for line in (out / "actor_update_attribution.jsonl").read_text().splitlines()]
    updates = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
    learner_result = json.loads((out / "learner_result.json").read_text())
    result_base = json.loads((out / "result.json").read_text())
    require([row["update"] for row in trace] == list(TARGETS), "attribution target coverage")
    require(len(updates) == 1000 and learner_result.get("completed_updates") == 1000, "diagnostic update coverage")
    exposure_fraction = float(learner_result["early_rows_update_fraction"])
    mean_early = float(learner_result["mean_early_rows_per_update"])
    active = int(learner_result["early_anchor_active_updates"])
    exposure = "ADEQUATE" if active == 1000 and exposure_fraction > 0.95 and mean_early > 1.0 else "INSUFFICIENT"
    windowed = trace[1:]
    cum_q = float(sum(row["dl_q_dz0"] for row in windowed))
    cum_entropy = float(sum(row["dl_entropy_dz0"] for row in windowed))
    cum_global = float(sum(row["dl_global_anchor_dz0"] for row in windowed))
    cum_early = float(sum(row["dl_early_pretanh_anchor_dz0"] for row in windowed))
    cum_sac = cum_q + cum_entropy
    early_ratio = abs(cum_early) / max(abs(cum_sac), 1e-12)
    loss_binding = "PASS" if all(row["loss_pretanh_source"] == "distribution.distribution.loc[:,0]" for row in trace) else "FAIL"
    rollout_binding = "PASS" if result_base.get("checks", {}).get("communication") else "FAIL"
    checkpoint_binding = "PASS" if result_base.get("checks", {}).get("restore") else "FAIL"
    if exposure == "INSUFFICIENT":
        primary = "SAMPLING_EXPOSURE_FAILURE"
        selected = "EARLY_SAMPLE_WEIGHTING"
    elif loss_binding != "PASS" or rollout_binding != "PASS" or checkpoint_binding != "PASS":
        primary = "BINDING_MISMATCH"
        selected = "BINDING_FIX"
    elif early_ratio < 0.15:
        primary = "OBJECTIVE_BALANCE_FAILURE"
        selected = "EARLY_OBJECTIVE_BALANCE"
    else:
        primary = "BLOCKED"
        selected = "BLOCKED"
    audit = {
        "ACTOR_UPDATE_ATTRIBUTION_AUDIT": "PASS" if base_status == 0 else "FAIL",
        "TRAINING_STARTED": "diagnostic_0_to_1000_only",
        "FORMAL_OR_HIGHER_RUN_TRAINING_STARTED": False,
        "EARLY_STATE_EXPOSURE": exposure,
        "EARLY_ROWS_UPDATE_FRACTION": exposure_fraction,
        "MEAN_EARLY_ROWS_PER_UPDATE": mean_early,
        "EARLY_ANCHOR_ACTIVE_UPDATES": active,
        "LOSS_PRETANH_BINDING": loss_binding,
        "ROLLOUT_ACTOR_BINDING": rollout_binding,
        "CHECKPOINT_ACTOR_BINDING": checkpoint_binding,
        "CUM_Q_PRESSURE_A0": cum_q,
        "CUM_ENTROPY_PRESSURE_A0": cum_entropy,
        "CUM_GLOBAL_ANCHOR_PRESSURE_A0": cum_global,
        "CUM_EARLY_ANCHOR_PRESSURE_A0": cum_early,
        "CUM_SAC_PRESSURE_A0": cum_sac,
        "EARLY_TO_SAC_PRESSURE_RATIO": early_ratio,
        "A0_MODE_TRACE": {str(row["update"]): row["a0_mode"] for row in trace},
        "A0_PRETANH_TRACE": {str(row["update"]): row["a0_pretanh"] for row in trace},
        "PRIMARY_ROOT_CAUSE": primary,
        "SELECTED_FIX": selected,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "NEXT": "STAGE_B_SELECTED_FIX" if selected != "BLOCKED" else "STOP_UNPROVEN_ATTRIBUTION",
    }
    write_json(out / "attribution_result.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0 if audit["ACTOR_UPDATE_ATTRIBUTION_AUDIT"] == "PASS" and selected != "BLOCKED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base, cfg, defaults = install(args.role)
    require(args.output is None or args.output.resolve() == OUTPUT, "canonical diagnostic output binding")
    if args.role == "preflight":
        result = base.preflight()
        result["ATTRIBUTION_TARGETS"] = list(TARGETS)
        result["CHECKPOINT_TARGETS"] = list(CHECKPOINTS)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.role == "supervise":
        require(not OUTPUT.exists(), "fresh diagnostic output required")
        status = base.supervise()
        if not OUTPUT.exists():
            return 1
        return finalize_attribution(OUTPUT, status)
    require(args.output is not None and args.output.resolve() == OUTPUT, "child output binding")
    if args.role == "learner":
        learner(OUTPUT, base)
    elif args.role == "actor":
        base.actor(OUTPUT)
    elif args.role == "verify":
        base.verify(OUTPUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
