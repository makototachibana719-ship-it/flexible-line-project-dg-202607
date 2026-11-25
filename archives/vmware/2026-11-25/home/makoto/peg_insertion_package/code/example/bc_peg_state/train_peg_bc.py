import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from flax.training import checkpoints
from serl_launcher.agents.continuous.bc import BCAgent


def make_agent(obs, action, seed):
    if not hasattr(BCAgent, "create_states"):
        raise RuntimeError(
            "当前 BCAgent 没有 create_states()。"
            "请先确认 State-only BC patch 已应用。"
        )

    return BCAgent.create_states(
        rng=jax.random.PRNGKey(seed),
        observations=obs,
        actions=action,

        network_kwargs={
            "activations": jax.nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },

        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },

        learning_rate=3e-4,
    )


def scalarize(info):
    result = {}

    for key, value in info.items():
        try:
            arr = np.asarray(
                jax.device_get(value)
            )

            if arr.size == 1:
                result[key] = float(
                    arr.reshape(-1)[0]
                )
        except Exception:
            pass

    return result


def deterministic_mse(
    agent,
    observations,
    expert_actions,
    max_samples=512,
):
    n = min(
        len(observations),
        max_samples,
    )

    obs = observations[:n]
    target = expert_actions[:n]

    pred = agent.sample_actions(
        observations=obs,
        argmax=True,
    )

    pred = np.asarray(
        jax.device_get(pred),
        dtype=np.float32,
    )

    mse = float(
        np.mean(
            (pred - target) ** 2
        )
    )

    return mse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo_path",
        default=(
            "/home/makoto/serl_data/"
            "peg_insert_bc_consistent.pkl"
        ),
    )

    parser.add_argument(
        "--checkpoint_path",
        default=(
            "/home/makoto/serl_checkpoints/"
            "peg_insert_bc_consistent_v1"
        ),
    )

    parser.add_argument(
        "--log_path",
        default=(
            "/home/makoto/serl_results/"
            "peg_insert_bc_consistent_v1/"
            "bc_train_log.jsonl"
        ),
    )

    parser.add_argument(
        "--train_steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # ------------------------------------------------
    # Demo
    # ------------------------------------------------

    with open(args.demo_path, "rb") as f:
        demos = pickle.load(f)

    observations = np.stack([
        np.asarray(
            t["observations"],
            dtype=np.float32,
        )
        for t in demos
    ])

    actions = np.stack([
        np.asarray(
            t["actions"],
            dtype=np.float32,
        )
        for t in demos
    ])

    next_observations = np.stack([
        np.asarray(
            t["next_observations"],
            dtype=np.float32,
        )
        for t in demos
    ])

    n = len(demos)

    print("=" * 70)
    print("[BC] transitions :", n)
    print("[BC] obs         :", observations.shape)
    print("[BC] actions     :", actions.shape)
    print("[BC] batch size  :", args.batch_size)
    print("[BC] train steps :", args.train_steps)
    print("[BC] save every  :", args.save_every)
    print("=" * 70)

    assert observations.shape[1:] == (20,)
    assert actions.shape[1:] == (7,)

    # ------------------------------------------------
    # Agent
    # ------------------------------------------------

    agent = make_agent(
        observations[0],
        actions[0],
        args.seed,
    )

    initial_mse = deterministic_mse(
        agent,
        observations,
        actions,
    )

    print(
        "[BC] initial deterministic MSE:",
        initial_mse,
    )

    # ------------------------------------------------
    # Output
    # ------------------------------------------------

    checkpoint_path = Path(
        args.checkpoint_path
    )

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = Path(args.log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if log_path.exists():
        log_path.unlink()

    rng = np.random.default_rng(
        args.seed
    )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    for step in range(
        1,
        args.train_steps + 1,
    ):
        indices = rng.integers(
            0,
            n,
            size=args.batch_size,
        )

        batch = {
            "observations":
                observations[indices],

            "actions":
                actions[indices],

            "next_observations":
                next_observations[indices],
        }

        agent, info = agent.update(
            batch
        )

        if (
            step == 1
            or step % args.log_every == 0
        ):
            metrics = scalarize(info)

            det_mse = deterministic_mse(
                agent,
                observations,
                actions,
            )

            record = {
                "step": step,
                "deterministic_mse":
                    det_mse,
                **metrics,
            }

            print(
                f"[BC] step={step:5d}",
                f"det_mse={det_mse:.6f}",
                metrics,
            )

            with log_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if (
            step % args.save_every == 0
            or step == args.train_steps
        ):
            saved = checkpoints.save_checkpoint(
                ckpt_dir=str(
                    checkpoint_path
                ),
                target=agent.state,
                step=step,
                overwrite=True,
                keep=20,
            )

            print(
                "[BC] checkpoint saved:",
                saved,
            )

    final_mse = deterministic_mse(
        agent,
        observations,
        actions,
    )

    print()
    print("=" * 70)
    print("[BC TRAINING COMPLETE]")
    print("Initial MSE:", initial_mse)
    print("Final MSE  :", final_mse)
    print(
        "Checkpoint :",
        checkpoint_path,
    )
    print(
        "Log        :",
        log_path,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
