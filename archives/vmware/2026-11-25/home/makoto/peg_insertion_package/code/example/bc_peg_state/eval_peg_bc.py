import argparse
import sys

import jax
import numpy as np

from flax.training import checkpoints
from serl_launcher.agents.continuous.bc import BCAgent


# 使用已经验证过的 Peg Adapter
sys.path.insert(
    0,
    "/home/makoto/serl/examples/"
    "async_rlpd_peg_sim",
)

from peg_env_adapter import make_peg_state_env


CONTROL_DT = 0.05


def make_agent(obs, action):
    return BCAgent.create_states(
        rng=jax.random.PRNGKey(0),
        observations=obs,
        actions=action,

        network_kwargs={
            "activations": jax.nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },

        policy_kwargs={
            "tanh_squash_distribution":
                False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },

        learning_rate=3e-4,
    )


def finite(v):
    try:
        x = float(v)
        return np.isfinite(x)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
    )

    parser.add_argument(
        "--step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--print_actions",
        action="store_true",
    )

    args = parser.parse_args()

    render_mode = (
        "human"
        if args.render
        else "rgb_array"
    )

    env = make_peg_state_env(
        render_mode=render_mode,
        max_episode_steps=500,
    )

    obs, _ = env.reset()

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    sample_action = np.asarray(
        env.action_space.sample(),
        dtype=np.float32,
    )

    agent = make_agent(
        obs,
        sample_action,
    )

    print(
        "[BC-EVAL] restoring:",
        args.checkpoint_dir,
        "step=",
        args.step,
    )

    restored_state = (
        checkpoints.restore_checkpoint(
            ckpt_dir=args.checkpoint_dir,
            target=agent.state,
            step=args.step,
        )
    )

    agent = agent.replace(
        state=restored_state,
    )

    print("[OK] BC checkpoint restored")

    episode_results = []

    for episode in range(
        args.episodes
    ):
        obs, _ = env.reset()

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        episode_return = 0.0

        first_position_error = None
        min_position_error = np.inf
        min_lateral_error = np.inf
        min_axial_error = np.inf

        max_force = 0.0

        terminated = False
        truncated = False

        last_info = {}

        saturated_steps = 0

        for t in range(500):
            raw_action = (
                agent.sample_actions(
                    observations=obs,
                    argmax=True,
                )
            )

            raw_action = np.asarray(
                jax.device_get(raw_action),
                dtype=np.float32,
            )

            # BC Gaussian mean 理论上可能越界。
            # 环境动作定义仍然是 [-1, 1]。
            action = np.clip(
                raw_action,
                env.action_space.low,
                env.action_space.high,
            )

            if np.any(
                np.abs(action[:6])
                >= 0.999
            ):
                saturated_steps += 1

            if (
                args.print_actions
                and episode == 0
                and t < 10
            ):
                print(
                    f"[ACTION {t:02d}]",
                    np.round(
                        raw_action,
                        4,
                    ),
                )

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            next_obs = np.asarray(
                next_obs,
                dtype=np.float32,
            )

            episode_return += float(
                reward
            )

            last_info = dict(info)

            pos = info.get(
                "position_error",
                np.nan,
            )

            lat = info.get(
                "lateral_error",
                np.nan,
            )

            axial = info.get(
                "axial_error",
                np.nan,
            )

            force = info.get(
                "force_norm",
                np.nan,
            )

            if finite(pos):
                pos = float(pos)

                if first_position_error is None:
                    first_position_error = pos

                min_position_error = min(
                    min_position_error,
                    pos,
                )

            if finite(lat):
                min_lateral_error = min(
                    min_lateral_error,
                    float(lat),
                )

            if finite(axial):
                min_axial_error = min(
                    min_axial_error,
                    abs(float(axial)),
                )

            if finite(force):
                max_force = max(
                    max_force,
                    float(force),
                )

            obs = next_obs

            if args.render:
                env.render()

            if terminated or truncated:
                break

        steps = t + 1

        success = bool(
            last_info.get(
                "is_success",
                terminated
                and not truncated,
            )
        )

        if (
            first_position_error is not None
            and np.isfinite(
                min_position_error
            )
            and first_position_error > 1e-9
        ):
            approach_reduction = (
                first_position_error
                - min_position_error
            ) / first_position_error
        else:
            approach_reduction = np.nan

        sat_ratio = (
            saturated_steps / steps
        )

        result = {
            "success": success,
            "steps": steps,
            "return": episode_return,
            "first_position_error":
                first_position_error,
            "min_position_error":
                min_position_error,
            "approach_reduction":
                approach_reduction,
            "min_lateral_error":
                min_lateral_error,
            "min_axial_error":
                min_axial_error,
            "max_force":
                max_force,
            "saturation_ratio":
                sat_ratio,
        }

        episode_results.append(
            result
        )

        print()
        print(
            f"[BC-EVAL] episode="
            f"{episode + 1}"
        )

        print(
            f"  success             = "
            f"{success}"
        )

        print(
            f"  steps               = "
            f"{steps}"
        )

        print(
            f"  cycle_time          = "
            f"{steps * CONTROL_DT:.3f}s"
        )

        print(
            f"  return              = "
            f"{episode_return:.3f}"
        )

        print(
            f"  first_position_err  = "
            f"{first_position_error}"
        )

        print(
            f"  min_position_err    = "
            f"{min_position_error}"
        )

        print(
            f"  approach_reduction  = "
            f"{approach_reduction:.1%}"
        )

        print(
            f"  min_lateral_err     = "
            f"{min_lateral_error}"
        )

        print(
            f"  min_axial_err       = "
            f"{min_axial_error}"
        )

        print(
            f"  max_force           = "
            f"{max_force:.3f}"
        )

        print(
            f"  action_saturation   = "
            f"{sat_ratio:.1%}"
        )

    successes = sum(
        int(r["success"])
        for r in episode_results
    )

    print()
    print("=" * 70)
    print("[BC EVAL SUMMARY]")
    print(
        f"Success: "
        f"{successes}/"
        f"{len(episode_results)}"
    )
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
