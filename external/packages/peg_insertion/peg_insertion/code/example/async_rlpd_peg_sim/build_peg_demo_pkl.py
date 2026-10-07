from pathlib import Path
import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo-dir",
        default=str(
            Path.home()
            / "serl_data"
            / "peg_demos_v1"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            Path.home()
            / "serl_data"
            / "peg_insert_20_demos_state.pkl"
        ),
    )

    args = parser.parse_args()

    demo_dir = Path(
        args.demo_dir
    ).expanduser()

    output = Path(
        args.output
    ).expanduser()

    files = sorted(
        demo_dir.glob("demo_*.npz")
    )

    if len(files) != 20:
        raise RuntimeError(
            f"预计 20 条 Demo，实际找到 {len(files)}"
        )

    transitions = []
    episode_lengths = []

    for path in files:
        data = np.load(
            path,
            allow_pickle=False,
        )

        obs = np.asarray(
            data["observations"],
            dtype=np.float32,
        )

        actions = np.asarray(
            data["actions"],
            dtype=np.float32,
        )

        next_obs = np.asarray(
            data["next_observations"],
            dtype=np.float32,
        )

        rewards = np.asarray(
            data["rewards"],
            dtype=np.float32,
        )

        masks = np.asarray(
            data["masks"],
            dtype=np.float32,
        )

        dones = np.asarray(
            data["dones"],
            dtype=np.bool_,
        )

        n = len(actions)

        assert obs.shape == (n, 20)
        assert next_obs.shape == (n, 20)
        assert actions.shape == (n, 7)
        assert rewards.shape == (n,)
        assert masks.shape == (n,)
        assert dones.shape == (n,)

        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(next_obs))
        assert np.all(np.isfinite(actions))
        assert np.all(np.isfinite(rewards))

        assert np.max(np.abs(actions)) <= 1.0001

        for i in range(n):
            transitions.append(
                {
                    "observations": obs[i].copy(),
                    "actions": actions[i].copy(),
                    "next_observations": (
                        next_obs[i].copy()
                    ),
                    "rewards": np.float32(
                        rewards[i]
                    ),
                    "masks": np.float32(
                        masks[i]
                    ),
                    "dones": np.bool_(
                        dones[i]
                    ),
                }
            )

        episode_lengths.append(n)

        print(
            f"[OK] {path.name}: "
            f"{n} transitions"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("wb") as f:
        pickle.dump(
            transitions,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print("=" * 60)
    print("Demo episodes :", len(files))
    print(
        "Transitions   :",
        len(transitions),
    )
    print(
        "Median length :",
        float(np.median(episode_lengths)),
    )
    print("Output        :", output)

    assert len(transitions) == 5512, (
        "transition 总数与本轮质量检查不一致"
    )

    print(
        "[OK] 20 条 Demo 已转换为 SERL transition PKL"
    )


if __name__ == "__main__":
    main()
