# Object Grasp Demo Quality Summary

Date: 2026-08-22

## What is measurable

The frozen file contains 2,000 finite transitions with valid replay shapes and action bounds.

| Metric | Result |
|---|---:|
| Reward min / max / mean / median | 0.001252706 / 0.995075345 / 0.593166131 / 0.847178966 |
| Action norm min / max / mean / median | 0.371015072 / 1.985642791 / 1.381497035 / 1.397868991 |
| Mean absolute action x/y/z/grasp | 0.646528 / 0.649548 / 0.590738 / 0.676197 |
| Exact zero-action count / ratio | 0 / 0.0 |
| NaN/Inf | 0 |

No trajectory was deleted or rewritten.

## What is not measurable per claimed trajectory

Although the README calls the file 20 trajectories, it is a flat transition list with no true `done`, mask-zero, trajectory ID, success, or block-state record. The repository therefore cannot truthfully produce per-trajectory episode length, return, success, final block height, or lift height. It also cannot validate that every claimed trajectory covers the current randomized reset distribution.

```text
CLAIMED_TRAJECTORIES = 20
RECOVERABLE_EPISODES = 0
PER_EPISODE_QUALITY = UNVERIFIED
```

## Initial-state audit

Twenty real EGL resets found:

- TCP position fixed at approximately `[0.307796, 0, 0.444215]`;
- block Z fixed at 0.02 m;
- block X ranged 0.266302–0.517496 m;
- block Y ranged -0.228343–0.227488 m;
- front/wrist initial images varied with the randomized scene;
- gripper value remained scalar-shaped.

```text
OBJECT_INITIAL_BLOCK_STATE = RANDOMIZED_XY
OBJECT_INITIAL_TCP_STATE = FIXED
OBJECT_CAMERA_INITIAL_STATE = RANDOMIZED_SCENE_DEPENDENT
```

Raw reset records are in `results/object_m1d_b/reset_audit.json`.
