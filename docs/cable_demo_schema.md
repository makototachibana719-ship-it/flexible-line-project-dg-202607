# Cable Routing Demo Schema

Date: 2026-08-21  
Consumer: Cable V2 DrQ pickle loader and memory-efficient replay buffer

## Consumer evidence

The Cable actor constructs each transition with exactly:

```python
{
    "observations": obs,
    "actions": actions,
    "next_observations": next_obs,
    "rewards": np.asarray(reward, dtype=np.float32),
    "masks": 1.0 - terminated,
    "dones": terminated or truncated,
}
```

The learner calls `pickle.load()`, iterates the top-level object, and inserts each element into a memory-efficient replay buffer. `populate_data_store()` uses the same list-of-transition behavior. Replay insertion recursively requires observation keys to exactly match its observation space.

## Wrapper-after structure

Native V2 images do not remain nested in recorded learner-facing transitions. The project observation wrapper produces top-level keys and horizon-one chunking adds the leading time dimension:

| Key | Shape | Dtype |
|---|---:|---|
| `state` | `(1,7)` | float32 |
| `front` | `(1,128,128,3)` | uint8 |
| `wrist` | `(1,128,128,3)` | uint8 |

The seven-value state is flattened with Gym's `flatten(state_space, state_observation)` operation—the same operation used by SERL's wrapper family—rather than by an independently assumed concatenation order.

The memory-efficient replay buffer is configured with `image_keys=["front", "wrist"]`. It requires each image input to have `(T,H,W,C)` form and internally stores/reconstructs image frames; `state` remains a normal top-level observation.

## Transition fields

| Field | Schema |
|---|---|
| `observations` | dict containing exactly `state`, `front`, `wrist` |
| `actions` | `(3,) float32`, finite and within `[-1,1]` |
| `next_observations` | same schema as observations |
| `rewards` | scalar float32 |
| `masks` | scalar float32; `1 - terminated` |
| `dones` | scalar bool; `terminated or truncated` |

All arrays are deep-copied/detached before storage.

## Termination semantics

- Ordinary transition: `dones=False`, `masks=1.0`.
- Successful terminal transition: `terminated=True`, therefore `dones=True`, `masks=0.0`.
- Time-limit transition would have `dones=True`, `masks=1.0`, but timeout episodes are excluded from formal Cable demos.
- Formal episode success is never inferred from reward; it requires `info["success"] is True`.

## Dataset and metadata

The pickle top level is a non-empty Python list of six-field transitions. It deliberately contains no `episode_id` or `info` fields because the current replay buffer asserts exact recursive keys.

The companion JSON contains:

- `episode_count`, `transition_count`, `success_count`;
- observation/action shapes and dtypes;
- environment/runtime/creation time;
- episode ID, `[start_index,end_index_exclusive)` transition range, steps, return, success, duration, and final distance;
- `success_only_policy` and `test_only` markers.

For a formal dataset, episode ranges must be contiguous, non-empty, cover every transition exactly once, and contain only successful episodes with a final terminal transition.

## Current compatibility boundary

The generated six-field pickle is accepted by the actual SERL population helper and memory-efficient replay implementation when constructed with the project wrapper's `state/front/wrist` observation space. The delivered training script's direct use of the current singular-`image` `SERLObsWrapper` remains incompatible before loader execution and requires a later project-side integration change; no SERL source was modified in this milestone.
