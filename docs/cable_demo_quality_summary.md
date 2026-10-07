# Cable Demo Quality Summary

Date: 2026-08-21  
Validated formal episodes: 5  
Total transitions: 10,521  
Zero-action tolerance: `all(abs(action) < 1e-8)`

## Per-Episode Quality

| Demo timestamp | Steps | Duration (s) | Return | Final distance | Zero actions | Zero ratio | Success | Terminated | Truncated |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 07-14-17 | 2,629 | 242.230976 | 850.830242 | 0.0298518794 | 2,119 | 80.6010% | True | True | False |
| 07-37-47 | 4,879 | 426.199972 | 838.617493 | 0.0297792268 | 4,564 | 93.5438% | True | True | False |
| 07-43-30 | 1,264 | 111.574474 | 418.196112 | 0.0283143827 | 799 | 63.2120% | True | True | False |
| 07-46-57 | 866 | 74.584904 | 234.841717 | 0.0299206823 | 446 | 51.5012% | True | True | False |
| 07-50-43 | 883 | 73.636716 | 245.221111 | 0.0296784192 | 515 | 58.3239% | True | True | False |

All episodes use `dataset_boundary_reason="info_success"`, and all final transitions have `done=True`, `mask=0.0`, `terminated=True`, and `truncated=False`.

## Aggregate Statistics

| Quantity | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|
| Steps | 866 | 4,879 | 2,104.2 | 1,264 |
| Duration (s) | 73.636716 | 426.199972 | 185.645408 | 111.574474 |
| Return | 234.841717 | 850.830242 | 517.541335 | 418.196112 |
| Final distance to slot | 0.0283143827 | 0.0299206823 | 0.0295089181 | 0.0297792268 |
| Zero-action count | 446 | 4,564 | 1,688.6 | 799 |
| Per-episode zero-action ratio | 51.5012% | 93.5438% | 69.4364% | 63.2120% |

Across the concatenated 10,521 transitions:

```text
TOTAL_ZERO_ACTION_TRANSITIONS = 8443
OVERALL_ZERO_ACTION_RATIO = 80.2490%
```

Zero-action transitions were retained. A high zero-action ratio is a quality characteristic for later training/ablation decisions, not grounds for silently deleting or rewriting authentic demonstrations.

## Interpretation

- All five episodes are structurally valid, successful, and terminally consistent.
- Episode lengths and durations vary substantially despite the fixed reset, showing operator trajectory/timing variation.
- Returns are not used as the success criterion; success comes from metadata/environment `info_success` semantics.
- Final distances all satisfy the environment's `< 0.03 m` success condition.
- Zero-action occupancy is high, particularly in the longest episode. Baseline training should preserve provenance and report whether idle-transition filtering or weighting is used in any later ablation; this milestone applies neither.
- Initial state is fixed for recorder, training, and evaluation. These demos are in-distribution for the current baseline evaluation, while OOD generalization risk remains high.
