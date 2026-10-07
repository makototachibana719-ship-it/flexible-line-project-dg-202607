# Peg Demo Quality Summary (M1D-D)

Date: 2026-08-23

This is dataset characterization, not policy evaluation.

| Metric across 20 episodes | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| episode length | 275.600 | 79.777 | 169 | 453 |
| recorded return | -1146.399 | 248.149 | -1665.626 | -749.820 |
| final lateral error (m) | 0.001637 | 0.000316 | 0.000601 | 0.001919 |
| final axial error (m) | 0.001912 | 0.000709 | 0.000475 | 0.002936 |
| final orientation error (rad) | 0.014130 | 0.003033 | 0.010826 | 0.020482 |
| per-episode mean force norm | 19.791 | 10.042 | 10.545 | 36.471 |
| per-episode max force norm | 55.156 | 32.836 | 12.884 | 86.603 |
| mean action L2 | 1.291 | 0.073 | 1.164 | 1.414 |

All episodes have zero-action ratio `0.0` at an all-seven-components tolerance of `1e-8`; every episode's maximum action L2 is `sqrt(2)`. All 20 are successful under current source semantics. High/saturated action magnitude and repeated force maxima are observations to monitor, not grounds for deleting data.

## Initial-state audit

Twenty resets with seeds 0–19 through `serl_mujoco`, EGL and the project staging adapter produced exactly one unique 20D observation and zero per-component span. TCP, relative target and goal quaternion were identical. Source defaults call `set_initial_state` without `options.random=True`; hole and camera scene geometry are fixed. Peg pose is coupled to the fixed robot/gripper assembly and is not separately exposed in the state vector.

`PEG_INITIAL_STATE = FIXED`  
This establishes a fixed-reset baseline and a generalization/diversity risk; it does not invalidate the supplied successful demos.

## Formal evaluation gap

An eval loop exists, but no formal result artifact exists. Later evaluation must record at least: episodes, success count/rate, return mean/std, episode-length mean/std, final lateral/axial/orientation and position error, force mean/max, termination/truncation, execution time, and failure categories (`TIMEOUT`, `LATERAL_MISS`, `AXIAL_MISS`, `ORIENTATION_MISS`, `NONFINITE`, `OTHER`). The source's cycle-time comment/formula is inconsistent (`control_dt=0.02` but multiplies steps by 0.05) and must be corrected project-side before formal reporting.

