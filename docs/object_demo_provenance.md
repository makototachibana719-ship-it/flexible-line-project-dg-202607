# Object Demo Provenance Reconstruction

Date: 2026-08-22  
Milestone: `M1D-B2`

## Search scope

The complete delivered Object package was searched for the Demo filename and all generation/recording/trajectory/pickle/success terms across source, scripts, README, manifest, configuration, diffs, and logs. The matching local SERL documentation and `async_drq_sim` consumer were also inspected.

## Generator result

```text
DEMO_GENERATOR_FOUND = NO
```

The package contains:

- the frozen pickle and an identical example-directory copy;
- `tmux_rlpd_launch.sh`, which downloads the file from the official SERL GitHub release;
- `async_drq_sim.py`, which only loads and inserts it;
- `scripts/inspect_demo.py`, which only reads it;
- `scripts/record_demo.sh`, which explicitly says the local recording workflow is not confirmed;
- README/demo-info claims that it is an official public 20-trajectory Demo.

There is no Object-specific collection loop, episode loop, `pickle.dump`, success-only filter, retry-until-success logic, or trajectory counter in the delivered package. The package builder copies the already-existing file from the submitter's SERL example directory; it does not generate it.

## 20 × 100 hypothesis

The following evidence aligns exactly:

- actual flat list length: 2,000 transitions;
- official SERL documentation: “20 demo trajectories”;
- package README/demo-info: 20 demonstrations;
- registered `PandaPickCubeVision-v0` outer limit: 100 steps;
- arithmetic: 20 × 100 = 2,000.

This is strong circumstantial provenance, but no generator establishes that each stored trajectory always contributed exactly 100 transitions. The actual file has no done or boundary markers. Therefore:

```text
DEMO_BOUNDARY_RECONSTRUCTION = STRONGLY_SUPPORTED
BOUNDARY_RECONSTRUCTION_AUTHORIZED = NO
```

The file was not split into 20 episodes, and no hypothetical per-episode statistics were promoted to evidence.

## Success provenance

No direct success/block/lift data exists in the pickle, and no generator guarantee was found. “Demo” and “20 trajectories” do not prove that failed attempts were discarded. There is no evidence of success counter, retry-until-success, human acceptance gate, or success-only persistence.

```text
SUCCESS_PROVENANCE = UNVERIFIED
SUCCESS_VALIDATION_METHOD = UNAVAILABLE_NO_DIRECT_DATA_OR_GENERATOR_GUARANTEE
```

The earlier schema/replay validation remains valid, but success remains unvalidated. If authoritative upstream generator/sidecar provenance cannot be obtained, a future task may propose newly recorded success-aware Demos; this round did not record anything.

Raw evidence: `results/object_m1d_b2/provenance_audit.json`.
