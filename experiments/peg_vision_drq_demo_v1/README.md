# Peg Vision DrQ + Demo Replay V1

This directory is physically isolated from the production Peg, Object, and Cable sources.

- Frozen Peg lifecycle: `AuboPegInsert-State-v0` through the existing `PegStateSERLEnv`.
- Observation: Obs20 plus `front` and `wrist` RGB images.
- Action: unchanged normalized Aubo Action7.
- Object contract: `SERLObsWrapper` followed by `ChunkingWrapper(obs_horizon=1, act_exec_horizon=None)`.
- Front camera: existing `frontview`.
- Wrist camera: one experiment-local staged-XML camera, `peg_wrist_cam_v1`.
- No production XML or Python source is edited.

## Final V1 status

Compatibility, camera, five-episode visual Demo, Object replay/AgentLace transport and GPU inference pass. The unchanged Object batch-256 recipe did not complete its first critic update within two bounded 15-minute probes, including the established non-Triton XLA runtime flag. `DRQ_SMOKE=FAIL`; zero optimizer updates and no checkpoint or scientific evaluation were produced. Do not start 500+ training until `VISION_DRQ_FAILURE_REVIEW` resolves this GPU compile gate.

Authoritative report: `results/evaluation/peg_vision_drq_demo_v1_20260910_190000/MASTER_REPORT.md`.
