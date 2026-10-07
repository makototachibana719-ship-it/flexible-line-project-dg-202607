# M1D-B4 Report

Date: 2026-08-23

The producing runtime was reproduced exactly in an isolated retained environment, while original `serl` remained unchanged. Storage diagnostics and individual leaves pass under TensorStore 0.1.78, but Orbax metadata, no-target restore, and matching-target restore still stall. No restored state was returned and policy inference was correctly skipped.

- `RUNTIME_REPRODUCTION = PASS`
- `ORIGINAL_SERL_UNCHANGED = TRUE`
- `NO_TARGET_RESTORE = TIMEOUT`
- `MATCHING_TARGET_RESTORE = TIMEOUT`
- `POLICY_INFERENCE = NOT_RUN_RESTORE_GATE_FAILED`
- `ROOT_CAUSE = UNRESOLVED_NATIVE_ORBAX_TENSORSTORE_COORDINATION`
- `OBJECT_CHECKPOINT_RESTORE_GATE = BLOCKED`
- `OBJECT_POLICY_INFERENCE_GATE = BLOCKED`
- `OBJECT_DEMO_DATA = ACCEPTED_FOR_BASELINE`
- `OBJECT_DEMO_SUCCESS_PROVENANCE = UNVERIFIED`
- `OBJECT_M1D_B_STATUS = PARTIAL_ACCEPTED_DEMO_CHECKPOINT_BLOCKED`

Do not enter M1D-C. The next milestone is an M1D-B5 native wait/reader diagnosis using the retained producing-runtime reference, without converting the checkpoint.
