# Object Producing-Runtime Restore

Date: 2026-08-23

## Environment isolation

`serl` was cloned to `/home/ubuntu/miniconda3/envs/object_ckpt_compat`. Before and after the experiment, the original environment path remained `/home/ubuntu/miniconda3/envs/serl` and its `pip freeze` SHA256 remained `bf73547e3e4761d56b66fb16facf2c009a216b3feb62098038d45392c7fdf14a`.

`ORIGINAL_SERL_UNCHANGED = TRUE`

The clone initially had the identical freeze hash. TensorStore 0.1.78 was then installed into the clone with `pip --no-deps`. The post-install freeze diff contains exactly one line change: TensorStore 0.1.51 to 0.1.78.

`TENSORSTORE_0_1_78_INSTALLATION = PASS`

`RUNTIME_REPRODUCTION = PASS`

| Component | Producing | Compatibility |
|---|---:|---:|
| Python | 3.10.20 | 3.10.20 |
| JAX | 0.4.35 | 0.4.35 |
| JAXlib | 0.4.35 | 0.4.35 |
| Flax | 0.8.5 | 0.8.5 |
| Orbax | 0.6.4 | 0.6.4 |
| TensorStore | 0.1.78 | 0.1.78 |
| Optax | 0.2.3 | 0.2.3 |

## Read results

- OCDBT open PASS: 0.00033 s.
- 1,196-key traversal PASS: 0.01217 s.
- Scalar read PASS: open 0.01093 s, read 0.00187 s.
- 9,437,184-byte encoder read PASS: open 0.01026 s, read 0.07725 s.
- 9,437,184-byte optimizer read PASS: open 0.00977 s, read 0.03361 s.
- `PyTreeCheckpointer.metadata()` TIMEOUT after 60 s.
- Target-free restore TIMEOUT after 120.34 s.
- Matching-target restore TIMEOUT after 300 s; no state returned.

The no-target process read 176,033,792 bytes and reached about 519 MiB RSS by nine seconds, then remained sleeping with unchanged CPU/I/O through the timeout.

`NO_TARGET_RESTORE = TIMEOUT`

`MATCHING_TARGET_RESTORE = TIMEOUT`

`POLICY_INFERENCE = NOT_RUN_RESTORE_GATE_FAILED`

`ROOT_CAUSE = UNRESOLVED_NATIVE_ORBAX_TENSORSTORE_COORDINATION`

TensorStore drift alone is not a sufficient cause: the exact producing version reproduces the stall. The checkpoint remains unmodified and storage/individual tensors remain readable. The compatibility environment is retained as `OBJECT CHECKPOINT COMPATIBILITY REFERENCE`.

## Demo decision

`OBJECT_DEMO_DATA = ACCEPTED_FOR_BASELINE`

`OBJECT_DEMO_USE_DECISION = ACCEPTED_BY_PROJECT`

`OBJECT_DEMO_SUCCESS_PROVENANCE = UNVERIFIED`

The baseline will consume the original 2,000 transitions without modification, splitting, or rerecording. Project acceptance is not transition-derived proof of success.
