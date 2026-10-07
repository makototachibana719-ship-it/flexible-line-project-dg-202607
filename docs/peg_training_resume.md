# Peg Safe Training Resume

Date: 2026-08-23

A complete bundle is committed in this order: legacy-Flax `checkpoint_N`, ordinary state replay pickle, then metadata JSON last. It restores agent/optimizer/target/RNG state, learner step, actor cumulative step, online replay arrays/size/index/RNG and exact W&B identity. Demo replay is rebuilt from the canonical SHA-pinned 5,512-transition dataset.

The bounded two-process test created learner step 123, actor step 17 and four real-schema online transitions; a new process restored exact agent leaf hashes, replay state/RNG, sampled `(2,20)/(2,7)` data, rebuilt 5,512 Demo transitions, performed one finite bounded inference, and committed step 125. Step 123 remained byte-identical.

Six tests failed closed: missing replay, wrong replay hash, wrong Demo hash, config mismatch, W&B mismatch and checkpoint collision. There is no silent reset.

The initial test exposed that Flax `keep=2` removed step 123 when step 125 was saved. That failed evidence is preserved under `runtime/peg_handoff/resume_roundtrip_m1de/`. After backing up the helper, formal retention was set to 20 and the clean PASS evidence was produced under `runtime/peg_handoff/resume_roundtrip_m1de_v2/`.

`PEG_RESUME_GATE = VALIDATED`

