# Provisional Integration Plan

Status: UNVERIFIED proposal based on completed static audit. No implementation is authorized by this document.

1. P0: run isolated import/reset/one-step/close environment smoke validation for Cable Routing.
2. P0: run the equivalent environment smoke validation for Object Grasping.
3. P0: run the equivalent factory/adapter smoke validation for Peg Insertion.
4. P1: design the project-owned Cable Routing Demo Recorder, then acquire authentic demos. Demo Acquisition is TODO and is not waiting for a teammate resend.
5. P1: reconcile Object Grasping's human-confirmed DPRL label with the delivered `DrQAgent` plus 50/50 demo/online replay implementation; then validate runtime and checkpoint 4500.
6. P1: validate Peg Insertion SAC/RLPD/BC mapping and selected checkpoints.
7. P1: inspect available Object/Peg demos without modification and validate their schemas.
8. Implement project-owned adapters only after the preceding evidence: multi-camera normalization for Cable/Object, state pass-through for Peg, and explicit success/termination normalization for all.
9. Run 1–5 episode quick evaluations and record truthful metrics/failures; keep all models UNVERIFIED until required formal evaluation is complete.
10. Finalize `RobotSkill`, `SkillTask`, and `SkillResult` only after adapter evidence exists.

Cable Demo Recorder and Demo Data remain MISSING. Recorder design and authentic Demo Acquisition are project TODO tasks; data must not be synthesized.
