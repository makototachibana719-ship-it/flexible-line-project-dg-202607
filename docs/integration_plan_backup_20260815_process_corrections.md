# Provisional Integration Plan

Status: UNVERIFIED proposal based on completed static audit. No implementation is authorized by this document.

1. Reproduce each package in an isolated dependency environment and run import/reset/one-step smoke checks.
2. Inspect Object/Peg demos without modification and restore one explicitly selected checkpoint per process.
3. Implement project-owned adapters: multi-camera normalization for Cable/Object, state pass-through for Peg, and explicit success/termination normalization for all.
4. Run 1–5 episode quick evaluations and record truthful metrics/failures; keep all models UNVERIFIED until required formal evaluation is complete.
5. Finalize `RobotSkill`, `SkillTask`, and `SkillResult` only after adapter evidence exists.

Cable demo remains MISSING and is not a reason to halt environment/checkpoint auditing. It must not be synthesized.
