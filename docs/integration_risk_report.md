# Integration Risk Report

| Priority | Risk | Evidence | Status | Required mitigation |
|---|---|---|---|---|
| P0 | Visual wrapper mismatch | Current SERL expects `image`; visual packages expose `images/front,wrist` | INCOMPATIBLE | Add project-level multi-camera adapter after isolated smoke verification; do not edit packages. |
| P0 | Success/termination mismatch | Cable terminates on success; Object reports success but terminates on time; Peg maps termination to success | INCOMPATIBLE | Define explicit adapter-level `success`, `terminated`, `truncated`, `failure_reason`. |
| P0 | No Cable demo | No demo data or recorder present | MISSING | Request authentic data/recorder or proceed with non-demo algorithm path; never synthesize data. |
| P0 | No formal evaluation | All package documents disclaim formal unified evaluation | MISSING | Build common short evaluation harness after runtime validation. |
| P1 | Heterogeneous agent inputs | Visual DrQ versus state SAC/RLPD/BC; action 3/4/7D | INCOMPATIBLE | Separate process adapters and policy factories. |
| P1 | Checkpoints not restored | Files/directories exist only | UNVERIFIED | Restore candidates with exact package dependencies and record step/seed. |
| P1 | Demo contents not independently checked | Object/Peg data files exist; supplied metadata only | UNVERIFIED | Use controlled, non-mutating schema inspection; treat pickle as untrusted. |
| P1 | Dependency/version drift | Separate environment files and mixed Gym/Gymnasium usage | UNVERIFIED | Build isolated dependency matrix; no broad upgrade/reinstall. |
| P1 | Peg reward documentation drift | README says axial coefficient -20; source applies two -20 terms | UNVERIFIED | Confirm intent with package owner before evaluation or adapter work. |
| P1 | Duplicate external code | Peg includes full `mujoco_env`; modified SERL BC file; visual packages carry env copies | UNVERIFIED | Diff against current external repositories before selecting wrappers/adapters. |
| P1 | Missing/empty delivery items | Cable README missing; two named checkpoint dirs empty; visual modified wrapper missing | MISSING | Request clarification/artifacts and record provenance. |
| P2 | Absolute submitter paths | Peg explicitly lists `/home/makoto/...`; other scripts may include local paths | INCOMPATIBLE | Parameterize only in project launch wrappers later. |
| P2 | Archive storage footprint | Extracted copies total about 1.7 GiB | PRESENT | Preserve originals and document provenance; avoid duplicate copying. |

Current integration readiness is UNVERIFIED. Proceeding directly to a unified real Skill implementation would be unsafe.
