# Current Architecture

Last updated: 2026-08-13

## Scope

The project is in **PHASE 1 — Integration Infrastructure Preparation**. This document records a provisional logical architecture and its current verification boundary. It does not claim that any real process package or integration is available.

## Current Verified Boundary

Only the project documentation structure created during initialization has been checked for presence. No runtime architecture component is currently validated.

## Intended Logical Flow

```text
Dynamic Orders
      |
      v
Order Manager
      |
      v
Task Graph
      |
      v
Scheduler / Decision Layer
      |
      v
Robot Manager
      |
      v
Multi-Robot Coordination
      |
      v
Provisional Skill Boundary
      |
      +--> Mock Skills (planned for infrastructure development)
      |
      +--> Real Skill Adapters (BLOCKED pending package audits)

Evaluation observes the logical system independently.
```

## Safe PHASE 1 Development Boundary

The following may be designed without real process packages:

- Extensible common domain models.
- Mock Skill behavior clearly identified as simulated.
- Dynamic order and robot state management.
- Scheduler interfaces and baseline contracts.
- Resource reservation and coordination skeletons.
- Evaluation interfaces, KPI definitions, and result schemas.
- Logical production-line simulation and deterministic test scenarios.

These interfaces must remain adaptable and must not encode assumptions about real observation or action spaces.

## Real Process Integration Boundary

| Process | Package | Interface status | Integration status |
|---|---|---|---|
| Cable Routing | PENDING | UNVERIFIED | BLOCKED — package not received |
| Object Grasping | PENDING | UNVERIFIED | BLOCKED — package not received |
| Peg Insertion | PENDING | UNVERIFIED | BLOCKED — package not received |

For every process, the environment implementation, Gym/Gymnasium behavior, observation space, action space, reward, success condition, assets, wrappers, demo format, replay buffer, agent code, checkpoint, and evaluation path are UNVERIFIED.

## Architectural Constraints

- The Scheduler must not call SERL or DrQ agent internals directly.
- Real process adapters must not be implemented before their packages are received and audited.
- A common Skill API is provisional until the three-process compatibility audit is complete.
- Mock Skills must never be presented as evidence that a real process works.
- External repositories should be integrated through wrappers or adapters and must not be refactored during this initialization phase.
- Training and evaluation remain separate; no training metric can substitute for independent evaluation.

## Required Gate Before Unified Integration

When all three complete packages are available, pause non-critical development and perform the THREE PROCESS COMPATIBILITY AUDIT. The audit must produce:

- `docs/process_comparison.md`
- `docs/integration_plan.md`
- `docs/interface_matrix.md`

Only after that gate may the project finalize the Unified Skill API and real adapters.
