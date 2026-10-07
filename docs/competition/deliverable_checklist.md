# Competition Deliverable Checklist

Last updated: 2026-08-13

Checklist states: `PENDING`, `UNVERIFIED`, `BLOCKED`, `DONE`, `VALIDATED`.

No item involving a real process implementation, experiment, or artifact is complete unless supported by actual inspection or test evidence.

## Process Packages and Single-Skill Evidence

| Deliverable | State | Blocking condition / evidence needed |
|---|---|---|
| Cable Routing complete package | PENDING | Package not received. |
| Object Grasping complete package | PENDING | Package not received. |
| Peg Insertion complete package | PENDING | Package not received. |
| Cable Routing package audit | BLOCKED | Package not received. |
| Object Grasping package audit | BLOCKED | Package not received. |
| Peg Insertion package audit | BLOCKED | Package not received. |
| Three-process compatibility audit | BLOCKED | All three complete packages are required. |
| Real environments and interfaces | UNVERIFIED | Inspect reset/step API, observations, actions, reward, termination, success, wrappers, assets, and registration. |
| Checkpoint inventory and validation | UNVERIFIED | No checkpoint presence or performance is currently confirmed. |
| Demo inventory and validation | UNVERIFIED | No demo presence, schema, episode count, or success evidence is currently confirmed. |
| Independent single-skill evaluations | UNVERIFIED | Requires explicit success logic and actual evaluation output. |

## System Source Code

| Deliverable | State |
|---|---|
| Common domain model | PENDING |
| Unified Skill API | PENDING |
| Mock Skills for three processes | PENDING |
| Real Skill adapters | BLOCKED |
| Dynamic Order Manager | PENDING |
| Robot Manager for at least three robots | PENDING |
| Task graph and process dependencies | PENDING |
| Shared-resource coordination and conflict handling | PENDING |
| Failure handling and dynamic replanning | PENDING |
| Logical production-line simulator | PENDING |
| Multi-robot simulation / visualization | PENDING |

## Scheduling and Decision Methods

| Deliverable | State |
|---|---|
| Fixed Assignment baseline | PENDING |
| FCFS baseline | PENDING |
| Greedy / Earliest Finish Time baseline | PENDING |
| Proposed autonomous decision method | PENDING |
| Fair benchmark configuration | PENDING |
| Ablation studies | PENDING |
| Stress tests | PENDING |

## Required Scenarios

| Scenario | State |
|---|---|
| Normal Production | PENDING |
| Dynamic Order Arrival | PENDING |
| High Priority Urgent Order | PENDING |
| Robot Failure | PENDING |
| Shared Workspace Conflict | PENDING |
| Process / Product Change | PENDING |

## Evaluation and Results

| Deliverable | State |
|---|---|
| Single-skill metric definitions | PENDING |
| System KPI mathematical definitions | PENDING |
| Reproducible seeds and scenario inputs | PENDING |
| `raw_results.csv` | PENDING |
| `summary.csv` | PENDING |
| `summary.json` | PENDING |
| Baseline results | UNVERIFIED |
| Proposed-method results | UNVERIFIED |
| Ablation and stress-test results | UNVERIFIED |
| Result figures | PENDING |

No experimental values currently exist in this checklist, and none may be fabricated.

## Documentation and Demonstration

| Deliverable | State |
|---|---|
| Project README and run instructions | PENDING |
| Architecture documentation | DONE |
| Process comparison | BLOCKED |
| Integration plan | BLOCKED |
| Interface matrix | BLOCKED |
| Technical report | PENDING |
| Reproduction guide | PENDING |
| Runnable three-robot / three-process demo | PENDING |
| Demo videos and presentation material | PENDING |
| Submission validator | PENDING |
| Final `deliverables/` package | PENDING |

## Submission Readiness Gate

Current state: **NOT READY FOR SUBMISSION**.

The project must not report `READY FOR SUBMISSION` until the validator confirms all required source code, three processes, three robots, checkpoints, demo data, dynamic scheduling, coordination, baselines, proposed method, benchmark evidence, KPI results, documentation, and demonstration materials.
