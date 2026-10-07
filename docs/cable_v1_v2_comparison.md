# Cable Routing V1 / V2 Comparison

Date: 2026-08-21

## Package mapping

| Version | Archive | SHA256 | Isolated extraction |
|---|---|---|---|
| V1 | `external/incoming/cable_routing_package.gz` | `efc19a740204e41df70b6ae2bf0840681bce74590b4478db641329ba259615cc` | `external/packages/cable_routing/CableRoute_交接包/` |
| V2 | `external/incoming/cable_routing_package(1).gz` | `5110d10c1e30f2567014fb1d58e0e0cadaf1dfe3d5bcda03259023ecfe202fcb` | `external/packages/cable_routing_v2/CableRoute_最终交接包_V2_20260820_221714/` |

Both files are gzip-compressed tar archives by content inspection. V2 passed `gzip -t`, tar enumeration, and a 289-member safe-path check: no absolute paths, parent traversal, device entries, symlinks, or hardlinks. The original archive and V1 extraction were not modified.

V2 contains 242 regular files. Its supplied `SHA256SUMS.txt` validates all 240 other listed files, but its self-entry does not match the checksum file itself. This manifest-generation defect is recorded as a remaining issue; the archive-level gzip integrity and archive SHA256 are valid.

## Major differences

| Area | V1 | V2 | Finding |
|---|---|---|---|
| Layout | Handover folders `01_...` through `06_...` | Runnable `franka_sim/`, `async_cable_route_sim/`, `checkpoints/`, `metadata/` | Repackaged; V1 preserved |
| Total files | 130 | 242 | V2 adds package files, metadata, backups, bytecode, README, and assets |
| Franka asset tree | MISSING | 67 files PRESENT | Primary functional repair |
| Total STL/OBJ mesh files | 0 | 75 | V2 supplies Franka and Robotiq meshes |
| Previous failing mesh | `link1.stl` MISSING | PRESENT, 15,084 bytes | Runtime load validated |
| Environment code | PRESENT | PRESENT | Core Cable environment SHA256 identical |
| Cable MJCF/XML | PRESENT | PRESENT | `panda.xml` and `cable_route.xml` SHA256 identical |
| Gym registration | PRESENT | PRESENT | `PandaCableRoute-v0` and `PandaCableRouteVision-v0`, 500-step limit |
| Observation/action/reward/success | Static only; create failed | Same source semantics; runtime validated | V2 fixes assets, not task logic |
| Training entry | `async_drq_sim.py` | `async_cable_route_sim/async_drq_sim.py` | SHA256 identical; imports `DrQAgent` and calls `make_drq_agent` |
| Checkpoints | 105 files, 625,944,376 bytes | Same count and byte size | PRESENT but still UNVERIFIED; no restore performed |
| Demo data | MISSING | MISSING | Human-confirmed; self-record required |
| Demo recorder | MISSING | MISSING | No Cable teleoperation/trajectory recorder supplied |
| Requirements | Conda/pip/system capture | README, requirements, environment/pip metadata | V2 explicitly declares MuJoCo 2.3.7 and Python 3.10.20 runtime evidence |

## Static behavior comparison

The main environment Python, Cable scene XML, Panda XML, and DrQ training entry have identical hashes between V1 and V2. Consequently the following semantics are unchanged:

- nested state plus optional `images/front` and `images/wrist` observations;
- state dimensions 3 + 3 + 1;
- action `Box(-1, 1, (3,), float32)` for xyz delta with closed gripper;
- reward `exp(-10 * distance_to_slot)` plus 5 on success;
- success when cable-tip/slot distance is below 0.03 m;
- success terminates and time limit truncates;
- DrQ training entry with optional demo replay input.

## Asset conclusion

`franka_emika_panda/assets/` is PRESENT in V2. MuJoCo successfully created the V2 vision environment, proving that the previously missing `link1.stl` and the other runtime-referenced meshes required for this lifecycle are resolvable in the current runtime.

## Validation boundary

V2 environment lifecycle is validated. Checkpoints, policy quality, training results, demo data, demo schema, and success rate remain UNVERIFIED or MISSING as separately recorded.
