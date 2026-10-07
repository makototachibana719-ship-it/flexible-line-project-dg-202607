# Package Inventory

Audit date: 2026-08-15. Sizes are exact bytes. Archive type was determined from file magic and verified with archive tooling, not inferred from the suffix.

| Package mapping | Incoming file | File type | Size | SHA256 | Integrity/path check | Extracted copy |
|---|---|---:|---:|---|---|---|
| Cable Routing | `external/incoming/cable_routing_package.gz` | gzip-compressed POSIX TAR archive | 643,212,868 | `efc19a740204e41df70b6ae2bf0840681bce74590b4478db641329ba259615cc` | VALIDATED | `external/packages/cable_routing/CableRoute_交接包/` |
| Object Grasping | `external/incoming/object_grasping_package.zip` | ZIP (deflate) | 239,862,488 | `16ecaa47295265664ecf095a0f98e790f626c4d8abaeaa567aae2140e7b088c3` | VALIDATED | `external/packages/object_grasping/object_grasp_package/` |
| Peg Insertion | `external/incoming/peg_insertion_package.zip` | ZIP (stored entries present) | 337,343,679 | `17ba3da5cbcaafd334492cee5e91f3403740f9aa6836a18a3170ae74982e0c06` | VALIDATED | `external/packages/peg_insertion/peg_insertion/` |

Mapping evidence is PRESENT in both the incoming names and top-level archive paths/content (`CableRoute_交接包`, `object_grasp_package`, `peg_insertion`). No ambiguity was found.

## Extraction evidence

| Extracted package | Files | Directories | Apparent bytes (`du -sb`) |
|---|---:|---:|---:|
| Cable Routing | 130 | 39 | 647,788,783 |
| Object Grasping | 175 | 28 | 568,099,508 |
| Peg Insertion | 1,038 | 242 | 594,304,247 |

All original archives remain PRESENT and unchanged. SHA256 values were measured before extraction. No archive entry used an absolute path or parent-directory traversal.
