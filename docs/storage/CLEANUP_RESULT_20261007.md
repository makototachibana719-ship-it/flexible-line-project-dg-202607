# Cleanup result — 2026-10-07

The authorized cleanup completed after preserving unique source files and lightweight diagnostic evidence in this private repository.

## Recovered space

- Windows redundant working trees, mirrors, expanded delivery content, and transient directories: approximately **11.97 GiB** removed.
- WSL extraction/portability copies, superseded return packages, and diagnostic roundtrip checkpoints: approximately **10 GiB** removed.
- WSL Python caches: **143 `__pycache__` directories** (6,601,079 bytes) removed.
- Supplemental WSL exact-reproduction duplicate: **4,767,141,004 bytes** removed.
- Supplemental Windows historical ZIP after delta preservation: **5,317,155,743 bytes** removed.
- Final free space reported by Windows drive `D:`: **58,099,019,776 bytes** (approximately 54.11 GiB).
- Post-cleanup free space reported inside WSL: **927 GiB**.

These figures are logical filesystem savings. The dynamically allocated WSL virtual disk file may not shrink on the Windows host until an explicit offline compaction is performed; compaction was not attempted during this cleanup.

## Integrity checks passed

- Windows delivery archive `PEG_Training_Return_Complete_20260910_006.tar.gz` remains present with its sidecar.
- Its SHA-256 is `7C1FB92FC0BE807F76DF23B10582A4EB4D205D735E4C6669C78D1F948218A50E`, matching the sidecar.
- Historical Windows archive `flexible_line_project.zip` was removed only after its 27 non-disposable deltas were extracted, hashed, committed, pushed, and verified remotely.
- The authoritative WSL project remains present at `/home/makoto/flexible_line_project`.
- Formal WSL checkpoints, demos, datasets, training, ablation, validation, and closure assets all remain present.
- Return package 005, its archive, and SHA-256 sidecar remain present; `sha256sum -c` reported `OK`.
- Every explicitly removed Windows and WSL target was checked and found absent.
- No `__pycache__` directory remains in the authoritative WSL project.

## Intentionally retained for safety

- All formal checkpoints and experiment outputs listed in the pre-cleanup audit.

## Supplemental deduplication review

The two initially retained large candidates were subsequently reviewed at file-content level:

- The failed exact-reproduction tree contained 2,884 files totaling 5,221,230,339 bytes. Every file matched the corresponding file in the authoritative WSL project by path, size, and CRC-32, so the redundant tree was approved for deletion.
- The historical Windows ZIP contained 2,043 files totaling 11,501,107,945 uncompressed bytes. Of these, 1,827 files (6,749,276,711 bytes) matched the authoritative WSL project. The large unmatched portion consisted of the already authorized disposable roundtrip checkpoint trees; Python caches were also disposable.
- The remaining 27 historical or path-only candidates, totaling 228,163 bytes, were extracted to `preserved/historical_zip_20260828/` with a per-file SHA-256 manifest before the ZIP was approved for deletion.

After GitHub remote verification, both reviewed targets were deleted by exact absolute path and independently checked to be absent. The authoritative WSL project and formal assets were not modified by this supplemental deletion.
