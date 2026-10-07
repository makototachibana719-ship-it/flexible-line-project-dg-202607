# Historical ZIP preservation — 2026-08-28 snapshot

This directory preserves the small amount of non-disposable content found only in, or different from the current WSL project in, the former Windows archive `flexible_line_project.zip`.

Before removal, the archive had:

- SHA-256: `79080BB62CCC4EA6AF349708E608694FC12D4B6F6EA0A472B6CE484CCF1DC76F`
- compressed size: 5,317,155,743 bytes;
- 2,043 regular files and 11,501,107,945 uncompressed bytes.

A path/size/CRC comparison against `/home/makoto/flexible_line_project` found:

- 1,827 exact matches totaling 6,749,276,711 bytes;
- 189 missing entries belonging only to authorized disposable roundtrip checkpoint trees or `__pycache__` data;
- 27 historical candidates totaling 228,163 bytes.

Those 27 candidates are preserved here under their original relative paths. `HISTORICAL_ZIP_MANIFEST.csv` records their status relative to the current project, original size, ZIP CRC-32, and extracted SHA-256.
