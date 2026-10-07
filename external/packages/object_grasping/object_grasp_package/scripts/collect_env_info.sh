#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/configs}"
mkdir -p "$OUT_DIR"
python - <<'PYENV' > "$OUT_DIR/environment_versions.txt"
import sys
print("python="+sys.version.replace("\n"," "))
for m in ["numpy","scipy","jax","jaxlib","flax","optax","chex","mujoco","gym","gymnasium","tensorflow"]:
    try:
        x=__import__(m); print(f"{m}={getattr(x,'__version__','unknown')}")
    except Exception as e:
        print(f"{m}=UNAVAILABLE ({type(e).__name__}: {e})")
PYENV
python -m pip freeze > "$OUT_DIR/pip_freeze.txt"
conda env export --no-builds > "$OUT_DIR/environment.yml" || true
cat "$OUT_DIR/environment_versions.txt"
