#!/usr/bin/env bash
set -euo pipefail
SERL_ROOT="${SERL_ROOT:-$HOME/serl}"
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/object_grasp_package}"
EXAMPLE="$SERL_ROOT/examples/async_drq_sim"
ENV_SRC="$SERL_ROOT/franka_sim/franka_sim"
CKPT_SRC="$SERL_ROOT/checkpoints/drq_seed0_5000_run2/checkpoint_4500"
DEMO_SRC="$EXAMPLE/franka_lift_cube_image_20_trajs.pkl"
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"/{code/example,code/env,code/modified_serl,scripts,checkpoints,demos,results,configs,logs}
cp "$SRC_ROOT/README.md" "$OUT_ROOT/README.md"
cp "$SRC_ROOT/MANIFEST.md" "$OUT_ROOT/MANIFEST.md"
cp "$SRC_ROOT/CHECKLIST.md" "$OUT_ROOT/CHECKLIST.md"
cp -a "$SRC_ROOT/scripts/." "$OUT_ROOT/scripts/"
cp -a "$SRC_ROOT/configs/." "$OUT_ROOT/configs/"
cp -a "$SRC_ROOT/results/." "$OUT_ROOT/results/"
cp "$SRC_ROOT/demos/demo_info.md" "$OUT_ROOT/demos/"
[[ -d "$EXAMPLE" ]] || { echo "Missing $EXAMPLE" >&2; exit 2; }
cp -a "$EXAMPLE/." "$OUT_ROOT/code/example/"
[[ -d "$ENV_SRC" ]] && cp -a "$ENV_SRC" "$OUT_ROOT/code/env/franka_sim" || echo "WARNING env missing: $ENV_SRC" >&2
[[ -f "$DEMO_SRC" ]] && cp -a "$DEMO_SRC" "$OUT_ROOT/demos/" || echo "WARNING demo missing: $DEMO_SRC" >&2
[[ -d "$CKPT_SRC" ]] && cp -a "$CKPT_SRC" "$OUT_ROOT/checkpoints/" || echo "WARNING checkpoint missing: $CKPT_SRC" >&2
for f in learner_5000_seed0_run2.log actor_5000_seed0_run2.log; do [[ -f "$EXAMPLE/$f" ]] && cp "$EXAMPLE/$f" "$OUT_ROOT/logs/" || echo "WARNING log missing: $f" >&2; done
bash "$OUT_ROOT/scripts/collect_env_info.sh" "$OUT_ROOT/configs" || true
if git -C "$SERL_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
 git -C "$SERL_ROOT" status --short > "$OUT_ROOT/configs/git_status.txt" || true
 git -C "$SERL_ROOT" diff > "$OUT_ROOT/configs/local_git_diff.patch" || true
 while IFS= read -r rel; do
   [[ -z "$rel" ]] && continue
   case "$rel" in serl_launcher/*) if [[ -f "$SERL_ROOT/$rel" ]]; then mkdir -p "$OUT_ROOT/code/modified_serl/$(dirname "$rel")"; cp "$SERL_ROOT/$rel" "$OUT_ROOT/code/modified_serl/$rel"; fi;; esac
 done < <(git -C "$SERL_ROOT" diff --name-only || true)
fi
if [[ -f "$OUT_ROOT/demos/franka_lift_cube_image_20_trajs.pkl" ]]; then python "$OUT_ROOT/scripts/inspect_demo.py" "$OUT_ROOT/demos/franka_lift_cube_image_20_trajs.pkl" > "$OUT_ROOT/demos/demo_info_generated.txt" || true; fi
find "$OUT_ROOT" -type f -printf '%P\t%s bytes\n' | sort > "$OUT_ROOT/FILE_INVENTORY.txt"
python - "$OUT_ROOT" <<'PYZIP'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import sys
root=Path(sys.argv[1]); out=root.parent/'object_grasp_package.zip'
if out.exists(): out.unlink()
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(root.parent))
print(out)
PYZIP
echo "DONE: $OUT_ROOT"
