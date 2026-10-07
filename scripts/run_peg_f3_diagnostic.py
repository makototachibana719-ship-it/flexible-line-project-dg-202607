"""Exactly two fixed-scene inference episodes with bounded execution."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/validation/peg/f3_diagnostic_20260904'
OUT.mkdir(exist_ok=False)
(OUT/'protocol.json').write_text(json.dumps(dict(checkpoints=[5000,9000],episodes_each=1,seed=0,argmax=True,reset='fixed default',horizon=500,
    timestamp_semantics='post-step state plus action just applied; one-based transition step; reset state in episode summary',
    classification='B/C when best-joint state violates respective thresholds; A when minimum position exceeds0.01m; E when >=10% action coordinates saturated (heuristic); D not concluded from force alone',
    limits='ABC/E are descriptive, not proven causal. Contact-force rejection needs causal contact evidence; do not infer from maxforce alone.'),indent=2))
for step in (5000,9000):
    with (OUT/f'checkpoint{step}.log').open('x') as f:
        try:r=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/peg_f3_diagnostic_worker.py'),str(step)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=600);code=r.returncode
        except subprocess.TimeoutExpired:code=124
    print('checkpoint',step,'exit',code,flush=True)
    if code:raise SystemExit(code)
