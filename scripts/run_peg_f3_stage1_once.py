"""Bounded supervisor for the explicitly authorized inference-only smoke."""
import subprocess,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
log=root/'results/validation/peg/f3_stage1_20260904.log'
with log.open('x') as stream:
    try:
        run=subprocess.run([sys.executable,'-u',str(root/'scripts/peg_f3_stage1.py')],cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        code=run.returncode
    except subprocess.TimeoutExpired:
        code=124
with log.with_suffix('.execution.json').open('x') as stream:json.dump(dict(exit_code=code,timeout=code==124,automatic_retry=False),stream)
print('F3_STAGE1_EXIT',code,flush=True)
raise SystemExit(code)
