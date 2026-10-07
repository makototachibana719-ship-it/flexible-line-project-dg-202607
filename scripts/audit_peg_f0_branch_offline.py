"""Supplemental read-only artifact checks. No GPU, W&B or training calls."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flax.serialization import msgpack_restore
from training.peg_insertion import branch_acceptance as b
from training.peg_insertion.formal_policy import sha

def flatten(tree,path=''):
    if isinstance(tree,dict):
        for key,value in tree.items():yield from flatten(value,path+'/'+str(key))
    else:yield path,np.asarray(tree)

before=dict(flatten(msgpack_restore(b.SOURCE.read_bytes())))
after=dict(flatten(msgpack_restore((b.OUT/'checkpoints/checkpoint_30').read_bytes())))
assert before.keys()==after.keys() and len(after)==198
assert all(np.isfinite(value).all() for value in after.values())
counts={path:dict(before=int(before[path]),after=int(value),delta=int(value)-int(before[path]))
        for path,value in after.items() if value.size==1 and (path.endswith('/count') or path=='/step')}
assert counts and all(v['delta']>0 for v in counts.values())
old=json.loads((b.ROOT/'results/diagnostics/peg_f0_verifier_fix_20260904/reaudit.json').read_text())['source_hashes']
unchanged={path:sha(b.ROOT/path)==expected for path,expected in old.items()
           if path!='training/peg_insertion/run_peg_training.py'}
assert all(unchanged.values())
protected=json.loads((b.OUT/'protected_before.json').read_text())
assert all(sha(Path(path))==expected for path,expected in protected.items())
rows=json.loads((b.OUT/'branch_cloud_logical_steps.json').read_text())
assert [r['_step'] for r in rows]==list(range(26,31))
result=dict(status='PASS',final_state_leaves=198,all_final_state_finite=True,
    optimizer_counters=counts,unchanged_source_files=unchanged,
    protected_artifacts_unchanged=True,cloud_steps_exact_26_30=True)
b.write_evidence('offline_final.json',result)
print(json.dumps(result,indent=2))
