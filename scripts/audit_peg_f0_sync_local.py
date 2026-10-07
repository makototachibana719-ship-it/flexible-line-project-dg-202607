"""Offline audit of the one completed B arm. No training or cloud calls."""
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
from flax.serialization import msgpack_restore

root=Path(__file__).resolve().parents[1]
out=root/'results/validation/peg/f0_sync_20260904'
rows=[json.loads(s) for s in (out/'learner/updates.jsonl').read_text().splitlines()]
phases=[json.loads(s) for s in (out/'learner/phases.jsonl').read_text().splitlines()]
checks={'updates_50':[r['update'] for r in rows]==list(range(50)),
        'finite':all(r['losses_finite'] for r in rows),
        'rng':all(r['rng_before']!=r['rng_after'] for r in rows) and all(a['rng_after']==b['rng_before'] for a,b in zip(rows,rows[1:])),
        'single_pid':len({r['pid'] for r in rows})==1,
        'sync_50':sum(p['phase']=='synchronization_complete' for p in phases)==50,
        'no_timeout':all(r['update_seconds']<120 for r in rows)}
def leaves(x):
    if isinstance(x,dict):
        for v in x.values():yield from leaves(v)
    else:yield np.asarray(x)
bundles=[]
for step in (0,25):
    cp=out/f'checkpoints/checkpoint_{step}'
    rp=cp.with_name(cp.name+'_online_replay.pkl');mp=cp.with_name(cp.name+'_resume_metadata.json')
    m=json.loads(mp.read_text());r=pickle.loads(rp.read_bytes());state=msgpack_restore(cp.read_bytes())
    valid=m['actor_steps']==r['actor_steps']==r['size']==r['insert_index']==m['online_replay_size']
    valid=valid and hashlib.sha256(rp.read_bytes()).hexdigest()==m['online_replay_sha256']
    valid=valid and np.asarray(state['rng']).tolist()==rows[step]['rng_after'] and all(np.isfinite(v).all() for v in leaves(state))
    bundles.append(dict(step=step,valid=bool(valid),actor_steps=r['actor_steps'],replay_size=r['size'],hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cp,rp,mp)}))
checks['bundles']=all(b['valid'] for b in bundles)
result=dict(sync_validation='PASS' if all(checks.values()) else 'FAIL',checks=checks,bundles=bundles,
            first_two=[p for p in phases if p['index']<2],
            steady_update_range=[min(r['update_seconds'] for r in rows[2:]),max(r['update_seconds'] for r in rows[2:])],
            formal_training='BLOCKED',cloud_verification='RAW_FILTERED_QUERY_EMPTY_UNRESOLVED')
with (out/'local_sync_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
