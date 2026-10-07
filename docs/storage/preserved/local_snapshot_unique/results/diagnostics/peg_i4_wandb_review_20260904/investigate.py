"""Read-only source/run investigation; writes only this report directory."""
import collections
import hashlib
import json
import os
from pathlib import Path
import time
import wandb
from wandb.sdk.internal.datastore import DataStore
from wandb.proto.wandb_internal_pb2 import Record

ROOT=Path('/home/makoto/flexible_line_project')
OUT=Path(__file__).resolve().parent
R3=ROOT/'results/diagnostics/peg_i3_checkpoint_20260904_132008'
R4=ROOT/'results/diagnostics/peg_i4_resume_20260904_133856'
KEYS=['_step','learner_update','critic_loss','actor_loss','temperature_loss']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):
    with (OUT/name).open('x') as f:json.dump(data,f,indent=2)
def summary(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[str(row.get('_step'))].append(row)
    return dict(rows=len(rows),counts={k:len(v) for k,v in groups.items()},
                conflicting_steps=[k for k,v in groups.items() if any(x!=v[0] for x in v[1:])])

result={'sdk':wandb.__version__,'local':{},'cloud':{},'time':time.time()}
for label,root in [('I3',R3),('I4',R4)]:
    files=list(root.glob('wandb/run-*/run-44yrxufd.wandb'))
    assert len(files)==1
    p=files[0];before=digest(p)
    ds=DataStore()
    # SDK open_for_scan uses r+b; initialize the same reader explicitly rb.
    ds._fname=str(p);ds._fp=p.open('rb');ds._index=0
    ds._size_bytes=p.stat().st_size;ds._opened_for_scan=True;ds._read_header()
    rows=[];types=collections.Counter()
    try:
        while True:
            data=ds.scan_data()
            if data is None:break
            rec=Record();rec.ParseFromString(data)
            kind=rec.WhichOneof('record_type');types[kind]+=1
            if kind=='history':
                row={}
                for item in rec.history.item:
                    key=item.key or '.'.join(item.nested_key)
                    row[key]=json.loads(item.value_json)
                rows.append(row)
    finally:ds._fp.close()
    calls=[json.loads(s) for s in (root/'wandb_calls.jsonl').read_text().splitlines()]
    updates=[json.loads(s) for s in (root/'updates.jsonl').read_text().splitlines()]
    result['local'][label]=dict(file=str(p),sha256=before,unchanged=before==digest(p),
        history_summary=summary([{k:r[k] for k in KEYS if k in r} for r in rows]),
        record_types=dict(types),log_call_indices=[c['index'] for c in calls],
        completed_update_indices=[r['index'] for r in updates if r['event']=='complete'])
    save(label+'_event_history.json',rows)
save('local.json',result['local'])
run=wandb.Api(timeout=30).run('makototachibana719-beihang-university/flexible-line-peg-insertion/44yrxufd')
result['run_state']=run.state
for name,query in [
    ('scan_uncached_100',lambda:list(run.scan_history(keys=KEYS,page_size=100,min_step=0,max_step=25,use_cache=False))),
    ('scan_uncached_7',lambda:list(run.scan_history(keys=KEYS,page_size=7,min_step=0,max_step=25,use_cache=False))),
    ('graphql_sampled',lambda:run._sampled_history(KEYS[1:],samples=1000)),
]:
    try:
        rows=query();save(name+'.json',rows);result['cloud'][name]=summary(rows)
    except Exception as exc:result['cloud'][name]={'error':type(exc).__name__+': '+str(exc)[:500]}
save('investigation.json',result)
print(json.dumps(result,indent=2))
