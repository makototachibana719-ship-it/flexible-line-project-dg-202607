"""Read-only local SDK decoding and public cloud history queries; no wandb.init."""
import collections
import json
import math
from pathlib import Path
import wandb
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

root=Path('/home/makoto/flexible_line_project')
source=root/'results/validation/peg/f0_sync_20260904'
raw=source/'logs/wandb/run-20260904_165030-hr88a6u1/run-hr88a6u1.wandb'
ds=DataStore();ds.open_for_scan(str(raw));local=[]
while True:
    data=ds.scan_data()
    if data is None:break
    record=wandb_internal_pb2.Record();record.ParseFromString(data)
    if record.HasField('history'):
        row={}
        for item in record.history.item:
            key=item.key or '.'.join(item.nested_key)
            row[key]=json.loads(item.value_json)
        local.append(row)
run=wandb.Api(timeout=30).run('makototachibana719-beihang-university/flexible-line-peg-insertion/hr88a6u1')
rows=list(run.scan_history(page_size=100,use_cache=False))
print(json.dumps(dict(state=run.state,local_rows=len(local),local_keys=sorted({k for r in local for k in r}),cloud_rows=len(rows),cloud_keys=sorted({k for r in rows for k in r}),local_first=local[:1],cloud_first=rows[:1]),indent=2))
out=Path(__file__).parent
with (out/'raw_evidence.json').open('x') as f:json.dump(dict(state=run.state,local=local,cloud=rows),f,indent=2)
