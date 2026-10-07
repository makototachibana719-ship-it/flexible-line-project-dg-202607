"""Fresh read-only query process, reversed page order; no run mutation."""
import json
from pathlib import Path
from collections import Counter
import wandb
out=Path(__file__).resolve().parent
run=wandb.Api(timeout=30).run('makototachibana719-beihang-university/flexible-line-peg-insertion/44yrxufd')
keys=['_step','learner_update','critic_loss','actor_loss','temperature_loss']
result={}
for page in (7,100):
    rows=list(run.scan_history(keys=keys,min_step=0,max_step=25,page_size=page,use_cache=False))
    result[str(page)]={'rows':rows,'counts':dict(Counter(r['_step'] for r in rows))}
with (out/'reversed_order.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:{'rows':len(v['rows']),'counts':v['counts']} for k,v in result.items()},indent=2))
