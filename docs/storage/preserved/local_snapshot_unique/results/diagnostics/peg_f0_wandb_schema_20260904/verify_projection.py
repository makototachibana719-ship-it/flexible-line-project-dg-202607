"""Read-only projection cross-check. Does not modify the production verifier/Gate."""
import json
import math
from pathlib import Path
import wandb
out=Path(__file__).parent
e=json.loads((out/'raw_evidence.json').read_text())
root=Path('/home/makoto/flexible_line_project')
updates=[json.loads(s) for s in (root/'results/validation/peg/f0_sync_20260904/learner/updates.jsonl').read_text().splitlines()]
keys=['critic.critic_loss','actor.actor_loss','temperature.temperature_loss']
run=wandb.Api(timeout=30).run('makototachibana719-beihang-university/flexible-line-peg-insertion/hr88a6u1')
dot=list(run.scan_history(keys=['_step']+keys,page_size=7,use_cache=False))
slash=list(run.scan_history(keys=['_step']+[k.replace('.','/') for k in keys],page_size=7,use_cache=False))
checks={}
for name,rows in [('sdk',e['local']),('cloud_unfiltered',e['cloud']),('cloud_dot',dot)]:
    checks[name+'_unique_complete']=len(rows)==50 and sorted(r['_step'] for r in rows)==list(range(50))
    checks[name+'_loss_match']=all(all(k in r and math.isfinite(r[k]) and math.isclose(r[k],updates[r['_step']]['losses'][k],rel_tol=1e-7,abs_tol=1e-9) for k in keys) for r in rows) and len(rows)==50
checks['slash_empty']=slash==[]
checks['finished']=run.state=='finished'
result=dict(status='PASS' if all(checks.values()) else 'FAIL',classification='VERIFIER_FIELD_SCHEMA_MISMATCH',checks=checks,dot_rows=dot,slash_rows=slash,formal_training='BLOCKED')
with (out/'projection_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('dot_rows','slash_rows')},indent=2))
