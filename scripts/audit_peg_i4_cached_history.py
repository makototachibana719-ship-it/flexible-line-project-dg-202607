#!/usr/bin/env python3
"""Offline-only I4 adjudication. Never imports W&B or executes training.

Preserves raw verifier FAIL and reports duplicate historical rows explicitly.
Only exact duplicates matching cached source I3 history are accepted as a
history warning; every new step must remain unique and match local metrics.
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/diagnostics/peg_i3_checkpoint_20260904_132008'


def audit(out):
    def read(path): return json.loads(path.read_text())
    raw=read(out/'i4_result.json')
    verification=read(out/'verification.json')
    remote=read(out/'wandb_remote.json')
    previous=read(SOURCE/'wandb_remote.json')
    calls=[json.loads(s) for s in (out/'wandb_calls.jsonl').read_text().splitlines()]
    groups={}
    for row in remote['rows']: groups.setdefault(row['_step'],[]).append(row)
    old={r['_step']:r for r in previous['rows']}
    checks=dict(
        old_history_20=set(old)==set(range(20)),
        global_step_coverage=set(groups)==set(range(25)),
        historical_rows_match_source=all(r==old.get(step) for step,rows in groups.items() if step<20 for r in rows),
        fresh_steps_unique=all(len(groups.get(step,[]))==1 for step in range(20,25)),
        local_steps=[c['index'] for c in calls]==list(range(20,25)),
        new_losses_match=all(len(groups.get(c['index'],[]))==1 and all(
            math.isfinite(groups[c['index']][0][key]) and math.isclose(groups[c['index']][0][key],c['metrics'][key],rel_tol=1e-7,abs_tol=1e-9)
            for key in ('critic_loss','actor_loss','temperature_loss')) for c in calls),
        step_semantics=all(r['_step']==r['learner_update'] for r in remote['rows']),
        finished=remote['state']=='finished',
        same_run_resumed=read(out/'wandb_run.json')['id']=='44yrxufd' and read(out/'wandb_run.json')['resumed'],
        other_checks=all(v for k,v in verification['checks'].items() if k!='wandb'),
        learner_actor_exit_zero=raw['exit_codes'][:2]==[0,0],
        source_unchanged=raw['source_unchanged'],
        no_owned_residuals=not raw['residual_owned_processes'],
    )
    result=dict(checks=checks,raw_verifier_status=raw['status'],
        lifecycle='PASS' if all(checks.values()) else 'FAIL',
        overall='PASS_WITH_WANDB_HISTORY_WARNING' if all(checks.values()) else 'FAIL',
        history_uniqueness='FAIL' if len(remote['rows'])!=25 else 'PASS',
        duplicate_historical_steps={str(k):len(v) for k,v in groups.items() if len(v)>1},
        remote_rows=len(remote['rows']),unique_steps=len(groups),
        explanation='Exact source-history duplicates only; no conflicting values or duplicate new steps. Cause of remote duplicate rows is unproven.',
        training_rerun=False,network_query=False,formal_training='BLOCKED')
    target=out/'i4_cached_history_audit.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    print('PEG_I4_RESUME_LIFECYCLE_5='+result['overall'])
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    if ROOT not in out.parents:raise ValueError('outside project')
    raise SystemExit(audit(out))
