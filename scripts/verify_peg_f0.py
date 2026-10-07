"""F0 process orchestration and audit, never implements a learner/update loop."""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.peg_insertion.formal_policy import ROOT,RUNTIME,SOURCES,sha,validate_runtime
OUT=ROOT/'results/validation/peg/f0_acceptance'
LOSS_KEYS=('critic.critic_loss','actor.actor_loss','temperature.temperature_loss')

def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2)

def unique_history(rows,expected,local):
    groups={}
    for row in rows:groups.setdefault(row['_step'],[]).append(row)
    if set(groups)!=set(expected):raise ValueError('missing/unexpected W&B steps')
    warning=[]
    for step,items in groups.items():
        if any(item!=items[0] for item in items[1:]):raise ValueError('conflicting W&B duplicate')
        if len(items)>1:warning.append(step)
        for key,value in local[step].items():
            if key not in items[0] or not math.isfinite(items[0][key]) or not math.isclose(value,items[0][key],rel_tol=1e-7,abs_tol=1e-9):raise ValueError('cloud/local loss mismatch')
    return warning

def audit(evidence_dir=None):
    import numpy as np
    import wandb
    from flax.serialization import msgpack_restore
    records=[json.loads(s) for s in (OUT/'learner/updates.jsonl').read_text().splitlines()]
    summary=json.loads((OUT/'learner/result.json').read_text())
    assert [r['update'] for r in records]==list(range(50))
    assert len({r['pid'] for r in records})==1 and summary['updates_completed']==50
    assert all(r['losses_finite'] and r['rng_before']!=r['rng_after'] and r['update_seconds']<120 for r in records)
    assert all(a['rng_after']==b['rng_before'] for a,b in zip(records,records[1:]))
    run=wandb.Api(timeout=30).run('/'.join((summary['wandb_entity'],summary['wandb_project'],summary['wandb_run_id'])))
    keys=['_step',*LOSS_KEYS]
    rows=list(run.scan_history(keys=keys,page_size=7,use_cache=False))
    write((OUT if evidence_dir is None else evidence_dir)/'cloud.json',{'state':run.state,'rows':rows})
    local={r['update']:{k:r['losses'][k] for k in LOSS_KEYS} for r in records}
    warning=unique_history(rows,range(50),local)
    assert run.state=='finished'
    bundles=[]
    def leaves(tree):
        if isinstance(tree,dict):
            for v in tree.values():yield from leaves(v)
        else:yield np.asarray(tree)
    for step in (0,25):
        cp=OUT/f'checkpoints/checkpoint_{step}'
        mp=cp.with_name(cp.name+'_resume_metadata.json');rp=cp.with_name(cp.name+'_online_replay.pkl')
        m=json.loads(mp.read_text());r=pickle.loads(rp.read_bytes());raw=msgpack_restore(cp.read_bytes())
        assert m['actor_steps']==r['actor_steps']==r['size']==r['insert_index']
        assert m['online_replay_sha256']==sha(rp) and m['online_replay_size']==r['size']
        assert m['checkpoint_step']==m['learner_update_steps']==step and m['wandb_run_id']==run.id
        assert np.asarray(raw['rng']).tolist()==records[step]['rng_after']
        assert all(np.all(np.isfinite(x)) for x in leaves(raw))
        bundles.append(dict(step=step,size=r['size'],actor_steps=r['actor_steps'],hashes={p.name:sha(p) for p in (cp,mp,rp)}))
    assert bundles[1]['size']>=bundles[0]['size']>=100
    text=(OUT/'learner.log').read_text(errors='replace')
    assert 'TRAINING_PROFILE: acceptance' in text and 'DIAGNOSTIC_ENABLED: False' in text
    return dict(status='PASS',updates=50,bundles=bundles,wandb_run_id=run.id,wandb_state=run.state,
                duplicate_identical_warning=warning,source_hashes={p:sha(ROOT/p) for p in SOURCES},runtime=RUNTIME,
                exact_path_resume='NOT_RUN_F0_R_REQUIRED',formal_ready=False)

def run_once(write_gate=True, extra_args=()):
    validate_runtime();OUT.mkdir(parents=True,exist_ok=False)
    print('F0_OUTPUT:',OUT,flush=True)
    children=[];streams=[];error=None
    def launch(role):
        f=(OUT/(role+'.log')).open('x');streams.append(f)
        cmd=[sys.executable,'-u','-m','training.peg_insertion.run_peg_training','--profile','acceptance','--role',role,'--require-gpu']
        cmd.extend(extra_args)
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    try:
        learn=launch('learner');time.sleep(3);act=launch('actor');deadline=time.monotonic()+240
        while learn.poll() is None:
            if act.poll() is not None:raise RuntimeError('actor exited before learner')
            if time.monotonic()>deadline:raise TimeoutError('bounded acceptance deadline')
            time.sleep(1)
        if learn.returncode:raise RuntimeError('learner failed; no retry')
    except Exception as exc:error=repr(exc)
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in streams:f.close()
    write(OUT/'processes.json',dict(pids=[p.pid for p in children],exit_codes=[p.returncode for p in children],actor_stop='bounded supervisor stop after learner exit',error=error))
    if error is None:
        try:result=audit()
        except Exception as exc:result={'status':'FAIL','error':repr(exc),'formal_ready':False}
    else:result={'status':'FAIL','error':error,'formal_ready':False}
    write(OUT/'audit.json',result)
    gate=dict(exact_path_candidate=result['status'],exact_path_resume='NOT_RUN',formal_ready=False,
              formal_training='BLOCKED',runtime=RUNTIME,source_hashes={p:sha(ROOT/p) for p in SOURCES},evidence=str(OUT/'audit.json'))
    if write_gate:write(OUT.parent/'formal_readiness.json',gate)
    print(json.dumps(result,indent=2),flush=True);print('PEG_FORMAL_EXACT_PATH_ACCEPTANCE_50='+result['status'],flush=True)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-once',action='store_true');a=p.parse_args()
    if not a.run_once:raise ValueError('--run-once required')
    raise SystemExit(run_once())
