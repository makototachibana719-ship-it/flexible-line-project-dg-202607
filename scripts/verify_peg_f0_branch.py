"""One-shot F0-R supervisor/auditor. No learner implementation or retries."""
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.peg_insertion import branch_acceptance as b
from training.peg_insertion.formal_policy import validate_runtime, sha, SOURCES
from verify_peg_f0 import LOSS_KEYS, unique_history

def read(path):return json.loads(path.read_text())
def cloud(run):
    return dict(id=run.id,state=run.state,rows=list(run.scan_history(
        keys=['_step',*LOSS_KEYS],page_size=7,use_cache=False)))

def audit():
    import numpy as np
    import wandb
    from flax.serialization import msgpack_restore
    records=[json.loads(line) for line in (b.OUT/'learner/updates.jsonl').read_text().splitlines()]
    summary=read(b.OUT/'learner/result.json')
    assert [r['update'] for r in records]==list(range(26,31))
    assert len({r['pid'] for r in records})==1 and summary['updates_completed']==5
    assert all(r['losses_finite'] and r['rng_before']!=r['rng_after'] and r['update_seconds']<120 for r in records)
    assert all(x['rng_after']==y['rng_before'] for x,y in zip(records,records[1:]))
    for label in ('state_restore','state_before_first_update','replay_restore'):
        assert read(b.OUT/(label+'.json'))['status']=='PASS'
    assert records[0]['rng_before']==read(b.OUT/'state_restore.json')['rng']
    assert summary['per_leaf_sync'] and summary['resume_branch'] and not summary['fresh_initialization']
    branch_id=read(b.OUT/'launch.json')['branch_run_id']
    assert summary['wandb_run_id']==branch_id and branch_id!=b.SOURCE_RUN
    api=wandb.Api(timeout=30)
    run=api.run('/'.join((summary['wandb_entity'],summary['wandb_project'],branch_id)))
    history=cloud(run);b.write_evidence('branch_cloud.json',history)
    local={r['update']:{k:r['losses'][k] for k in LOSS_KEYS} for r in records}
    warning=unique_history(history['rows'],range(26,31),local)
    assert run.state=='finished'
    assert all(run.config.get(k)==v for k,v in b.provenance().items())
    logical=list(run.scan_history(keys=['_step','learner_update'],page_size=7,use_cache=False))
    b.write_evidence('branch_cloud_logical_steps.json',logical)
    assert all(row['_step']==row['learner_update'] for row in logical)
    assert set(range(26,31)).issubset({r['_step'] for r in logical})
    bundles=[]
    for step in (26,30):
        cp=b.OUT/f'checkpoints/checkpoint_{step}'
        rp=cp.with_name(cp.name+'_online_replay.pkl');mp=cp.with_name(cp.name+'_resume_metadata.json')
        raw=msgpack_restore(cp.read_bytes());r=pickle.loads(rp.read_bytes());m=read(mp)
        assert m['checkpoint_step']==m['learner_update_steps']==step
        assert m['actor_steps']==r['actor_steps']==r['size']==r['insert_index']
        assert r['size']>=1261 and m['online_replay_size']==r['size'] and m['online_replay_sha256']==sha(rp)
        assert m['wandb_run_id']==branch_id and m['wandb_name']==b.RUN_NAME
        assert all(m['source_snapshot'].get(k)==v for k,v in b.provenance().items())
        assert np.asarray(raw['rng']).tolist()==records[step-26]['rng_after']
        if step==30:
            assert b.state_fingerprint(raw)==read(b.OUT/'final_state_fingerprint.json')
            assert r['size']>1261
        bundles.append(dict(step=step,actor_steps=r['actor_steps'],size=r['size'],
            hashes={p.name:sha(p) for p in (cp,rp,mp)}))
    actor=(b.OUT/'actor.log').read_text(errors='replace')
    learner=(b.OUT/'learner.log').read_text(errors='replace')
    assert 'BRANCH_ACTOR_START_STEP: 1261' in actor and 'BRANCH_ACTOR_NETWORK_RECEIVED' in actor
    assert 'TRAINING_PROFILE: acceptance' in learner and 'DIAGNOSTIC_ENABLED: False' in learner
    assert summary['final_actor_steps']>1261
    return dict(status='PASS',updates=[26,27,28,29,30],bundles=bundles,
        wandb_run_id=branch_id,wandb_state=run.state,duplicate_identical_warning=warning,
        live_transitions_accepted=summary['final_actor_steps']-1261,
        formal_ready=False,**b.provenance())

def main():
    validate_runtime();b.validate_source(b.SOURCE,b.SOURCE_RUN)
    b.OUT.mkdir(parents=True,exist_ok=False)
    protected=list((b.SOURCE_ROOT/'checkpoints').glob('checkpoint_*'))
    protected += [b.SOURCE_ROOT/'audit.json',b.SOURCE_ROOT/'cloud.json',b.SOURCE_ROOT.parent/'formal_readiness.json']
    hashes={str(p):sha(p) for p in protected}
    b.write_evidence('protected_before.json',hashes)
    b.write_evidence('source_hashes.json',{p:sha(b.ROOT/p) for p in SOURCES})
    children=[];streams=[];error=None;before=None
    try:
        import wandb
        src=read(b.SOURCE_ROOT/'learner/result.json')
        source_path='/'.join((src['wandb_entity'],src['wandb_project'],b.SOURCE_RUN))
        before=cloud(wandb.Api(timeout=30).run(source_path))
        b.write_evidence('source_cloud_before.json',before)
        original=[json.loads(line) for line in (b.SOURCE_ROOT/'learner/updates.jsonl').read_text().splitlines()]
        unique_history(before['rows'],range(50),{r['update']:{k:r['losses'][k] for k in LOSS_KEYS} for r in original})
        assert before['state']=='finished'
        b.write_evidence('launch.json',dict(branch_run_id=uuid.uuid4().hex[:8],**b.provenance()))
        def launch(role):
            stream=(b.OUT/(role+'.log')).open('x');streams.append(stream)
            child=subprocess.Popen(['bash','scripts/run_peg_f0_branch_role.sh',role],
                cwd=b.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);return child
        learner=launch('learner');time.sleep(0.5);actor=launch('actor')
        deadline=time.monotonic()+210
        while learner.poll() is None:
            if actor.poll() is not None:raise RuntimeError('actor exited before learner')
            if time.monotonic()>deadline:raise TimeoutError('F0-R total deadline')
            time.sleep(1)
        if learner.returncode:raise RuntimeError('learner failed; no retry')
    except Exception as exc:error=repr(exc)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()
        b.write_evidence('processes.json',dict(pids=[p.pid for p in children],
            exit_codes=[p.returncode for p in children],error=error,retry=False,restart=False))
    result=dict(status='FAIL',error=error)
    try:
        if error is None:result=audit()
        if before is not None:
            after=cloud(wandb.Api(timeout=30).run(source_path))
            b.write_evidence('source_cloud_after.json',after)
            # Preserve raw views; compare logical values to tolerate identical API duplicates.
            def normalized(view):return {r['_step']:{k:r[k] for k in LOSS_KEYS} for r in view['rows']}
            unique_history(after['rows'],range(50),normalized(before))
            assert after['state']==before['state'] and normalized(before)==normalized(after)
        assert all(sha(Path(p))==h for p,h in hashes.items())
        assert not (b.ROOT/'configs/peg_insertion/runtime_gpu_formal.env').exists()
        residual=subprocess.run(['pgrep','-af','training.peg_insertion.run_peg_training'],capture_output=True,text=True)
        b.write_evidence('residual.json',dict(returncode=residual.returncode,output=residual.stdout))
        assert residual.returncode==1
    except Exception as exc:result=dict(status='FAIL',error=repr(exc))
    result.update(formal_training_started=False,retry=False,restart=False,
        sync_exact_root_cause='UNRESOLVED',formal_ready=False,wandb_mode='NEW_BRANCH_RUN')
    b.write_evidence('audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    print('PEG_FORMAL_EXACT_PATH_CHECKPOINT_RESUME_5='+result['status'],flush=True)
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
