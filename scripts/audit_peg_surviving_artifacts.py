"""Read-only artifact adjudication; decode CPU data only, never instantiate an agent."""
import json, math, pickle, sys
from pathlib import Path
import numpy as np
from flax.serialization import msgpack_restore
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.peg_insertion import formal_policy as p, formal_preflight as f

def leaves(value):
    if isinstance(value,dict):
        for child in value.values():yield from leaves(child)
    elif isinstance(value,(list,tuple)):
        for child in value:yield from leaves(child)
    else:yield np.asarray(value)

def main():
    out=ROOT/'results/validation/peg/f2_artifact_adjudication_20260904'
    out.mkdir(exist_ok=False)
    old=ROOT/'results/validation/peg/f2_run_20260904'
    gatepath=ROOT/'results/validation/peg/formal_readiness_v3.json'
    gate=json.loads(gatepath.read_text());cfg=f.config()
    cpdir=ROOT/cfg['outputs']['checkpoint_dir']
    previous=json.loads((old/'post_audit.json').read_text())
    raw=json.loads((old/'result.json').read_text())
    monitor=Path(raw['monitor_directory'])/'formal_monitor_updates.jsonl'
    protected=list(cpdir.glob('checkpoint_*'))+list((ROOT/'results/validation/peg').glob('formal_readiness_v*.json'))
    protected+=list(old.glob('*.json'))+[monitor,ROOT/cfg['demo']['path'],ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml',ROOT/'configs/peg_insertion/runtime_gpu_formal.env']
    before={str(x.relative_to(ROOT)):p.sha(x) for x in protected}
    errors=[];bundles=[]
    def require(ok,message):
        if not ok:raise ValueError(message)
    try:
        require(p.sha(gatepath)==previous['gate_sha256'],'historical gate hash')
        require(p.bindings()==gate['bindings'],'runtime/config/demo/manifest binding')
        rows=[json.loads(x) for x in monitor.read_text().splitlines()]
        require([x['global_update'] for x in rows]==list(range(10000)),'10000 continuous updates')
        require(all(all(math.isfinite(x[k]) for k in ('critic_loss','actor_loss','temperature_loss')) for x in rows),'local finite losses')
        require(all(x['rng_before']!=x['rng_after'] for x in rows) and all(a['rng_after']==b['rng_before'] for a,b in zip(rows,rows[1:])),'RNG chain')
        require(all(x['true_update_seconds']<=120 for x in rows),'timeout')
        cloud=json.loads((old/'cloud_history.json').read_text())
        require(sorted(x['_step'] for x in cloud)==list(range(0,10000,10)),'cached cloud unique complete history')
        for x in cloud:
            for key,local in [('critic.critic_loss','critic_loss'),('actor.actor_loss','actor_loss'),('temperature.temperature_loss','temperature_loss')]:
                require(math.isfinite(x[key]) and math.isclose(x[key],rows[x['_step']][local],rel_tol=1e-6,abs_tol=1e-6),'cloud/local loss')
        for step in range(3000,10000,1000):
            cp=cpdir/f'checkpoint_{step}';rp=cp.with_name(cp.name+'_online_replay.pkl');mp=cp.with_name(cp.name+'_resume_metadata.json')
            saved=json.loads((old/f'checkpoint_{step}_audit.json').read_text())
            hashes={x.name:p.sha(x) for x in (cp,rp,mp)}
            require(hashes==saved['hashes'],'save-time hash mismatch '+str(step))
            # Validate against the historical training source binding, NOT the repaired current-source gate.
            meta=f.bundle(cp,'e6fea100',cfg,gate)
            state=msgpack_restore(cp.read_bytes())
            arrays=list(leaves(state));require(bool(arrays) and all(np.isfinite(a).all() for a in arrays),'model/optimizer finite')
            require('params' in state and 'opt_states' in state and 'rng' in state,'state structure')
            require(np.asarray(state['rng']).tolist()==rows[step]['rng_after'],'learner RNG vs monitor')
            replay=pickle.loads(rp.read_bytes());size=replay['size']
            require(all(len(a)==size and np.isfinite(a).all() for a in replay['dataset_dict'].values()),'replay content finite/size')
            require(replay['dataset_dict']['observations'].shape==(size,20),'observation shape')
            require(replay['dataset_dict']['actions'].shape==(size,7),'action shape')
            rng=replay['rng_state']
            if rng['kind']=='generator':
                cls=getattr(np.random,rng['state']['bit_generator']);probe=cls();probe.state=rng['state']
            else:
                probe=np.random.RandomState();probe.set_state(rng['state'])
            bundles.append(dict(step=step,status='PASS',state_leaves=len(arrays),state_keys=list(state),replay_size=size,
                                insert_index=replay['insert_index'],actor_steps=replay['actor_steps'],hashes=hashes,
                                learner_rng_matches_monitor=True,replay_rng_valid=True,historical_binding='PASS'))
    except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
    history={str(step):[x.name for x in cpdir.glob(f'checkpoint_{step}*')] for step in (0,1000,2000)}
    after={str(x.relative_to(ROOT)):p.sha(x) for x in protected}
    if before!=after:errors.append('protected bytes changed')
    passed=not errors and len(bundles)==7
    result=dict(PEG_F2_SURVIVING_CHECKPOINT_AUDIT='PASS' if passed else 'FAIL',
                PEG_FORMAL_TRAINING_RAW_GATE='FAIL',PEG_FORMAL_CHECKPOINT_RETENTION='FAIL_HISTORICAL_EARLY_LOSS',
                PEG_FORMAL_WANDB_FINALIZATION='FAIL_KNOWN_BUG',
                PEG_FORMAL_TRAINING_ADJUDICATION='PASS_WITH_DISCLOSED_ARTIFACT_AND_FINALIZATION_LIMITATIONS' if passed else 'BLOCKED',
                PEG_FORMAL_EVALUATION='READY' if passed else 'BLOCKED',evaluation_scope='Surviving hash-bound checkpoints only; separate execution authorization and evaluator preflight required',
                run_id='e6fea100',cloud_state='failed (preserved evidence; no new cloud query)',
                eligible_steps=[x['step'] for x in bundles],recommended_initial_checkpoint=9000 if passed else None,
                latest_checkpoint_is_not_final_9999=True,not_evaluated=True,errors=errors,bundles=bundles,early_history=history,
                protected_files_unchanged=before==after,training_rerun=False,checkpoint_resumed=False,cloud_modified=False,evaluation_started=False)
    (out/'protected_before.json').write_text(json.dumps(before,indent=2))
    (out/'protected_after.json').write_text(json.dumps(after,indent=2))
    (out/'adjudication.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
