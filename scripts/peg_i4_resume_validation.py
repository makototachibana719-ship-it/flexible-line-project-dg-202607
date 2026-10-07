#!/usr/bin/env python3
"""One-shot I4: source-locked I3 resume, five updates, one isolated bundle.

Run --role supervise with locked GPU Python and SERL/franka_sim PYTHONPATH.
No formal launcher, retry, algorithm changes, or old-bundle overwrites.
Actor environment/action RNG are fresh: this is not trajectory-exact resume.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import peg_i2_wandb_integration as i2

i1, base, ROOT = i2.i1, i2.base, i2.ROOT
SOURCE = ROOT / 'results/diagnostics/peg_i3_checkpoint_20260904_132008'
CP = SOURCE / 'checkpoints/checkpoint_19'
MANIFEST = ROOT / 'deliverables/gpu_training_handoff/peg_insertion/demo_manifest.json'
HASHES = {
    'checkpoint_19': 'aacc7116b24fc162b57701bc9f1c7b8403d2042647084ea1d0183e481c7b0579',
    'checkpoint_19_online_replay.pkl': '73bb5d9f915b92f146d1d712a5a6817faf927455b0c4815d4058a9f9f9f52ace',
    'checkpoint_19_resume_metadata.json': 'cbe97c02b793b41894cd936228b039cc4f153936632587dd045115fc4f3f99e6',
}
RUN_ID = '44yrxufd'


def source_bundle():
    sys.path.insert(0, str(ROOT))
    from training.peg_insertion.resume import load_bundle
    for name, expected in HASHES.items():
        if base.sha256(CP.parent / name) != expected:
            raise ValueError('source hash mismatch: ' + name)
    cfg = json.loads((SOURCE / 'bundle_config.json').read_text())
    meta = load_bundle(CP, cfg, MANIFEST, RUN_ID)
    payload = pickle.loads(meta['online_replay_path'].read_bytes())
    assert meta['actor_steps'] == payload['actor_steps'] == payload['size'] - 121 == 1171
    assert payload['insert_index'] == payload['size'] == 1292
    return cfg, meta, payload


def exact_tree(left, right):
    """Structure/dtype/value equality, including all optimizer and RNG leaves."""
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(exact_tree(a,b) for a,b in zip(left,right))
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a,b)


def restored_agent(factory, kwargs, out, role):
    import jax
    from flax.serialization import msgpack_restore, to_state_dict
    from training.peg_insertion.checkpoint import restore_fresh_state
    target = factory(**kwargs)
    state = restore_fresh_state(CP.parent, target.state, 19)
    raw = msgpack_restore(CP.read_bytes())
    equal = exact_tree(to_state_dict(state), raw)
    leaves = jax.tree_util.tree_leaves(state)
    audit = dict(full_state_equal=equal, leaves=len(leaves), rng=base.rng_value(jax,state),
                 state_step=int(np.asarray(state.step)), optimizer_restored=True,
                 leaf_signature=[dict(type=type(x).__name__, shape=list(np.shape(x)), dtype=str(np.asarray(x).dtype)) for x in leaves])
    base.write_json(out / (role + '_restore.json'), audit)
    assert equal and len(leaves) == 198
    return target.replace(state=state)


def actor(out):
    source_bundle()
    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    import async_sac_state_sim as delivered
    factory = delivered.make_sac_agent
    delivered.make_sac_agent = lambda **kw: restored_agent(factory, kw, out, 'actor')
    # Existing real environment/client loop, bounded by learner stop marker.
    i1.actor(out)


def learner(out):
    import jax
    import jaxlib
    from flax.serialization import to_state_dict
    from agentlace.trainer import TrainerServer
    from serl_launcher.utils.launcher import make_replay_buffer
    cfg, meta, payload = source_bundle()
    sys.path[:0] = [str(ROOT), str(base.EXAMPLE), str(base.ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent, concat_rlpd_batches
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.resume import restore_replay, capture_replay_snapshot, save_bundle
    assert jax.default_backend() == 'gpu' and jax.__version__ == jaxlib.__version__ == '0.4.35'
    base.write_json(out/'bundle_config.json', cfg)
    server = run = None
    success = False
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode='rgb_array', max_episode_steps=500)
        try:
            agent = restored_agent(make_sac_agent, dict(seed=0, sample_obs=env.observation_space.sample(),
                                   sample_action=env.action_space.sample()), out, 'learner')
            online = make_replay_buffer(env, capacity=payload['capacity'], type='replay_buffer')
            restore_info = restore_replay(online, meta['online_replay_path'])
            captured = capture_replay_snapshot(online)
            replay_equal = exact_tree(captured, payload)
            base.write_json(out/'replay_restore.json', dict(**restore_info, full_replay_equal=replay_equal,
                            replay_rng_equal=exact_tree(captured['rng_state'],payload['rng_state']),
                            rng_state=repr(captured['rng_state'])))
            assert replay_equal
            demo_path = ROOT / cfg['demo']['path']
            assert base.sha256(demo_path) == cfg['demo']['sha256']
            demos = pickle.loads(demo_path.read_bytes())
            assert len(demos) == 5512
            demo = make_replay_buffer(env, capacity=max(10000,len(demos)+100), type='replay_buffer')
            for transition in demos: demo.insert(transition)
        finally:
            env.close()
        base.write_json(out/'preflight.json', dict(pid=os.getpid(),jax=jax.__version__,jaxlib=jaxlib.__version__,
                        backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],
                        source=str(CP),global_updates=list(range(20,25)),normal_rng=True,
                        actor_steps_start=1171,actor_environment_restored=False,actor_sampling_rng_restored=False))
        try:
            os.environ['WANDB_MODE'] = 'online'
            import wandb
            run, seconds = i2.timed(out,'resume_init',lambda: wandb.init(entity=i2.ENTITY,project=i2.PROJECT,
                  id=RUN_ID,resume='must',mode='online',dir=str(out),
                  settings=wandb.Settings(init_timeout=60,save_code=False)))
            assert run.id == RUN_ID and run.resumed and run.settings.mode == 'online'
            base.write_json(out/'wandb_run.json',dict(id=run.id,path=run.path,url=run.url,resumed=run.resumed,init_seconds=seconds))
            live = [0]
            def event(value):
                base.append_jsonl(out/'communication.jsonl',dict(value,timestamp_ns=time.time_ns()))
            def received(name,data):
                live[0] += len(data['data'])
                event(dict(event='live_insert',count=len(data['data']),actor_steps=int(online.project_actor_steps)))
            server = TrainerServer(i1.configure(),data_callback=received,request_callback=lambda kind,data: {})
            server.register_data_store('actor_env',online)
            server.start(threaded=True)
            base.write_json(out/'server_ready.json',dict(pid=os.getpid()))
            deadline = time.monotonic()+90
            while live[0] < 31:
                if time.monotonic()>deadline: raise TimeoutError('actor startup')
                time.sleep(.05)
            server.publish_network(agent.state.params)
            event(dict(event='publish_initial'))
            sharding = jax.sharding.PositionalSharding(jax.local_devices()).replicate()
            oi = online.get_iterator(sample_args={'batch_size':32},device=sharding)
            di = demo.get_iterator(sample_args={'batch_size':32},device=sharding)
            contract = None
            for index in range(20,25):
                start = time.perf_counter()
                batch = concat_rlpd_batches(next(oi),next(di))
                sample_seconds = time.perf_counter()-start
                current = base.batch_contract(batch)
                assert current['observations']['shape']==[64,20] and current['actions']['shape']==[64,7]
                assert contract is None or current == contract
                contract = current
                before = base.rng_value(jax,agent.state)
                base.append_jsonl(out/'updates.jsonl',dict(event='start',index=index,rng_before=before,timestamp_ns=time.time_ns()))
                phase = {'value':'dispatch'}
                watch = threading.Timer(30,base.snapshot_30s,args=(out,index,phase))
                watch.daemon=True; watch.start()
                def update():
                    begin=time.perf_counter()
                    updated, info = agent.update_high_utd(batch,utd_ratio=1)
                    dispatch=time.perf_counter()-begin
                    phase['value']='full_block_until_ready'
                    sync_start=time.perf_counter()
                    base.block_tree(jax,(updated,info))
                    return updated,info,dispatch,time.perf_counter()-sync_start
                try:
                    (updated,info,dispatch,sync),total = i2.timed(out,'update_'+str(index),update)
                finally: watch.cancel()
                loss = base.losses(jax,info)
                after=base.rng_value(jax,updated.state)
                assert loss and all(np.isfinite(v) for v in loss.values()) and before!=after
                agent=updated
                record=dict(event='complete',index=index,sample_seconds=sample_seconds,dispatch_seconds=dispatch,
                            sync_seconds=sync,true_update_seconds=total,rng_before=before,rng_after=after,
                            losses=loss,losses_finite=True,timestamp_ns=time.time_ns())
                base.append_jsonl(out/'updates.jsonl',record)
                metrics=dict(learner_update=index,dispatch_seconds=dispatch,sync_seconds=sync,true_update_seconds=total)
                for key,value in loss.items():
                    for name in ('critic_loss','actor_loss','temperature_loss'):
                        if name in key: metrics[name]=value
                assert all(k in metrics for k in ('critic_loss','actor_loss','temperature_loss'))
                _,log_seconds=i2.timed(out,'log',lambda:run.log(metrics,step=index,commit=True))
                base.append_jsonl(out/'wandb_calls.jsonl',dict(index=index,metrics=metrics,log_seconds=log_seconds))
                server.publish_network(agent.state.params)
                event(dict(event='publish_update',index=index))
                print('UPDATE_COMPLETE:',json.dumps(record),flush=True)
            directory=out/'checkpoints'; directory.mkdir(exist_ok=False)
            source=dict(identifier='peg-i4-five-update-resume',profile='smoke',diagnostic_enabled=False,
                        parent_checkpoint=str(CP),parent_hashes=HASHES,initial_replay_transitions=121,
                        inherited_actor_steps=1171,actor_steps_semantics='cumulative accepted live transitions',
                        save_reason='requested terminal I4 bundle after five complete updates')
            cp,seconds=i2.timed(out,'checkpoint_save',lambda:save_bundle(directory,agent.state,online,24,
                    int(online.project_actor_steps),cfg,MANIFEST,RUN_ID,source))
            from flax.serialization import msgpack_restore
            assert exact_tree(to_state_dict(agent.state),msgpack_restore(cp.read_bytes()))
            base.write_json(out/'save.json',dict(checkpoint=str(cp),seconds=seconds,full_state_equal=True,
                            live_accepted=live[0],actor_steps_live=int(online.project_actor_steps),batch_contract=contract))
            success=True
        finally:
            base.write_json(out/'stop_actor.json',dict(stop=True))
            if server is not None: server.stop()
            if run is not None:
                _,secs=i2.timed(out,'finish',lambda:run.finish(exit_code=0 if success else 1))
                base.write_json(out/'wandb_finish.json',dict(success=success,seconds=secs))


def verify(out):
    import wandb
    from flax.serialization import msgpack_restore
    cfg,meta,payload=source_bundle()
    from training.peg_insertion.resume import load_bundle
    updates=[x for x in i1.read_events(out/'updates.jsonl') if x['event']=='complete']
    cp=out/'checkpoints/checkpoint_24'
    new=load_bundle(cp,cfg,MANIFEST,RUN_ID)
    replay=pickle.loads(new['online_replay_path'].read_bytes())
    raw=msgpack_restore(cp.read_bytes())
    def arrays(tree):
        if isinstance(tree,dict):
            for v in tree.values(): yield from arrays(v)
        else: yield np.asarray(tree)
    leaves=list(arrays(raw))
    remote=wandb.Api(timeout=30).run(i2.ENTITY+'/'+i2.PROJECT+'/'+RUN_ID)
    rows=list(remote.scan_history(keys=['_step','learner_update','critic_loss','actor_loss','temperature_loss'],page_size=100))
    calls=i1.read_events(out/'wandb_calls.jsonl')
    online_ok=remote.state=='finished' and [r['_step'] for r in rows]==list(range(25)) and [r['learner_update'] for r in rows]==list(range(25))
    online_ok=online_ok and len(calls)==5 and all(math.isclose(r[k],c['metrics'][k],rel_tol=1e-7,abs_tol=1e-9)
        for r,c in zip(rows[20:],calls) for k in ('critic_loss','actor_loss','temperature_loss'))
    base.write_json(out/'wandb_remote.json',dict(state=remote.state,rows=rows,pass_check=online_ok))
    def read(name): return json.loads((out/name).read_text())
    restore=read('learner_restore.json'); rr=read('replay_restore.json')
    comm=i1.read_events(out/'communication.jsonl')
    checks=dict(updates_5=[x['index'] for x in updates]==list(range(20,25)),
        restore=restore['full_state_equal'] and rr['full_replay_equal'] and rr['replay_rng_equal'],
        rng_start=updates[0]['rng_before']==restore['rng'],
        rng_progression=all(a['rng_after']==b['rng_before'] for a,b in zip(updates,updates[1:])),
        rng_saved=np.asarray(raw['rng']).tolist()==updates[-1]['rng_after'],
        finite=len(leaves)==198 and all(np.all(np.isfinite(a)) for a in leaves),
        counters=new['actor_steps']==replay['actor_steps']==replay['size']-121,
        index=new['online_replay_size']==replay['size']==replay['insert_index'],
        new_live=new['actor_steps']>1171,old_prefix=all(np.array_equal(replay['dataset_dict'][k][:1292],v) for k,v in payload['dataset_dict'].items()),
        replay_rng_advanced=not exact_tree(replay['rng_state'],payload['rng_state']),
        wandb=online_ok,full_saved_state=read('save.json')['full_state_equal'],
        no_timeout=not (out/'wandb_timeout.json').exists(),
        publications=sum(x['event']=='publish_update' for x in comm)==5,
        live_overlap=any(x['event']=='live_insert' and updates[0]['timestamp_ns']<x['timestamp_ns']<updates[-1]['timestamp_ns'] for x in comm),
        actor_networks=bool(i1.read_events(out/'actor_network.jsonl')))
    base.write_json(out/'verification.json',dict(checks=checks,passed=all(checks.values()),
        actor_steps=new['actor_steps'],replay_size=replay['size'],rng=np.asarray(raw['rng']).tolist(),
        files={p.name:base.sha256(p) for p in cp.parent.glob('checkpoint_24*')}))
    if not all(checks.values()): raise RuntimeError('I4 audit failed')


def supervise():
    source_bundle()
    for port in (5595,5596):
        with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    out=ROOT/'results/diagnostics'/time.strftime('peg_i4_resume_%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    print('I4_OUTPUT:',out,flush=True)
    env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_enable_triton_gemm=false',
             MUJOCO_GL='egl',JAX_LOG_COMPILES='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    children=[]; streams=[]; error=None
    def launch(role):
        stream=(out/(role+'.log')).open('w');streams.append(stream)
        p=subprocess.Popen([sys.executable,'-u',__file__,'--role',role,'--output',str(out)],env=env,
                           stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    try:
        learn=launch('learner');deadline=time.monotonic()+300
        while not (out/'server_ready.json').exists():
            if learn.poll() is not None or time.monotonic()>deadline:raise RuntimeError('learner failed before server ready')
            time.sleep(.2)
        act=launch('actor')
        while learn.poll() is None:
            if act.poll() is not None and not (out/'stop_actor.json').exists():raise RuntimeError('actor exited prematurely')
            if time.monotonic()>deadline:raise TimeoutError('I4 deadline')
            time.sleep(1)
        act.wait(timeout=15)
        if learn.returncode or act.returncode:raise RuntimeError('learner/actor failure')
        check=launch('verify');check.wait(timeout=60)
        if check.returncode:raise RuntimeError('verification failure')
    except Exception as exc: error=repr(exc)
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in streams:f.close()
    unchanged=all(base.sha256(CP.parent/name)==h for name,h in HASHES.items())
    result=dict(status='PASS' if error is None and unchanged else 'FAIL',error=error,source_unchanged=unchanged,
                child_pids=[p.pid for p in children],exit_codes=[p.returncode for p in children],
                residual_owned_processes=[p.pid for p in children if p.poll() is None],retry=False,formal_training='BLOCKED')
    base.write_json(out/'i4_result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    print('PEG_I4_RESUME_LIFECYCLE_5='+result['status'],flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=['supervise','learner','actor','verify'],required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.role=='supervise':raise SystemExit(supervise())
    if args.output is None or ROOT not in args.output.resolve().parents:raise ValueError('invalid output')
    {'learner':learner,'actor':actor,'verify':verify}[args.role](args.output)
