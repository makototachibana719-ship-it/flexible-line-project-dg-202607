#!/usr/bin/env python3
"""One-shot I2 plus normal periodic bundle saving; absolutely no resume.

Entry: --role supervise, locked GPU Python and SERL PYTHONPATH as for I2.
Uses candidate checkpoint period 19, hence ordinary saves at steps 0 and 19.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import time

import peg_i2_wandb_integration as i2

i1, base, ROOT = i2.i1, i2.base, i2.ROOT
NAME = 'peg_rlpd_gpu_i3_checkpoint_seed0'
MANIFEST = ROOT / 'deliverables/gpu_training_handoff/peg_insertion/demo_manifest.json'


def learner(out):
    import yaml
    sys.path.insert(0, str(ROOT))
    from training.peg_insertion.resume import save_bundle, install_actor_step_tracking
    original_populate = base.populate_dynamic_buffers
    original_block = base.block_tree
    original_append = base.append_jsonl
    cfg = yaml.safe_load(base.CONFIG.read_text())
    period = cfg['formal_candidate']['checkpoint_period']
    assert period == 19
    cfg['wandb']['run_name'] = NAME
    cp_dir = out / 'checkpoints'
    cp_dir.mkdir(exist_ok=False)
    base.write_json(out / 'bundle_config.json', cfg)
    state = {}

    def populate(env, factory):
        online, demo, metadata = original_populate(env, factory)
        install_actor_step_tracking(online, 0)
        state['online'] = online
        return online, demo, metadata

    def block(jax, tree):
        original_block(jax, tree)
        state['train_state'] = tree[0].state

    def append(path, event):
        original_append(path, event)
        # I1 emits this only after the I2 log call and completed publication.
        # Never save inside update_high_utd or its synchronization wrapper.
        if path.name != 'communication.jsonl' or event.get('event') != 'publish_update':
            return
        index = event['index']
        if index % period:
            return
        run_info = json.loads((out / 'wandb_run.json').read_text())
        source = {'identifier': 'peg-i3-normal-periodic-io', 'profile': 'smoke',
                  'formal_candidate': False, 'diagnostic_enabled': False,
                  'checkpoint_period': period, 'initial_replay_transitions': 121,
                  'actor_steps_semantics': 'new live transitions accepted in this I3 run'}
        original_append(out / 'checkpoint_events.jsonl',
                        {'event': 'save_start', 'index': index, 'timestamp_ns': time.time_ns()})
        cp, seconds = i2.timed(out, 'checkpoint_save', lambda: save_bundle(
            cp_dir, state['train_state'], state['online'], index,
            int(state['online'].project_actor_steps), cfg, MANIFEST, run_info['id'], source))
        original_append(out / 'checkpoint_events.jsonl',
                        {'event': 'save_complete', 'index': index, 'checkpoint': str(cp),
                         'save_seconds': seconds, 'timestamp_ns': time.time_ns()})

    base.populate_dynamic_buffers = populate
    base.block_tree = block
    base.append_jsonl = append
    i2.NAME = NAME
    # Retain the tested I2 W&B mechanics; correct only the run's rung config.
    import wandb
    original_init = wandb.init
    def init(**kwargs):
        kwargs['config'].update(integration_rung='I3', checkpoint_writes=True,
                                checkpoint_period=period, resume=False)
        return original_init(**kwargs)
    wandb.init = init
    i2.learner(out)


def verify(out):
    import numpy as np
    from flax.serialization import msgpack_restore
    sys.path.insert(0, str(ROOT))
    from training.peg_insertion.resume import REQUIRED_METADATA, canonical_hash
    i2.verify_online(out)
    cfg = json.loads((out / 'bundle_config.json').read_text())
    run = json.loads((out / 'wandb_run.json').read_text())
    updates = [x for x in i1.read_events(out / 'updates.jsonl') if x['event']=='complete']
    saves = [x for x in i1.read_events(out / 'checkpoint_events.jsonl') if x['event']=='save_complete']
    def leaves(tree):
        if isinstance(tree, dict):
            for value in tree.values():
                yield from leaves(value)
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                yield from leaves(value)
        else:
            yield np.asarray(tree)
    bundles = []
    for step in (0,19):
        cp = out / 'checkpoints' / f'checkpoint_{step}'
        rp = cp.with_name(cp.name + '_online_replay.pkl')
        mp = cp.with_name(cp.name + '_resume_metadata.json')
        metadata = json.loads(mp.read_text())
        replay = pickle.loads(rp.read_bytes())
        raw = msgpack_restore(cp.read_bytes())  # decode only; no restore into an agent
        arrays = list(leaves(raw))
        finite = all(np.all(np.isfinite(a)) for a in arrays if np.issubdtype(a.dtype, np.number))
        rng = np.asarray(raw['rng'], dtype=np.uint32).tolist()
        checks = dict(required_metadata=REQUIRED_METADATA <= set(metadata),
                      step=metadata['checkpoint_step']==metadata['learner_update_steps']==step,
                      config=metadata['config_hash']==canonical_hash(cfg),
                      run=metadata['wandb_run_id']==run['id'] and metadata['wandb_name']==NAME,
                      demo=metadata['demo_sha256']==cfg['demo']['sha256'] and metadata['demo_transition_count']==5512,
                      replay_hash=metadata['online_replay_sha256']==base.sha256(rp),
                      replay_size=metadata['online_replay_size']==replay['size'],
                      actor_steps=metadata['actor_steps']==replay['actor_steps'],
                      finite=finite, rng=rng==updates[step]['rng_after'], leaves=len(arrays)==198)
        bundles.append(dict(step=step, checks=checks, passed=all(checks.values()),
                            rng=rng, leaves=len(arrays), replay_size=replay['size'],
                            actor_steps=metadata['actor_steps'],
                            files={p.name: {'bytes': p.stat().st_size, 'sha256': base.sha256(p)} for p in (cp,rp,mp)}))
    passed = [x['index'] for x in saves]==[0,19] and all(x['passed'] for x in bundles)
    base.write_json(out / 'checkpoint_verification.json',
                    {'pass': passed, 'bundles': bundles, 'resume_executed': False,
                     'validation': 'file decode only; no state injected into agent'})
    if not passed:
        raise RuntimeError('checkpoint file verification failed')


def supervise():
    for port in (5595,5596):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',port))
    if base.sha256(i1.BATCH)!=base.EXPECTED_FIXED_BATCH_SHA256:
        raise ValueError('R2 baseline batch changed')
    out = ROOT / 'results/diagnostics' / time.strftime('peg_i3_checkpoint_%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    print('I3_OUTPUT:',out,flush=True)
    before={str(p):base.sha256(p) for p in base.CANDIDATE.parent.glob('checkpoint_0*') if p.is_file()}
    base.write_json(out/'protected_before.json',before)
    env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_enable_triton_gemm=false',
             MUJOCO_GL='egl',JAX_LOG_COMPILES='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    children,streams=[],[]
    error=None
    def launch(role):
        stream=(out/(role+'.log')).open('w');streams.append(stream)
        p=subprocess.Popen([sys.executable,'-u',__file__,'--role',role,'--output',str(out)],
                           env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    gpu=(out/'gpu_dmon.log').open('w')
    monitor=subprocess.Popen(['nvidia-smi','dmon','-d','1'],stdout=gpu,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        learn=launch('learner');deadline=time.monotonic()+300
        while not (out/'server_ready.json').exists():
            if learn.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('learner failed before server ready')
            time.sleep(.2)
        act=launch('actor')
        while learn.poll() is None:
            with (out/'cpu_threads.log').open('a') as stream:
                stream.write(str(time.time_ns())+'\n')
                subprocess.run(['ps','-L','-p',f'{learn.pid},{act.pid}','-o','pid,tid,pcpu,stat,wchan:32,comm'],stdout=stream,check=False)
            if act.poll() is not None and not (out/'stop_actor.json').exists():
                raise RuntimeError('actor exited prematurely')
            if time.monotonic()>deadline:raise TimeoutError('I3 deadline')
            time.sleep(1)
        act.wait(timeout=15)
        if learn.returncode or act.returncode:raise RuntimeError('learner/actor failed')
        verification=launch('verify');verification.wait(timeout=60)
        if verification.returncode:raise RuntimeError('verification failed; no retry')
    except Exception as exc:error=repr(exc)
    finally:
        for p in children+[monitor]:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for stream in streams+[gpu]:stream.close()
    events=i1.read_events(out/'updates.jsonl');done=[x for x in events if x['event']=='complete']
    comm=i1.read_events(out/'communication.jsonl');net=i1.read_events(out/'actor_network.jsonl')
    def read(name):return json.loads((out/name).read_text()) if (out/name).exists() else {}
    overlap=[x for x in comm if x['event']=='live_insert' and events and done
             and events[0]['timestamp_ns']<x['timestamp_ns']<done[-1]['timestamp_ns']]
    checks=dict(updates_20=[x['index'] for x in done]==list(range(20)),finite=bool(done) and all(x['losses_finite'] for x in done),
                rng=bool(done) and all(x['rng_before']!=x['rng_after'] for x in done)
                and all(a['rng_after']==b['rng_before'] for a,b in zip(done,done[1:])),
                no_stall=bool(done) and all(x['true_update_seconds']<10 for x in done[1:]),
                live_overlap=bool(overlap),networks=bool(net),publications_20=sum(x['event']=='publish_update' for x in comm)==20,
                online_history=read('wandb_remote.json').get('pass',False),
                checkpoint_bundles=read('checkpoint_verification.json').get('pass',False),
                old_checkpoint_unchanged=all(base.sha256(Path(p))==h for p,h in before.items()))
    status='PASS' if error is None and all(checks.values()) else 'FAIL'
    result=dict(status=status,checks=checks,error=error,updates=len(done),live_transitions=sum(x['count'] for x in comm if x['event']=='live_insert'),
                overlapping_uploads=len(overlap),received_networks=len(net),pids=[p.pid for p in children],exit_codes=[p.returncode for p in children],
                compiles=i2.compile_counts((out/'learner.log').read_text(errors='replace')),
                resume_executed=False,formal_training='BLOCKED',retry=False)
    base.write_json(out/'i3_result.json',result)
    print(json.dumps(result,indent=2),flush=True);print('PEG_I3_CHECKPOINT_IO_20='+status,flush=True)
    return 0 if status=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=['supervise','learner','actor','verify'],required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.role=='supervise':raise SystemExit(supervise())
    if args.output is None or ROOT not in args.output.resolve().parents:raise ValueError('output outside project')
    {'learner':learner,'actor':i1.actor,'verify':verify}[args.role](args.output)
