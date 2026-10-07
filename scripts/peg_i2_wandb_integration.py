#!/usr/bin/env python3
"""I2: one I1 run plus online W&B init/log/finish, never checkpoint saving.

Use locked serl_peg_gpu Python with SERL/franka_sim on PYTHONPATH.
Entry: --role supervise. All output goes into a new I2 diagnostic directory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import peg_i1_live_communication as i1
from analyze_peg_runtime_reliability import compile_counts

ROOT = i1.ROOT
base = i1.baseline
ENTITY = 'makototachibana719-beihang-university'
PROJECT = 'flexible-line-peg-insertion'
NAME = 'peg_rlpd_gpu_i2_wandb_seed0'


def timed(out, phase, operation):
    def expired():
        base.write_json(out / 'wandb_timeout.json', {'phase': phase, 'seconds': 120})
        os._exit(124)
    watch = threading.Timer(120, expired)
    watch.daemon = True
    watch.start()
    started = time.perf_counter()
    try:
        return operation(), time.perf_counter() - started
    finally:
        watch.cancel()


def learner(out):
    os.environ['WANDB_MODE'] = 'online'
    import wandb
    append_original = base.append_jsonl
    run, init_seconds = timed(out, 'init', lambda: wandb.init(
        entity=ENTITY, project=PROJECT, name=NAME, mode='online', resume='never',
        dir=str(out), settings=wandb.Settings(init_timeout=60, save_code=False),
        config={'integration_rung': 'I2', 'seed': 0, 'learner_updates': 20,
                'online_batch': 32, 'demo_batch': 32, 'rng': 'normal',
                'update_method': 'original SACAgent.update_high_utd',
                'checkpoint_writes': False, 'resume': False, 'formal': False}))
    if run.settings.mode != 'online':
        raise RuntimeError('W&B online required; no offline fallback')
    base.write_json(out / 'wandb_run.json', {'id': run.id, 'path': run.path,
                    'url': run.url, 'mode': run.settings.mode, 'init_seconds': init_seconds,
                    'wandb_version': wandb.__version__})
    def append(path, event):
        append_original(path, event)
        if path.name != 'updates.jsonl' or event.get('event') != 'complete':
            return
        metrics = {key: event[key] for key in (
            'index', 'dispatch_seconds', 'sync_seconds', 'true_update_seconds', 'sample_seconds')}
        metrics['learner_update'] = event['index']
        for name, value in event['losses'].items():
            for loss in ('critic_loss', 'actor_loss', 'temperature_loss'):
                if loss in name:
                    metrics[loss] = value
        _, seconds = timed(out, 'log', lambda: run.log(metrics, step=event['index'], commit=True))
        append_original(out / 'wandb_calls.jsonl', {'index': event['index'],
                        'log_seconds': seconds, 'timestamp_ns': time.time_ns(), 'metrics': metrics})
    base.append_jsonl = append
    success = False
    try:
        i1.learner(out)
        success = True
    finally:
        _, finish_seconds = timed(out, 'finish', lambda: run.finish(exit_code=0 if success else 1))
        base.write_json(out / 'wandb_finish.json', {'success': success, 'finish_seconds': finish_seconds})


def verify_online(out):
    import wandb
    info = json.loads((out / 'wandb_run.json').read_text())
    remote = wandb.Api(timeout=30).run(info['path'])
    keys = ['_step', 'learner_update', 'critic_loss', 'actor_loss', 'temperature_loss']
    rows = list(remote.scan_history(keys=keys, page_size=100))
    local = i1.read_events(out / 'wandb_calls.jsonl')
    good = len(rows) == 20 and [row['learner_update'] for row in rows] == list(range(20))
    good = good and [row['_step'] for row in rows] == list(range(20))
    good = good and len(local) == 20 and all(
        math.isfinite(row[key]) and math.isclose(row[key], call['metrics'][key], rel_tol=1e-7, abs_tol=1e-9)
        for row, call in zip(rows, local) for key in keys[2:])
    base.write_json(out / 'wandb_remote.json', {'state': remote.state, 'rows': rows,
                    'matches_local': good, 'pass': good and remote.state == 'finished'})


def supervise():
    for port in (5595, 5596):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    if base.sha256(i1.BATCH) != base.EXPECTED_FIXED_BATCH_SHA256:
        raise ValueError('R2 fixed batch changed')
    out = ROOT / 'results/diagnostics' / time.strftime('peg_i2_wandb_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    print('I2_OUTPUT:', out, flush=True)
    protected = [p for p in base.CANDIDATE.parent.glob('checkpoint_0*') if p.is_file()]
    before = {str(p): base.sha256(p) for p in protected}
    base.write_json(out / 'protected_before.json', before)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_enable_triton_gemm=false', MUJOCO_GL='egl',
               JAX_LOG_COMPILES='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
    children, streams = [], []
    error = None
    def launch(role):
        stream = (out / (role + '.log')).open('w')
        streams.append(stream)
        proc = subprocess.Popen([sys.executable, '-u', __file__, '--role', role, '--output', str(out)],
                                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc)
        return proc
    gpu_stream = (out / 'gpu_dmon.log').open('w')
    monitor = subprocess.Popen(['nvidia-smi', 'dmon', '-d', '1'], stdout=gpu_stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        learn = launch('learner')
        deadline = time.monotonic() + 300
        while not (out / 'server_ready.json').exists():
            if learn.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('learner failed before server ready; see learner.log')
            time.sleep(.2)
        act = launch('actor')
        while learn.poll() is None:
            with (out / 'cpu_threads.log').open('a') as stream:
                stream.write(str(time.time_ns()) + '\n')
                subprocess.run(['ps', '-L', '-p', f'{learn.pid},{act.pid}', '-o',
                                'pid,tid,pcpu,stat,wchan:32,comm'], stdout=stream, check=False)
            if act.poll() is not None and not (out / 'stop_actor.json').exists():
                raise RuntimeError('actor exited before learner finished updates')
            if time.monotonic() > deadline:
                raise TimeoutError('I2 process deadline')
            time.sleep(1)
        act.wait(timeout=15)
        if learn.returncode or act.returncode:
            raise RuntimeError(f'child failures: {learn.returncode}, {act.returncode}')
        verifier = launch('verify')
        verifier.wait(timeout=60)
        if verifier.returncode:
            raise RuntimeError('online verification failed; no repeated query')
    except Exception as exc:
        error = repr(exc)
    finally:
        for child in children + [monitor]:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for stream in streams + [gpu_stream]:
            stream.close()
    events = i1.read_events(out / 'updates.jsonl')
    done = [x for x in events if x['event'] == 'complete']
    comm = i1.read_events(out / 'communication.jsonl')
    networks = i1.read_events(out / 'actor_network.jsonl')
    remote = json.loads((out / 'wandb_remote.json').read_text()) if (out / 'wandb_remote.json').exists() else {}
    overlap = [x for x in comm if x['event'] == 'live_insert' and events and done
               and events[0]['timestamp_ns'] < x['timestamp_ns'] < done[-1]['timestamp_ns']]
    checks = dict(updates_20=[x['index'] for x in done] == list(range(20)),
                  finite=bool(done) and all(x['losses_finite'] for x in done),
                  rng=bool(done) and all(x['rng_before'] != x['rng_after'] for x in done)
                  and all(a['rng_after'] == b['rng_before'] for a,b in zip(done, done[1:])),
                  no_stall=bool(done) and all(x['true_update_seconds'] < 10 for x in done[1:]),
                  live_overlap=bool(overlap), networks_received=bool(networks),
                  publications_20=sum(x['event']=='publish_update' for x in comm)==20,
                  online_history=remote.get('pass', False),
                  protected_unchanged=all(base.sha256(Path(p))==h for p,h in before.items()),
                  children_exited=all(p.poll() is not None for p in children))
    status = 'PASS' if error is None and all(checks.values()) else 'FAIL'
    result = dict(status=status, checks=checks, error=error, updates=len(done),
                  live_transitions=sum(x['count'] for x in comm if x['event']=='live_insert'),
                  overlapping_uploads=len(overlap), received_networks=len(networks),
                  child_pids=[p.pid for p in children], exit_codes=[p.returncode for p in children],
                  compiles=compile_counts((out/'learner.log').read_text(errors='replace')),
                  formal_training='BLOCKED', checkpoint_writes=False, retry=False)
    base.write_json(out / 'i2_result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print('PEG_I2_WANDB_20=' + status, flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['supervise', 'learner', 'actor', 'verify'], required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.role == 'supervise':
        raise SystemExit(supervise())
    if args.output is None or ROOT not in args.output.resolve().parents:
        raise ValueError('output must be inside project')
    {'learner': learner, 'actor': i1.actor, 'verify': verify_online}[args.role](args.output)
