#!/usr/bin/env python3
"""One-shot I1 diagnostic. Reuses R2; adds live Agentlace transport only.

Run with the locked serl_peg_gpu Python: this_file.py --role supervise.
Outputs only into a newly created diagnostics directory. Never invokes a launcher.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import peg_runtime_continuous_update_probe as baseline

ROOT = baseline.ROOT
BATCH = ROOT / 'results/diagnostics/peg_jax_xla_differential_20260903/fixed_batch.npz'


def configure():
    from serl_launcher.utils.launcher import make_trainer_config
    return dataclasses.replace(make_trainer_config(), port_number=5595, broadcast_port=5596)


def learner(out):
    import jax
    from agentlace.trainer import TrainerServer
    sys.path[:0] = [str(ROOT), str(baseline.EXAMPLE), str(baseline.ENV_PARENT)]
    import async_sac_state_sim as delivered
    original_factory = delivered.make_sac_agent
    original_populate = baseline.populate_dynamic_buffers
    original_block = baseline.block_tree
    original_append = baseline.append_jsonl
    state = {}
    lock = threading.Lock()

    def event(value):
        value['timestamp_ns'] = time.time_ns()
        with lock:
            original_append(out / 'communication.jsonl', value)

    def factory(**kwargs):
        state['agent'] = original_factory(**kwargs)
        return state['agent']

    def populate(environment, make_buffer):
        online, demo, metadata = original_populate(environment, make_buffer)
        def received(name, payload):
            event({'event': 'live_insert', 'count': len(payload['data']), 'online_size': len(online)})
            state['live'] = state.get('live', 0) + len(payload['data'])
        server = TrainerServer(configure(), data_callback=received,
                               request_callback=lambda kind, payload: {})
        state['server'] = server
        server.register_data_store('actor_env', online)
        server.start(threaded=True)
        baseline.write_json(out / 'server_ready.json', {'pid': os.getpid()})
        deadline = time.monotonic() + 90
        while state.get('live', 0) < 31:
            if time.monotonic() > deadline:
                raise TimeoutError('live actor data startup deadline')
            time.sleep(.05)
        server.publish_network(state['agent'].state.params)
        event({'event': 'publish_initial'})
        metadata['live_transitions_before_updates'] = state['live']
        metadata['online_size_after_live_start'] = len(online)
        return online, demo, metadata

    def block(jax_module, tree):
        original_block(jax_module, tree)
        state['updated_agent'] = tree[0]

    def append(path, value):
        original_append(path, value)
        if value.get('event') == 'complete':
            start = time.perf_counter()
            state['server'].publish_network(state['updated_agent'].state.params)
            event({'event': 'publish_update', 'index': value['index'],
                   'publish_seconds': time.perf_counter() - start})

    delivered.make_sac_agent = factory
    baseline.populate_dynamic_buffers = populate
    baseline.block_tree = block
    baseline.append_jsonl = append
    try:
        assert jax.default_backend() == 'gpu'
        baseline.run('R2', BATCH, out)
    finally:
        baseline.write_json(out / 'stop_actor.json', {'stop': True})
        if 'server' in state:
            state['server'].stop()


def actor(out):
    import jax
    import numpy as np
    from agentlace.data.data_store import QueuedDataStore
    from agentlace.trainer import TrainerClient
    sys.path[:0] = [str(ROOT), str(baseline.EXAMPLE), str(baseline.ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode='rgb_array', max_episode_steps=500)
        agent = make_sac_agent(seed=0, sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        sampling_rng = jax.random.split(jax.random.PRNGKey(0))[1]
        store = QueuedDataStore(2000)
        client = TrainerClient('actor_env', '127.0.0.1', configure(), store, wait_for_server=False)
        count = [0]
        def params_received(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))
            count[0] += 1
            baseline.append_jsonl(out / 'actor_network.jsonl',
                                  {'event': 'received_params', 'timestamp_ns': time.time_ns()})
        client.recv_network_callback(params_received)
        obs, _ = env.reset()
        start = time.monotonic()
        steps = 0
        try:
            while not (out / 'stop_actor.json').exists():
                if steps >= 3000 or time.monotonic() - start > 180:
                    raise TimeoutError('bounded actor limit before learner completion')
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = np.asarray(jax.device_get(agent.sample_actions(
                    observations=jax.device_put(obs), seed=key, deterministic=False)))
                next_obs, reward, terminated, truncated, _ = env.step(actions)
                store.insert(dict(observations=obs, actions=actions,
                                  next_observations=np.asarray(next_obs, dtype=np.float32),
                                  rewards=np.asarray(reward, dtype=np.float32),
                                  masks=np.float32(0 if terminated else 1),
                                  dones=bool(terminated or truncated)))
                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset()
                if steps % 30 == 0 and not client.update():
                    raise RuntimeError('actor upload failed; no retry')
                steps += 1
        finally:
            baseline.write_json(out / 'actor_result.json', dict(
                pid=os.getpid(), steps=steps, received_networks=count[0],
                backend=jax.default_backend(), stopped_by_learner=(out / 'stop_actor.json').exists()))
            client.stop()
            env.close()


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def supervise():
    if baseline.sha256(BATCH) != baseline.EXPECTED_FIXED_BATCH_SHA256:
        raise ValueError('R2 fixed batch integrity check failed')
    for port in (5595, 5596):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    out = ROOT / 'results/diagnostics' / time.strftime('peg_i1_live_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    print('I1_OUTPUT:', out, flush=True)
    protected = list(baseline.CANDIDATE.parent.glob('checkpoint_0*'))
    before = {str(p): baseline.sha256(p) for p in protected if p.is_file()}
    baseline.write_json(out / 'protected_before.json', before)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_enable_triton_gemm=false', MUJOCO_GL='egl',
               JAX_LOG_COMPILES='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
    children = []
    streams = []
    error = None
    def launch(role):
        stream = (out / (role + '.log')).open('w')
        streams.append(stream)
        child = subprocess.Popen([sys.executable, '-u', __file__, '--role', role,
                                  '--output', str(out)], env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    monitor_stream = (out / 'gpu_dmon.log').open('w')
    monitor = subprocess.Popen(['nvidia-smi', 'dmon', '-d', '1'], stdout=monitor_stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        learn = launch('learner')
        deadline = time.monotonic() + 300
        while not (out / 'server_ready.json').exists():
            if learn.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('learner failed before server ready')
            time.sleep(.2)
        act = launch('actor')
        while learn.poll() is None:
            with (out / 'cpu_threads.log').open('a') as stream:
                stream.write(str(time.time_ns()) + '\n')
                subprocess.run(['ps', '-L', '-p', f'{learn.pid},{act.pid}', '-o',
                                'pid,tid,pcpu,stat,wchan:32,comm'], stdout=stream, check=False)
            if act.poll() is not None and not (out / 'stop_actor.json').exists():
                raise RuntimeError('actor exited before learner')
            if time.monotonic() > deadline:
                raise TimeoutError('I1 process deadline')
            time.sleep(1)
        act.wait(timeout=15)
        if learn.returncode or act.returncode:
            raise RuntimeError(f'child exit codes: {learn.returncode}, {act.returncode}')
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
        for stream in streams + [monitor_stream]:
            stream.close()
    updates = read_events(out / 'updates.jsonl')
    complete = [x for x in updates if x['event'] == 'complete']
    comm = read_events(out / 'communication.jsonl')
    networks = read_events(out / 'actor_network.jsonl')
    start_ns = updates[0]['timestamp_ns'] if updates else 0
    end_ns = complete[-1]['timestamp_ns'] if complete else 0
    overlap = [x for x in comm if x['event'] == 'live_insert' and start_ns < x['timestamp_ns'] < end_ns]
    unchanged = all(baseline.sha256(Path(p)) == h for p, h in before.items())
    checks = dict(updates_20=len(complete) == 20,
                  finite=bool(complete) and all(x['losses_finite'] for x in complete),
                  normal_rng=bool(complete) and all(x['rng_before'] != x['rng_after'] for x in complete)
                  and all(a['rng_after'] == b['rng_before'] for a, b in zip(complete, complete[1:])),
                  live_insert_overlap=bool(overlap), network_received=bool(networks),
                  publishes_20=sum(x['event'] == 'publish_update' for x in comm) == 20,
                  protected_unchanged=unchanged, children_exited=all(p.poll() is not None for p in children))
    passed = error is None and all(checks.values())
    result = dict(status='PASS' if passed else 'FAIL', checks=checks, error=error,
                  completed_updates=len(complete), live_transitions=sum(x['count'] for x in comm if x['event']=='live_insert'),
                  overlapping_uploads=len(overlap), received_networks=len(networks),
                  child_pids=[p.pid for p in children], child_exit_codes=[p.returncode for p in children],
                  formal_training='BLOCKED', retry=False, wandb=False, checkpoint_writes=False)
    baseline.write_json(out / 'i1_result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print('PEG_I1_LIVE_COMMUNICATION_20=' + result['status'], flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['supervise', 'learner', 'actor'], required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.role == 'supervise':
        raise SystemExit(supervise())
    if args.output is None or ROOT not in args.output.resolve().parents:
        raise ValueError('output must be inside project')
    (learner if args.role == 'learner' else actor)(args.output)
