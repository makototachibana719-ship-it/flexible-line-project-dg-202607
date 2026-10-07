"""Bounded F0-R policy and read-only restore assertions; no training loop."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / 'results/validation/peg/f0_sync_20260904'
SOURCE = SOURCE_ROOT / 'checkpoints/checkpoint_25'
OUT = ROOT / 'results/validation/peg/f0_r_branch_20260904'
SOURCE_RUN = 'hr88a6u1'
RUN_NAME = 'peg_rlpd_gpu_f0_resume_from25_seed0'
HASHES = {
    'checkpoint_25': 'd12e282e291bc4cedd6ae58d1d1bba4ab6b300326fe1d742bd6a436d7e3f1333',
    'checkpoint_25_online_replay.pkl': '95f0ae051a00b894f749f05e24adc906e7fb71b3680433340cfcb430564caf25',
    'checkpoint_25_resume_metadata.json': '9124495baaec2e418c5b5f4bf7f6b2e600b00ec91059e15600de25489ee4eca4',
}

def provenance():
    return dict(validation_type='exact_path_resume_branch', resume_branch=True,
                source_run_id=SOURCE_RUN, source_checkpoint_path=str(SOURCE),
                source_checkpoint_step=25, source_learner_update=25,
                same_run_history_continuation=False)

def source_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['formal_candidate'].update(learner_updates=50, checkpoint_period=25,
        wandb_run_name='peg_f0_exact_path_seed0', checkpoint_dir=str(SOURCE_ROOT/'checkpoints'),
        log_dir=str(SOURCE_ROOT/'logs'), result_dir=str(SOURCE_ROOT/'learner'))
    cfg['wandb']['run_name'] = 'peg_f0_exact_path_seed0'
    return cfg

def validate_source(path, run_id):
    if path is None or path.resolve() != SOURCE or run_id != SOURCE_RUN:
        raise ValueError('F0-R requires exact approved B checkpoint25/source run')
    for name, expected in HASHES.items():
        if hashlib.sha256((SOURCE.parent/name).read_bytes()).hexdigest() != expected:
            raise ValueError('source artifact hash mismatch: '+name)
    audit = json.loads((ROOT/'results/diagnostics/peg_f0_verifier_fix_20260904/reaudit.json').read_text())
    if audit.get('status') != 'PASS' or audit.get('wandb_run_id') != SOURCE_RUN:
        raise ValueError('approved B re-audit missing')

def write_evidence(name, data):
    with (OUT/name).open('x') as stream:
        json.dump(data, stream, indent=2)

def assert_equal_tree(actual, expected, path='state'):
    import numpy as np
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError('structure mismatch: '+path)
        return sum(assert_equal_tree(actual[k], v, path+'.'+str(k)) for k,v in expected.items())
    if isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError('sequence mismatch: '+path)
        return sum(assert_equal_tree(a,b,path+'.'+str(i)) for i,(a,b) in enumerate(zip(actual,expected)))
    a,b = np.asarray(actual),np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a,b):
        raise AssertionError('value mismatch: '+path)
    return 1

def audit_state(state, label):
    import numpy as np
    from flax.serialization import to_state_dict, msgpack_restore
    raw = msgpack_restore(SOURCE.read_bytes())
    leaves = assert_equal_tree(to_state_dict(state), raw)
    if leaves != 198:
        raise AssertionError('unexpected train-state leaf count')
    result = dict(status='PASS', leaves=leaves, rng=np.asarray(state.rng).tolist(),
                  model_optimizer_rng_exact=True)
    write_evidence(label+'.json', result)
    return result

def audit_replay(buf):
    from training.peg_insertion.resume import capture_replay_snapshot
    expected = pickle.loads(SOURCE.with_name(SOURCE.name+'_online_replay.pkl').read_bytes())
    snapshot = capture_replay_snapshot(buf)
    assert_equal_tree(snapshot, expected, 'replay')
    if not snapshot['size'] == snapshot['insert_index'] == snapshot['actor_steps'] == 1261:
        raise AssertionError('source replay counter invariant')
    write_evidence('replay_restore.json', dict(status='PASS',size=snapshot['size'],
        insert_index=snapshot['insert_index'],actor_steps=snapshot['actor_steps'],
        contents_exact=True,replay_rng_exact=True))

def actor_range(start, count):
    """Continue environment-step scheduling, without replaying random warmup."""
    return range(start, start+count)

def state_fingerprint(state):
    import numpy as np
    from flax.serialization import to_state_dict
    tree = to_state_dict(state)
    result = {}
    def visit(value,path):
        if isinstance(value,dict):
            for key,item in value.items():visit(item,path+'/'+str(key))
        else:
            array=np.asarray(value)
            result[path]=dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())
    visit(tree,'')
    return result
