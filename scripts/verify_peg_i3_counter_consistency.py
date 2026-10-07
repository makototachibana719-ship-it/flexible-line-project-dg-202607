"""Read-only I3 bundle counter audit; does not import JAX or restore an agent."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
root = args.directory.resolve()
raw = json.loads((root / 'i3_result.json').read_text())
checks = []
for step in (0, 19):
    metadata = json.loads((root / 'checkpoints' / f'checkpoint_{step}_resume_metadata.json').read_text())
    initial = metadata['source_snapshot']['initial_replay_transitions']
    size = metadata['online_replay_size']
    capacity = metadata['online_replay_capacity']
    if size >= capacity:
        raise ValueError('counter invariant requires non-wrapped replay')
    expected = size - initial
    actual = metadata['actor_steps']
    checks.append(dict(step=step, initial_replay=initial, replay_size=size,
                       expected_live_steps=expected, recorded_actor_steps=actual,
                       undercount=expected-actual, passed=expected==actual))
status = 'PASS' if raw['status']=='PASS' and all(x['passed'] for x in checks) else 'FAIL'
audit = dict(raw_gate=raw['status'], audited_gate=status, counter_checks=checks,
             resume_executed=False, formal_training='BLOCKED',
             note='Raw gate checked metadata/sidecar agreement, not replay/live-counter invariant.')
with (root / 'i3_counter_audit.json').open('x') as stream:
    json.dump(audit, stream, indent=2)
print(json.dumps(audit, indent=2))
print('PEG_I3_CHECKPOINT_IO_20=' + status)
raise SystemExit(0 if status=='PASS' else 1)
