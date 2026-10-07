"""One authorized post-regression I3 run with stricter counter verification.

Preserves the original I3 harness and period; new outputs and W&B run ID.
"""
import argparse
import json
from pathlib import Path
import peg_i3_checkpoint_integration as i3

i3.NAME='peg_rlpd_gpu_i3_snapshot_fix_seed0'
i3.__file__=__file__  # Existing supervisor launches this role dispatcher.

def verify(out):
    i3.verify(out)
    bundles=json.loads((out/'checkpoint_verification.json').read_text())['bundles']
    results=[]
    for bundle in bundles:
        step=bundle['step']
        m=json.loads((out/'checkpoints'/f'checkpoint_{step}_resume_metadata.json').read_text())
        initial=m['source_snapshot']['initial_replay_transitions']
        valid=m['online_replay_size']<m['online_replay_capacity']
        valid=valid and m['actor_steps']==m['online_replay_size']-initial
        results.append(dict(step=step,replay_size=m['online_replay_size'],initial_replay=initial,
                            actor_steps=m['actor_steps'],pass_check=valid))
    passed=all(x['pass_check'] for x in results)
    i3.base.write_json(out/'i3_counter_audit.json',dict(passed=passed,bundles=results))
    if not passed:raise RuntimeError('I3 replay/live counter invariant failed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['supervise','learner','actor','verify'],required=True)
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.role=='supervise':raise SystemExit(i3.supervise())
    if args.output is None or i3.ROOT not in args.output.resolve().parents:raise ValueError('output outside project')
    {'learner':i3.learner,'actor':i3.i1.actor,'verify':verify}[args.role](args.output)
