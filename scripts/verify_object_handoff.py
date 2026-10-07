"""Verify Object GPU handoff inputs without credentials or training."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import jax, yaml
ROOT=Path(__file__).resolve().parents[1]; H=ROOT/'deliverables/gpu_training_handoff/object_grasping'
sys.path.insert(0,str(ROOT))
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): d.update(b)
 return d.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--require-gpu',action='store_true'); a=p.parse_args()
 required=[H/x for x in ('README.md','demo_manifest.json','environment_spec.md','baseline_config_snapshot.yaml','run_gpu_smoke_learner.sh','run_gpu_smoke_actor.sh','run_train_learner.sh','run_train_actor.sh','run_resume_learner.sh','run_resume_actor.sh','run_eval.sh','expected_outputs.md')]+[ROOT/'training/object_grasping/run_object_training.py',ROOT/'training/object_grasping/resume.py',ROOT/'evaluation/object_grasping/evaluate_policy.py',ROOT/'evaluation/object_grasping/resume_roundtrip.py']
 for x in required:
  if not x.is_file(): raise FileNotFoundError(x)
 m=json.loads((H/'demo_manifest.json').read_text()); d=ROOT/m['relative_path']
 if d.stat().st_size!=m['size_bytes'] or sha(d)!=m['sha256'] or m['raw_transitions']!=2000: raise AssertionError('Demo manifest mismatch')
 c=yaml.safe_load((H/'baseline_config_snapshot.yaml').read_text());
 if c['fresh_initialization'] is not True or c['load_legacy_checkpoint'] is not False: raise AssertionError('fresh policy')
 if c['wandb']['project']!='flexible-line-object-grasp' or c['wandb']['mode']!='online': raise AssertionError('W&B contract')
 from training.object_grasping.resume import REQUIRED_METADATA
 required_metadata={'checkpoint_step','learner_update_steps','actor_steps','seed','environment','algorithm','observation','action','demo_sha256','demo_transition_count','config_hash','wandb_project','wandb_name','wandb_run_id','online_replay_sha256','created_at','source_snapshot'}
 if REQUIRED_METADATA!=required_metadata: raise AssertionError('resume metadata schema mismatch')
 if not c['fresh_initialization'] or c['load_legacy_checkpoint']: raise AssertionError('fresh/resume policy mismatch')
 print('RESUME_BUNDLE_SCHEMA: PASS')
 for key in ('checkpoint_dir','log_dir','smoke_checkpoint_dir','smoke_log_dir'):
  x=ROOT/c['outputs'][key]
  if x.exists() and any(x.iterdir()): raise FileExistsError(f'non-empty output: {x}')
 sys.path.insert(0,str(ROOT/'external/packages/object_grasping/object_grasp_package/code/env'))
 from training.object_grasping.run_object_training import adapt, assert_contract
 import gym, franka_sim
 env=gym.make(c['environment']); native,_=env.reset(seed=c['seed']); assert_contract(adapt(env,native),env.action_space,c); env.close()
 print('OBSERVATION_ACTION_CONTRACT: PASS')
 backend=jax.default_backend(); print('JAX_DEVICES:',jax.devices()); print('JAX_BACKEND:',backend)
 if a.require_gpu and backend!='gpu': raise RuntimeError('GPU required')
 print('OBJECT_HANDOFF_VERIFY: PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
