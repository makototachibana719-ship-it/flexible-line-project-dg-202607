#!/usr/bin/env python3
"""Fail-fast Peg GPU handoff verifier; no training or credentials."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import jax,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];H=ROOT/'deliverables/gpu_training_handoff/peg_insertion';sys.path.insert(0,str(ROOT))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--require-gpu',action='store_true');a=p.parse_args()
 required=[H/x for x in ('README.md','demo_manifest.json','environment_spec.md','expected_outputs.md','run_gpu_smoke_learner.sh','run_gpu_smoke_actor.sh','run_train_learner.sh','run_train_actor.sh','run_resume_learner.sh','run_resume_actor.sh','run_eval.sh')]+[ROOT/x for x in ('configs/peg_insertion/peg_rlpd_baseline_seed0.yaml','configs/peg_insertion/environment_gpu.yaml','configs/peg_insertion/runtime_gpu_formal_candidate.env','training/peg_insertion/run_peg_training.py','training/peg_insertion/peg_replay_adapter.py','training/peg_insertion/peg_runtime_adapter.py','training/peg_insertion/checkpoint.py','training/peg_insertion/resume.py','evaluation/peg_insertion/evaluate_policy.py','evaluation/peg_insertion/resume_roundtrip.py','scripts/run_peg_formal_candidate_smoke.sh','scripts/verify_peg_formal_candidate.py')]
 for x in required:
  if not x.is_file():raise FileNotFoundError(x)
 manifest=json.loads((H/'demo_manifest.json').read_text());assets=manifest['assets']
 for item in assets:
  f=ROOT/item['relative_path']
  if f.stat().st_size!=item['size_bytes'] or sha(f)!=item['sha256']:raise AssertionError(f'Demo manifest mismatch: {f}')
 primary=[x for x in assets if x['role']=='primary_formal_training_data']
 if len(primary)!=1 or primary[0]['transition_count']!=5512 or primary[0]['episode_count']!=20:raise AssertionError('primary Demo identity')
 if len([x for x in assets if x['role']=='primary_provenance_episode' and x['success_status']=='VALIDATED_SUCCESS'])!=20:raise AssertionError('20/20 success provenance')
 c=yaml.safe_load((ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml').read_text())
 checks=[c['algorithm']=='RLPD_STYLE_SAC_PLUS_50_50_DEMO_REPLAY',c['fresh_initialization'] is True,c['load_legacy_checkpoint'] is False,c['observation']=={'shape':[20],'dtype':'float32'},c['action']=={'shape':[7],'dtype':'float32','low':-1.0,'high':1.0},c['training']['online_batch_size']==32,c['training']['demo_batch_size']==32,c['training']['total_batch_size']==64,c['training']['utd_ratio']==1,c['training']['training_starts']==100,c['training']['max_steps']==10000,c['training']['checkpoint_period']==1000,c['wandb']['project']=='flexible-line-peg-insertion',c['wandb']['mode']=='online',c['formal_candidate']['learner_updates']==20,c['formal_candidate']['update_timeout_seconds']==120,c['formal_candidate']['wandb_run_name']=='peg_rlpd_gpu_formal_candidate_seed0']
 if not all(checks):raise AssertionError('frozen config contract')
 runtime=(ROOT/'configs/peg_insertion/runtime_gpu_formal_candidate.env').read_text().splitlines()
 if runtime!=['XLA_PYTHON_CLIENT_PREALLOCATE=false','XLA_FLAGS=--xla_gpu_enable_triton_gemm=false','MUJOCO_GL=egl']:raise AssertionError('formal-candidate runtime contract')
 from training.peg_insertion.resume import REQUIRED_METADATA
 needed={'checkpoint_step','learner_update_steps','actor_steps','seed','environment','algorithm','observation','action','demo_sha256','demo_transition_count','config_hash','wandb_project','wandb_name','wandb_run_id','online_replay_sha256','created_at','source_snapshot'}
 if REQUIRED_METADATA!=needed:raise AssertionError('resume schema')
 for key in ('checkpoint_dir','log_dir','actor_log_dir','smoke_checkpoint_dir','smoke_log_dir','smoke_actor_log_dir'):
  out=(ROOT/c['outputs'][key]).resolve()
  if ROOT not in out.parents:raise ValueError('unsafe output')
  if out.exists() and any(out.iterdir()):raise FileExistsError(f'non-empty output: {out}')
 pkg=ROOT/'external/packages/peg_insertion/peg_insertion';sys.path[:0]=[str(pkg/'code/example/async_rlpd_peg_sim'),str(pkg/'code/env')]
 from peg_env_adapter import make_peg_state_env
 from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
 from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
 sys.path.insert(0,str(ROOT/'external/serl/serl_launcher'));from serl_launcher.utils.launcher import make_replay_buffer
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=5);obs,_=env.reset(seed=0)
  if np.asarray(obs).shape!=(20,) or np.asarray(obs).dtype!=np.float32 or env.action_space.shape!=(7,):raise AssertionError('env contract')
  b=make_replay_buffer(env,capacity=8,type='replay_buffer');t={'observations':obs,'actions':np.zeros(7,np.float32),'next_observations':obs.copy(),'rewards':np.float32(0),'masks':np.float32(1),'dones':False};b.insert(t);batch=b.sample(batch_size=1)
  if len(b)!=1 or np.asarray(batch['actions']).shape!=(1,7):raise AssertionError('replay adapter')
  env.close()
 print('DEMO_MANIFEST: PASS');print('CONFIG_CONTRACT: PASS');print('RUNTIME_ADAPTER: PASS');print('REPLAY_ADAPTER: PASS');print('RESUME_SCHEMA: PASS')
 backend=jax.default_backend();print('JAX_DEVICES:',jax.devices());print('JAX_BACKEND:',backend)
 if a.require_gpu and backend!='gpu':raise RuntimeError('GPU required')
 print('PEG_HANDOFF_VERIFY: PASS')
if __name__=='__main__':main()
