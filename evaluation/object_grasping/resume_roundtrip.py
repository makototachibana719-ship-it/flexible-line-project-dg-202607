"""Cross-process Object safe-resume bundle, replay, failure and inference test."""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, shutil, subprocess, sys
from pathlib import Path
import jax, numpy as np
from training.object_grasping.run_object_training import ROOT, DEFAULT_CONFIG, ENV_SRC, load_config, make_adapter_class, validate_demo
from training.object_grasping.resume import (canonical_hash, install_actor_step_tracking, legacy_restore, object_bundle, restore_online_replay, save_bundle, sha256_file, sidecar_paths)
MANIFEST=ROOT/'deliverables/gpu_training_handoff/object_grasping/demo_manifest.json'
DEFAULT_OUTPUT=ROOT/'runtime/object_handoff/resume_roundtrip'
sys.path.insert(0,str(ROOT/'external/serl/serl_launcher'))
from serl_launcher.utils.launcher import make_drq_agent, make_replay_buffer

def env_agent_buffer(cfg,capacity=32):
 sys.path.insert(0,str(ENV_SRC)); import gym,franka_sim
 env=make_adapter_class()(gym.make(cfg['environment'])); obs,_=env.reset(seed=cfg['seed']); agent=make_drq_agent(seed=cfg['seed'],sample_obs=obs,sample_action=env.action_space.sample(),image_keys=['front','wrist'],encoder_type=cfg['training']['encoder_type']); replay=make_replay_buffer(env,capacity=capacity,type='memory_efficient_replay_buffer',image_keys=['front','wrist']); return env,agent,replay
def leaf_hashes(tree):
 out=[]
 for x in jax.tree_util.tree_leaves(tree):
  a=np.asarray(jax.device_get(x)); out.append(hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest())
 return out
def add_real_schema(replay,cfg,count=4):
 for transition in validate_demo(cfg)[:count]: replay.insert(transition)
def create(cfg,out):
 out.mkdir(parents=True,exist_ok=False); env,agent,replay=env_agent_buffer(cfg); install_actor_step_tracking(replay,13); add_real_schema(replay,cfg,4); cp=save_bundle(out,agent.state,replay,123,17,cfg,MANIFEST,'object-resume-unit-run',{'identifier':'object-m1dc2-bounded'}); meta,rp=sidecar_paths(cp); (out/'create_evidence.json').write_text(json.dumps({'agent_hashes':leaf_hashes(agent.state),'checkpoint_sha256':sha256_file(cp),'replay_sha256':sha256_file(rp),'replay_size':len(replay),'insert_index':int(replay._insert_index),'actor_steps':17},indent=2)+'\n'); env.close()
def copy_bundle(src,dst,include_replay=True):
 dst.mkdir(); cp=dst/'checkpoint_123'; shutil.copy2(src/'checkpoint_123',cp); m,r=sidecar_paths(cp); shutil.copy2(src/'checkpoint_123_resume_metadata.json',m)
 if include_replay: shutil.copy2(src/'checkpoint_123_online_replay.pkl',r)
 return cp
def failure_tests(cfg,out):
 failures={}
 def closed(name,fn):
  try: fn()
  except Exception as e: failures[name]={'result':'FAIL_CLOSED','error':type(e).__name__}
  else: raise AssertionError(f'{name} did not fail closed')
 cp=out/'checkpoint_123'
 closed('missing_replay',lambda:object_bundle(copy_bundle(out,out/'fail_missing',False),cfg,MANIFEST,'object-resume-unit-run'))
 def alter(name,key,value):
  x=copy_bundle(out,out/name); m,_=sidecar_paths(x); d=json.loads(m.read_text()); d[key]=value; m.write_text(json.dumps(d)); return x
 closed('wrong_replay_hash',lambda:object_bundle(alter('fail_replay_hash','online_replay_sha256','0'*64),cfg,MANIFEST,'object-resume-unit-run'))
 closed('wrong_demo_hash',lambda:object_bundle(alter('fail_demo_hash','demo_sha256','0'*64),cfg,MANIFEST,'object-resume-unit-run'))
 bad=dict(cfg); bad['seed']=99; closed('config_mismatch',lambda:object_bundle(cp,bad,MANIFEST,'object-resume-unit-run'))
 closed('wandb_run_mismatch',lambda:object_bundle(cp,cfg,MANIFEST,'wrong-run'))
 env,agent,replay=env_agent_buffer(cfg); install_actor_step_tracking(replay,17); closed('existing_checkpoint_collision',lambda:save_bundle(out,agent.state,replay,123,17,cfg,MANIFEST,'object-resume-unit-run',{})); env.close(); return failures
def resume(cfg,out):
 cp=out/'checkpoint_123'; before=sha256_file(cp); original=json.loads((out/'create_evidence.json').read_text()); bundle=object_bundle(cp,cfg,MANIFEST,'object-resume-unit-run'); env,agent,replay=env_agent_buffer(cfg); restored=legacy_restore(out,agent.state,123); hashes=leaf_hashes(restored)
 if hashes!=original['agent_hashes']: raise AssertionError('agent values mismatch')
 info=restore_online_replay(replay,bundle['online_replay_path'])
 if info['actor_steps']!=17 or info['size']!=original['replay_size'] or int(replay._insert_index)!=original['insert_index']: raise AssertionError('replay progress mismatch')
 rng_before=pickle.loads(Path(bundle['online_replay_path']).read_bytes())['rng_state']; rng_after={'kind':'generator','state':replay.np_random.bit_generator.state} if hasattr(replay.np_random,'bit_generator') else {'kind':'random_state','state':replay.np_random.get_state()}
 if pickle.dumps(rng_before)!=pickle.dumps(rng_after): raise AssertionError('replay RNG mismatch')
 batch=next(replay.get_iterator(sample_args={'batch_size':2,'pack_obs_and_next_obs':True}));
 for key in ('state','front','wrist'):
  if key not in batch['observations'] or not np.isfinite(np.asarray(batch['observations'][key])).all(): raise AssertionError('sample observation')
 if np.asarray(batch['actions']).shape[-1]!=4 or not np.isfinite(np.asarray(batch['actions'])).all(): raise AssertionError('sample action')
 demo=make_replay_buffer(env,capacity=cfg['training']['replay_capacity'],type='memory_efficient_replay_buffer',image_keys=['front','wrist']); data=validate_demo(cfg)
 for transition in data: demo.insert(transition)
 if len(data)!=2000 or len(demo)!=2001: raise AssertionError('Demo rebuild')
 restored_agent=agent.replace(state=restored); obs,_=env.reset(seed=0); action=np.asarray(jax.device_get(restored_agent.sample_actions(observations=jax.device_put(obs),argmax=True))); infer=action.shape==(4,) and np.isfinite(action).all() and np.all(action>=-1) and np.all(action<=1)
 if not infer: raise AssertionError('inference')
 if bundle['wandb_project']!=cfg['wandb']['project'] or bundle['wandb_name']!=cfg['wandb']['run_name'] or bundle['wandb_run_id']!='object-resume-unit-run': raise AssertionError('W&B identity')
 next_step=125; save_bundle(out,restored,replay,next_step,17,cfg,MANIFEST,'object-resume-unit-run',{'identifier':'resumed-next'});
 if sha256_file(cp)!=before: raise AssertionError('old checkpoint changed')
 failures=failure_tests(cfg,out); report={'restored_learner_step':bundle['learner_update_steps'],'restored_actor_steps':info['actor_steps'],'remaining_random_steps':max(0,cfg['training']['random_steps']-info['actor_steps']),'agent_structure_value_match':True,'online_replay_size':info['size'],'online_replay_index_match':True,'online_replay_rng_match':True,'online_replay_resume':'VALIDATED','sample_batch':'PASS','demo_raw_transitions':len(data),'demo_replay_size':len(demo),'demo_replay_rebuilt':'VALIDATED','wandb_project':bundle['wandb_project'],'wandb_name':bundle['wandb_name'],'wandb_run_id':bundle['wandb_run_id'],'wandb_resume':'must','next_checkpoint_step':next_step,'old_checkpoint_unchanged':True,'collision_protection':'PASS','failure_tests':failures,'restored_policy_inference':'PASS','object_resume_gate':'VALIDATED'}; (out/'resume_roundtrip_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); env.close()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT); p.add_argument('--phase',choices=('create','resume')); a=p.parse_args(); cfg=load_config(a.config.resolve()); out=a.output_dir.resolve()
 if ROOT not in out.parents: raise ValueError('project-local output required')
 if a.phase=='create': create(cfg,out); return 0
 if a.phase=='resume': resume(cfg,out); return 0
 if out.exists() and any(out.iterdir()): raise FileExistsError('non-empty output')
 base=[sys.executable,'-m','evaluation.object_grasping.resume_roundtrip','--config',str(a.config.resolve()),'--output-dir',str(out)]; env=os.environ.copy(); subprocess.run(base+['--phase','create'],check=True,env=env); subprocess.run(base+['--phase','resume'],check=True,env=env); print('PROCESS_RECONSTRUCTION: PASS'); print('OBJECT_RESUME_GATE: VALIDATED'); return 0
if __name__=='__main__': raise SystemExit(main())
