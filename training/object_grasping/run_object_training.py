"""Object GPU handoff launcher; fresh DrQ plus 50/50 Demo replay only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np, yaml
from training.object_grasping.resume import (install_actor_step_tracking, legacy_restore, object_bundle,
 restore_online_replay, save_bundle)

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs/object_grasping/object_baseline_seed0.yaml'
ENV_SRC=ROOT/'external/packages/object_grasping/object_grasp_package/code/env'
DELIVERED=ROOT/'external/packages/object_grasping/object_grasp_package/code/example/async_drq_sim.py'
MANIFEST=ROOT/'deliverables/gpu_training_handoff/object_grasping/demo_manifest.json'

def sha256(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): d.update(b)
 return d.hexdigest()
def load_config(path): return yaml.safe_load(path.read_text())
def adapt(env, obs):
 from gym.spaces import flatten
 state=np.asarray(flatten(env.unwrapped.observation_space['state'],obs['state']),dtype=np.float32)[None]
 return {'state':state,'front':np.asarray(obs['images']['front'])[None],'wrist':np.asarray(obs['images']['wrist'])[None]}
def make_adapter_class():
 import gym
 class ObjectTrainingAdapter(gym.ObservationWrapper):
  def __init__(self, env):
   super().__init__(env); native=env.observation_space
   self.observation_space=gym.spaces.Dict({
    'state':gym.spaces.Box(-np.inf,np.inf,(1,7),dtype=np.float32),
    'front':gym.spaces.Box(0,255,(1,128,128,3),dtype=np.uint8),
    'wrist':gym.spaces.Box(0,255,(1,128,128,3),dtype=np.uint8)})
  def observation(self, observation): return adapt(self.env,observation)
 return ObjectTrainingAdapter
def assert_contract(obs, action_space, cfg):
 if set(obs)!=set(cfg['observation']['keys']): raise AssertionError(f"observation keys: {set(obs)}")
 for key in ('state','front','wrist'):
  if list(obs[key].shape)!=cfg['observation'][f'{key}_shape']: raise AssertionError(f'{key} shape {obs[key].shape}')
 if tuple(action_space.shape)!=tuple(cfg['action']['shape']): raise AssertionError('action shape')
 sample=np.asarray(action_space.sample())
 if sample.dtype!=np.float32 or not np.isfinite(sample).all() or not action_space.contains(sample): raise AssertionError('action contract')
def validate_demo(cfg):
 p=ROOT/cfg['demo']['path']
 if p.stat().st_size!=cfg['demo']['size_bytes'] or sha256(p)!=cfg['demo']['sha256']: raise AssertionError('Demo integrity')
 with p.open('rb') as f: data=pickle.load(f)
 if len(data)!=cfg['demo']['raw_transitions']: raise AssertionError('Demo transition count')
 required={'observations','actions','next_observations','rewards','masks','dones'}
 if any(set(x)!=required for x in data): raise AssertionError('Demo schema')
 return data
def parse():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--dry-run',action='store_true'); p.add_argument('--role',choices=('learner','actor'),required=True); p.add_argument('--smoke',action='store_true'); p.add_argument('--ip',default='localhost'); p.add_argument('--resume-from'); p.add_argument('--resume-actor-step',type=int); p.add_argument('--wandb-run-id'); return p.parse_args()
def main():
 a=parse(); cfg=load_config(a.config); sys.path.insert(0,str(ENV_SRC)); import gym, franka_sim
 env=gym.make(cfg['environment']); native,_=env.reset(seed=cfg['seed']); obs=adapt(env,native); assert_contract(obs,env.action_space,cfg); data=validate_demo(cfg)
 resume=None
 if a.resume_from:
  if not a.wandb_run_id or (a.role=='actor' and a.resume_actor_step is None): raise ValueError('resume requires W&B ID and actor step for actor role')
  resume=object_bundle(Path(a.resume_from),cfg,MANIFEST,a.wandb_run_id)
  if a.role=='actor' and a.resume_actor_step!=resume['actor_steps']: raise ValueError('actor step does not match bundle metadata')
  print(f"RESUME_BUNDLE: PASS checkpoint={resume['checkpoint_step']} actor_steps={resume['actor_steps']}")
 print(f"FRESH_INITIALIZATION: {'FALSE' if resume else 'TRUE'}"); print('LOAD_LEGACY_CHECKPOINT_4500: FALSE'); print(f'RAW_DEMO_TRANSITIONS: {len(data)}'); print('OBSERVATION_CONTRACT: PASS'); print('ACTION_CONTRACT: PASS')
 if a.dry_run: print('OBJECT_TRAINING_DRY_RUN: PASS'); env.close(); return 0
 env.close()
 # The delivered algorithm remains authoritative. Project integration arguments are explicit and no legacy checkpoint is passed.
 steps=cfg['smoke']['actor_steps'] if a.smoke and a.role=='actor' else (cfg['smoke']['learner_updates'] if a.smoke else cfg['training']['max_steps'])
 if resume: steps=max(0,steps-(resume['actor_steps'] if a.role=='actor' else resume['learner_update_steps']+1))
 random_steps=max(0,cfg['training']['random_steps']-(resume['actor_steps'] if resume and a.role=='actor' else 0))
 out=cfg['outputs'][('smoke_' if a.smoke else '')+('checkpoint_dir' if a.role=='learner' else 'log_dir')]
 argv=[str(DELIVERED),f'--{a.role}',f'--env={cfg["environment"]}',f'--seed={cfg["seed"]}',f'--encoder_type={cfg["training"]["encoder_type"]}',f'--batch_size={cfg["training"]["batch_size"]}',f'--replay_buffer_capacity={cfg["training"]["replay_capacity"]}',f'--training_starts={cfg["training"]["training_starts"]}',f'--random_steps={random_steps}',f'--critic_actor_ratio={cfg["training"]["critic_actor_ratio"]}',f'--steps_per_update={cfg["training"]["steps_per_update"]}',f'--max_steps={steps}',f'--log_period={cfg["training"]["log_period"]}',f'--eval_period={cfg["training"]["eval_period"]}',f'--eval_n_trajs={cfg["training"]["eval_n_trajs"]}',f'--ip={a.ip}',f'--demo_path={ROOT/cfg["demo"]["path"]}',f'--checkpoint_path={ROOT/out}',f'--checkpoint_period={cfg["smoke"]["checkpoint_period"] if a.smoke else cfg["training"]["checkpoint_period"]}',f'--exp_name={cfg["wandb"]["smoke_run_name"] if a.smoke else cfg["wandb"]["run_name"]}']
 print('DELIVERED_ENTRY:',DELIVERED); print('COMMAND_ARGS:',json.dumps(argv[1:])); print('NOTE: real execution delegates the unchanged delivered algorithm; resume bundle restoration is a pre-launch project gate.')
 spec=importlib.util.spec_from_file_location('object_delivered_train',DELIVERED); mod=importlib.util.module_from_spec(spec); sys.argv=argv; spec.loader.exec_module(mod)
 mod.SERLObsWrapper=make_adapter_class()
 original_make_agent=mod.make_drq_agent
 def resume_make_agent(**kwargs):
  agent=original_make_agent(**kwargs)
  if resume: agent=agent.replace(state=legacy_restore(Path(resume['checkpoint_path']).parent,agent.state,resume['checkpoint_step']))
  return agent
 mod.make_drq_agent=resume_make_agent
 original_make_replay=mod.make_replay_buffer; replay_holder={}; replay_calls={'count':0}
 def resume_make_replay(*args,**kwargs):
  replay=original_make_replay(*args,**kwargs); replay_calls['count']+=1
  if a.role=='learner' and replay_calls['count']==1:
   replay_holder['online']=replay
   if resume: restore_online_replay(replay,resume['online_replay_path'])
   else: install_actor_step_tracking(replay,0)
  return replay
 mod.make_replay_buffer=resume_make_replay
 active_run={'id':a.wandb_run_id}
 class ExplicitWandbLogger:
  def __init__(self):
   import wandb
   kwargs={'project':cfg['wandb']['project'],'name':cfg['wandb']['smoke_run_name'] if a.smoke else cfg['wandb']['run_name'],'mode':'online'}
   if a.resume_from: kwargs.update(id=a.wandb_run_id,resume='must')
   self.run=wandb.init(**kwargs)
   active_run['id']=self.run.id
  def log(self,payload,step=None): self.run.log(payload,step=None if step is None else step+(resume['checkpoint_step']+1 if resume else 0))
 mod.make_wandb_logger=lambda **_kwargs: ExplicitWandbLogger()
 original_save=mod.checkpoints.save_checkpoint
 def bundle_save(directory,target,step,keep=20,**_kwargs):
  if 'online' not in replay_holder: raise RuntimeError('online replay unavailable for bundle save')
  global_step=(resume['checkpoint_step']+1+int(step)) if resume else int(step)
  return str(save_bundle(Path(directory),target,replay_holder['online'],global_step,int(getattr(replay_holder['online'],'project_actor_steps',resume['actor_steps'] if resume else 0)),cfg,MANIFEST,active_run['id'],{'identifier':'object-training-launcher'}))
 mod.checkpoints.save_checkpoint=bundle_save
 mod.app.run(mod.main)
 return 0
if __name__=='__main__': raise SystemExit(main())
