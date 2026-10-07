"""Single project entry for fresh Peg RLPD GPU training and safe resume."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,os,pickle,sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np,yaml
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import install_actor_step_tracking,load_bundle,restore_replay,save_bundle
from training.peg_insertion.checkpoint import restore_fresh_state

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml'
ENTRY=ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/async_sac_state_sim.py'
EXAMPLE=ENTRY.parent;ENV_PARENT=ROOT/'external/packages/peg_insertion/peg_insertion/code/env'
MANIFEST=ROOT/'deliverables/gpu_training_handoff/peg_insertion/demo_manifest.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_config(path):return yaml.safe_load(path.read_text())
def demo_data(cfg):
 p=ROOT/cfg['demo']['path']
 if sha(p)!=cfg['demo']['sha256']:raise AssertionError('Demo SHA256 mismatch')
 data=pickle.loads(p.read_bytes())
 if len(data)!=cfg['demo']['transitions']:raise AssertionError('Demo transition mismatch')
 return data
def safe_output(path):
 p=(ROOT/path).resolve()
 if ROOT not in p.parents:raise ValueError('output outside project')
 if p.exists() and any(p.iterdir()):raise FileExistsError(f'non-empty output: {p}')
 return p
def assert_contract(env,cfg):
 obs,_=env.reset(seed=cfg['seed']);obs=np.asarray(obs)
 if obs.shape!=(20,) or obs.dtype!=np.float32:raise AssertionError('observation contract')
 a=env.action_space
 if a.shape!=(7,) or a.dtype!=np.float32 or not np.allclose(a.low,-1) or not np.allclose(a.high,1):raise AssertionError('action contract')
 if cfg['algorithm']!='RLPD_STYLE_SAC_PLUS_50_50_DEMO_REPLAY' or not cfg['fresh_initialization'] or cfg['load_legacy_checkpoint']:raise AssertionError('config identity')
 if cfg['wandb']!={'project':'flexible-line-peg-insertion','run_name':'peg_rlpd_baseline_seed0','smoke_run_name':'peg_rlpd_gpu_smoke_seed0','mode':'online'}:raise AssertionError('W&B contract')
 return obs
def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--dry-run',action='store_true');p.add_argument('--role',choices=('learner','actor'),required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--ip',default='localhost');p.add_argument('--resume-from',type=Path);p.add_argument('--resume-actor-step',type=int);p.add_argument('--wandb-run-id');p.add_argument('--require-gpu',action='store_true');return p.parse_args()
def main():
 a=parse();cfg=load_config(a.config.resolve());smoke=a.smoke
 if a.resume_from and not smoke:
  cp=a.resume_from.resolve();meta=cp.parent/f'{cp.name}_resume_metadata.json'
  resume_name=json.loads(meta.read_text()).get('wandb_name')
  if resume_name==cfg['wandb']['smoke_run_name']:smoke=True
  elif resume_name!=cfg['wandb']['run_name']:raise ValueError('unknown resume W&B profile')
 bundle_cfg=copy.deepcopy(cfg)
 if smoke:bundle_cfg['wandb']['run_name']=cfg['wandb']['smoke_run_name']
 data=demo_data(cfg)
 sys.path[:0]=[str(EXAMPLE),str(ENV_PARENT)]
 import jax
 from peg_env_adapter import make_peg_state_env
 from serl_launcher.utils.launcher import make_replay_buffer
 with ExitStack() as stack:
  stack.enter_context(portable_peg_runtime(ROOT));stack.enter_context(repaired_replay_datastore_insert())
  env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=500);assert_contract(env,cfg)
  demo=make_replay_buffer(env,capacity=max(10000,len(data)+100),type='replay_buffer')
  for t in data:demo.insert(t)
  if len(demo)!=5512:raise AssertionError('Demo replay population')
  print('DEMO_SUCCESS_PROVENANCE: 20/20 VALIDATED');print('DEMO_REPLAY_SIZE:',len(demo));print('REPLAY_ADAPTER: ACTIVE')
  backend=jax.default_backend();print('JAX_DEVICES:',jax.devices());print('JAX_BACKEND:',backend)
  if (a.require_gpu or not a.dry_run) and backend!='gpu':raise RuntimeError('GPU backend required')
  resume=None
  if a.resume_from:
   if not a.wandb_run_id:raise ValueError('resume requires --wandb-run-id')
   resume=load_bundle(a.resume_from,bundle_cfg,MANIFEST,a.wandb_run_id)
   if a.role=='actor' and a.resume_actor_step!=resume['actor_steps']:raise ValueError('actor step mismatch')
  out_key=('smoke_' if smoke else '')+('checkpoint_dir' if a.role=='learner' else 'actor_log_dir')
  out=(ROOT/cfg['outputs'][out_key]).resolve() if resume else safe_output(Path(cfg['outputs'][out_key]))
  if resume and (ROOT not in out.parents or (a.role=='learner' and out!=a.resume_from.resolve().parent)):raise ValueError('resume output mismatch')
  if a.dry_run:print('PEG_TRAINING_DRY_RUN: PASS');env.close();return 0
  env.close()
  steps=cfg['smoke']['actor_steps'] if smoke and a.role=='actor' else (cfg['smoke']['learner_updates'] if smoke else cfg['training']['max_steps'])
  if resume:steps=max(0,steps-(resume['actor_steps'] if a.role=='actor' else resume['learner_update_steps']+1))
  argv=[str(ENTRY),f'--{a.role}',f'--env={cfg["environment"]}',f'--seed={cfg["seed"]}',f'--max_steps={steps}',f'--random_steps={cfg["training"]["random_steps"]}',f'--training_starts={cfg["training"]["training_starts"]}',f'--batch_size={cfg["training"]["total_batch_size"]}',f'--critic_actor_ratio={cfg["training"]["critic_actor_ratio"]}',f'--utd_ratio={cfg["training"]["utd_ratio"]}',f'--replay_buffer_capacity={cfg["training"]["replay_capacity"]}',f'--checkpoint_period={cfg["smoke"]["checkpoint_period"] if smoke else cfg["training"]["checkpoint_period"]}',f'--checkpoint_path={out}',f'--demo_path={ROOT/cfg["demo"]["path"]}',f'--ip={a.ip}',f'--exp_name={cfg["wandb"]["smoke_run_name"] if smoke else cfg["wandb"]["run_name"]}']
  spec=importlib.util.spec_from_file_location('peg_delivered_rlpd',ENTRY);mod=importlib.util.module_from_spec(spec);sys.argv=argv;spec.loader.exec_module(mod)
  original_agent=mod.make_sac_agent
  def make_agent(**kw):
   agent=original_agent(**kw)
   return agent.replace(state=restore_fresh_state(a.resume_from.parent,agent.state,resume['checkpoint_step'])) if resume and a.role=='learner' else agent
  mod.make_sac_agent=make_agent
  original_replay=mod.make_replay_buffer;holder={};calls={'n':0}
  def make_replay(*args,**kw):
   b=original_replay(*args,**kw);calls['n']+=1
   if a.role=='learner' and calls['n']==1:
    holder['online']=b
    if resume:restore_replay(b,resume['online_replay_path'])
    else:install_actor_step_tracking(b,0)
   return b
  mod.make_replay_buffer=make_replay
  active={'id':a.wandb_run_id}
  class Logger:
   def __init__(self):
    import wandb
    kw={'project':cfg['wandb']['project'],'name':cfg['wandb']['smoke_run_name'] if smoke else cfg['wandb']['run_name'],'mode':'online','dir':str((ROOT/cfg['outputs'][('smoke_' if smoke else '')+'log_dir']).resolve())}
    if resume:kw.update(id=a.wandb_run_id,resume='must')
    self.run=wandb.init(**kw);active['id']=self.run.id
   def log(self,payload,step=None):self.run.log(payload,step=None if step is None else step+(resume['checkpoint_step']+1 if resume else 0))
  mod.make_wandb_logger=lambda **kw:Logger()
  def save(directory,state,step,keep=20,**kw):
   if 'online' not in holder:raise RuntimeError('online replay unavailable')
   global_step=resume['checkpoint_step']+1+int(step) if resume else int(step)
   return str(save_bundle(Path(directory),state,holder['online'],global_step,int(holder['online'].project_actor_steps),bundle_cfg,MANIFEST,active['id'],{'identifier':'peg-project-launcher'}))
  mod.checkpoints.save_checkpoint=save
  print('FRESH_INITIALIZATION:',not bool(resume));print('LOAD_LEGACY_CHECKPOINT: FALSE');print('RESUME_PROFILE:','smoke' if smoke else 'formal');print('COMMAND_ARGS:',json.dumps(argv[1:]));mod.app.run(mod.main)
if __name__=='__main__':raise SystemExit(main())
