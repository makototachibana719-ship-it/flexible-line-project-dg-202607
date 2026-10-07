"""Single project entry for fresh Peg RLPD GPU training and safe resume."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,os,pickle,sys,time
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
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--dry-run',action='store_true');p.add_argument('--role',choices=('learner','actor'),required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--ip',default='localhost');p.add_argument('--resume-from',type=Path);p.add_argument('--resume-actor-step',type=int);p.add_argument('--wandb-run-id');p.add_argument('--require-gpu',action='store_true');p.add_argument('--diagnose-updates',action='store_true');p.add_argument('--split-utd1',action='store_true');p.add_argument('--diagnostic-rng-from-step',type=int);p.add_argument('--diagnostic-no-save',action='store_true');p.add_argument('--diagnostic-save-after-update',action='store_true');p.add_argument('--step-indexed-rng',action='store_true');p.add_argument('--max-resume-updates',type=int);return p.parse_args()
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
  if a.max_resume_updates is not None:
   if not resume or a.role!='learner' or a.max_resume_updates<1:raise ValueError('--max-resume-updates requires learner resume and a positive value')
   steps=min(steps,a.max_resume_updates);print('RESUME_UPDATE_CAP:',steps,flush=True)
  if a.diagnostic_save_after_update and not (resume and a.role=='learner' and a.diagnostic_no_save and (a.step_indexed_rng or a.diagnostic_rng_from_step is not None)):raise ValueError('--diagnostic-save-after-update requires learner resume, --diagnostic-no-save, and an explicit diagnostic RNG policy')
  checkpoint_period=0 if a.diagnostic_no_save else (cfg['smoke']['checkpoint_period'] if smoke else cfg['training']['checkpoint_period'])
  argv=[str(ENTRY),f'--{a.role}',f'--env={cfg["environment"]}',f'--seed={cfg["seed"]}',f'--max_steps={steps}',f'--random_steps={cfg["training"]["random_steps"]}',f'--training_starts={cfg["training"]["training_starts"]}',f'--batch_size={cfg["training"]["total_batch_size"]}',f'--critic_actor_ratio={cfg["training"]["critic_actor_ratio"]}',f'--utd_ratio={cfg["training"]["utd_ratio"]}',f'--replay_buffer_capacity={cfg["training"]["replay_capacity"]}',f'--checkpoint_period={checkpoint_period}',f'--checkpoint_path={out}',f'--demo_path={ROOT/cfg["demo"]["path"]}',f'--ip={a.ip}',f'--exp_name={cfg["wandb"]["smoke_run_name"] if smoke else cfg["wandb"]["run_name"]}']
  spec=importlib.util.spec_from_file_location('peg_delivered_rlpd',ENTRY);mod=importlib.util.module_from_spec(spec);sys.argv=argv;spec.loader.exec_module(mod)
  diag={'update_calls':0,'concat_calls':0}
  def signature(tree):
   leaves,treedef=jax.tree_util.tree_flatten_with_path(tree)
   return {'treedef':str(treedef),'leaves':[{'path':str(path),'shape':list(getattr(x,'shape',())),'dtype':str(getattr(x,'dtype',None)),'type':type(x).__name__} for path,x in leaves]}
  if a.diagnose_updates:
   original_concat=mod.concat_rlpd_batches
   def diagnosed_concat(online,demo):
    idx=diag['concat_calls'];diag['concat_calls']+=1
    print(f'[DIAG][update={idx}] after_sample t={time.time():.6f}',flush=True)
    print('[DIAG] online_signature='+json.dumps(signature(online),sort_keys=True),flush=True)
    print('[DIAG] demo_signature='+json.dumps(signature(demo),sort_keys=True),flush=True)
    t0=time.perf_counter();batch=original_concat(online,demo)
    print(f'[DIAG][update={idx}] after_merge dt={time.perf_counter()-t0:.6f}s',flush=True)
    print('[DIAG] mixed_signature='+json.dumps(signature(batch),sort_keys=True),flush=True)
    return batch
   mod.concat_rlpd_batches=diagnosed_concat
  original_agent=mod.make_sac_agent
  def make_agent(**kw):
   agent=original_agent(**kw)
   agent=agent.replace(state=restore_fresh_state(a.resume_from.parent,agent.state,resume['checkpoint_step'])) if resume and a.role=='learner' else agent
   if a.diagnostic_rng_from_step is not None:
    if not (resume and a.role=='learner' and a.diagnostic_no_save):raise ValueError('diagnostic RNG override requires learner resume with --diagnostic-no-save')
    reference=restore_fresh_state(a.resume_from.parent,agent.state,a.diagnostic_rng_from_step)
    agent=agent.replace(state=agent.state.replace(rng=reference.rng))
    print(f'[DIAG] RNG_OVERRIDDEN_FROM_STEP={a.diagnostic_rng_from_step}',flush=True)
   if (a.diagnose_updates or a.split_utd1 or a.step_indexed_rng or a.diagnostic_save_after_update) and a.role=='learner':
    cls=type(agent)
    if not getattr(cls,'_peg_update_diagnostics',False):
     original_update=cls.update_high_utd
     original_base_update=cls.update
     def diagnosed_update(self,batch,*,utd_ratio,pmap_axis=None):
      idx=diag['update_calls'];diag['update_calls']+=1
      if a.step_indexed_rng:
       global_step=(resume['checkpoint_step']+1 if resume else 0)+idx
       indexed_rng=jax.random.fold_in(jax.random.PRNGKey(cfg['seed']),global_step)
       self=self.replace(state=self.state.replace(rng=indexed_rng))
       print(f'[DIAG][update={idx}] STEP_INDEXED_RNG_GLOBAL_STEP={global_step} key={np.asarray(indexed_rng).tolist()}',flush=True)
      print(f'[DIAG][update={idx}] before_update t={time.time():.6f} utd_ratio={utd_ratio}',flush=True)
      t0=time.perf_counter()
      if a.split_utd1:
       if utd_ratio!=1:raise ValueError('--split-utd1 requires utd_ratio=1')
       phase=time.perf_counter();new_agent,critic_info=original_base_update(self,batch,pmap_axis=pmap_axis,networks_to_update=frozenset({'critic'}));jax.block_until_ready(new_agent)
       print(f'[DIAG][update={idx}] after_critic dt={time.perf_counter()-phase:.6f}s',flush=True)
       phase=time.perf_counter();new_agent,actor_info=original_base_update(new_agent,batch,pmap_axis=pmap_axis,networks_to_update=frozenset({'actor','temperature'}));jax.block_until_ready(new_agent)
       print(f'[DIAG][update={idx}] after_actor_temperature dt={time.perf_counter()-phase:.6f}s',flush=True)
       critic_info={k:v for k,v in critic_info.items() if k not in {'actor','temperature'}}
       actor_info={k:v for k,v in actor_info.items() if k!='critic'}
       info={**critic_info,**actor_info}
      else:new_agent,info=original_update(self,batch,utd_ratio=utd_ratio,pmap_axis=pmap_axis)
      jax.block_until_ready(new_agent)
      jax.tree_util.tree_map(lambda x:x.block_until_ready() if hasattr(x,'block_until_ready') else x,info)
      print(f'[DIAG][update={idx}] after_update dt={time.perf_counter()-t0:.6f}s',flush=True)
      if a.diagnostic_save_after_update:
       global_step=resume['checkpoint_step']+1+idx
       print(f'[DIAG][checkpoint={global_step}] before_post_update_checkpoint t={time.time():.6f}',flush=True)
       phase=time.perf_counter();save_bundle(out,new_agent.state,holder['online'],global_step,int(holder['online'].project_actor_steps),bundle_cfg,MANIFEST,active['id'],{'identifier':'peg-project-launcher-post-update','diagnostic_rng_from_step':a.diagnostic_rng_from_step,'step_indexed_rng':a.step_indexed_rng})
       print(f'[DIAG][checkpoint={global_step}] after_post_update_checkpoint dt={time.perf_counter()-phase:.6f}s',flush=True)
      return new_agent,info
     cls.update_high_utd=diagnosed_update;cls._peg_update_diagnostics=True
   return agent
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
   if a.diagnose_updates:print(f'[DIAG][checkpoint={global_step}] before_checkpoint t={time.time():.6f}',flush=True)
   t0=time.perf_counter();result=str(save_bundle(Path(directory),state,holder['online'],global_step,int(holder['online'].project_actor_steps),bundle_cfg,MANIFEST,active['id'],{'identifier':'peg-project-launcher'}))
   if a.diagnose_updates:print(f'[DIAG][checkpoint={global_step}] after_checkpoint dt={time.perf_counter()-t0:.6f}s',flush=True)
   return result
  mod.checkpoints.save_checkpoint=save
  print('FRESH_INITIALIZATION:',not bool(resume));print('LOAD_LEGACY_CHECKPOINT: FALSE');print('RESUME_PROFILE:','smoke' if smoke else 'formal');print('COMMAND_ARGS:',json.dumps(argv[1:]));mod.app.run(mod.main)
if __name__=='__main__':raise SystemExit(main())
