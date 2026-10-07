"""Single project entry for fresh Peg RLPD GPU training and safe resume."""
from __future__ import annotations
import argparse,copy,faulthandler,hashlib,importlib.util,json,os,pickle,sys,threading,time
from collections.abc import Mapping
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
def wandb_exit_code(exception):
 """Classify normal absl SystemExit without suppressing the original exception."""
 if exception is None:return 0
 if isinstance(exception,SystemExit):
  code=exception.code
  if code is None:return 0
  return code if isinstance(code,int) else 1
 return 1

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
 from training.peg_insertion.observation_contract import validate_config,require_match
 require_match(validate_config(cfg,for_training=True),env.observation_contract_version)
 obs,_=env.reset(seed=cfg['seed']);obs=np.asarray(obs)
 if obs.shape!=(20,) or obs.dtype!=np.float32:raise AssertionError('observation contract')
 a=env.action_space
 if a.shape!=(7,) or a.dtype!=np.float32 or not np.allclose(a.low,-1) or not np.allclose(a.high,1):raise AssertionError('action contract')
 if cfg['algorithm']!='RLPD_STYLE_SAC_PLUS_50_50_DEMO_REPLAY' or not cfg['fresh_initialization'] or cfg['load_legacy_checkpoint']:raise AssertionError('config identity')
 if cfg['wandb']!={'project':'flexible-line-peg-insertion','run_name':'peg_rlpd_baseline_seed0','smoke_run_name':'peg_rlpd_gpu_smoke_seed0','mode':'online'}:raise AssertionError('W&B contract')
 return obs
def parse():
 # Keep this one-shot branch permission separate from general resume/formal flags.
 branch_parser=argparse.ArgumentParser(add_help=False)
 branch_parser.add_argument('--resume-branch-acceptance',action='store_true')
 branch_args,remaining=branch_parser.parse_known_args()
 sys.argv=[sys.argv[0],*remaining]
 parsed=_parse_base()
 parsed.resume_branch_acceptance=branch_args.resume_branch_acceptance
 return parsed
def _parse_base():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--profile',choices=('formal','smoke','diagnostic','acceptance'),required=True);p.add_argument('--formal-candidate',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--role',choices=('learner','actor'),required=True);p.add_argument('--ip',default='localhost');p.add_argument('--resume-from',type=Path);p.add_argument('--resume-actor-step',type=int);p.add_argument('--wandb-run-id');p.add_argument('--require-gpu',action='store_true');p.add_argument('--diagnose-updates',action='store_true');p.add_argument('--split-utd1',action='store_true');p.add_argument('--diagnostic-rng-from-step',type=int);p.add_argument('--diagnostic-rng-key');p.add_argument('--diagnostic-no-save',action='store_true');p.add_argument('--diagnostic-save-after-update',action='store_true');p.add_argument('--step-indexed-rng',action='store_true');p.add_argument('--max-resume-updates',type=int);p.add_argument('--diagnostic-disable-wandb',action='store_true');p.add_argument('--diagnostic-stack-dumps',action='store_true');p.add_argument('--observe-update1',action='store_true');p.add_argument('--per-leaf-sync',action='store_true');return p.parse_args()
def main():
 a=parse()
 if a.profile=='formal':
  if a.config.resolve()!=CONFIG.resolve():raise ValueError('formal config must be the source-bound baseline')
  from training.peg_insertion.formal_preflight import role_lock,preflight
  role=('resume_'+a.role) if a.resume_from else ('fresh_learner' if a.role=='learner' else 'actor_join')
  with role_lock(a.role):
   checked=preflight(role,a.resume_from,a.wandb_run_id,a.ip,own_role=a.role,check_cloud=True)
   if a.role=='actor' and a.resume_from:a.resume_actor_step=checked['actor_steps']
   return run_main(a)
 return run_main(a)
def run_main(a):
 cfg=load_config(a.config.resolve())
 from training.peg_insertion.observation_contract import validate_config
 validate_config(cfg,for_training=True)
 branch=getattr(a,'resume_branch_acceptance',False)
 branch_policy=None;source_bundle_cfg=None
 if branch:
  from training.peg_insertion import branch_acceptance as branch_policy
  if a.profile!='acceptance' or not a.per_leaf_sync or not a.observe_update1 or a.formal_candidate:raise ValueError('branch resume requires observed per-leaf acceptance')
  branch_policy.validate_source(a.resume_from,a.wandb_run_id)
  source_bundle_cfg=branch_policy.source_config(cfg)
 from training.peg_insertion.formal_policy import require_gate,validate_runtime
 acceptance=a.profile=='acceptance'
 if a.observe_update1 and not acceptance:raise ValueError('update observation requires acceptance profile')
 # Per-leaf waits are the shared production policy, not a diagnostic workaround flag.
 if a.profile=='formal':require_gate()
 if acceptance:
  validate_runtime()
  if (a.resume_from and not branch) or a.formal_candidate:raise ValueError('F0 acceptance is fresh only; use explicit bounded branch approval')
  root=ROOT/'results/validation/peg/f0_acceptance'
  if a.observe_update1:root=ROOT/'results/validation/peg/f0_observe_20260904'
  if a.per_leaf_sync:root=ROOT/'results/validation/peg/f0_sync_20260904'
  cfg=copy.deepcopy(cfg)
  cfg['formal_candidate'].update(learner_updates=50,checkpoint_period=25,wandb_run_name='peg_f0_exact_path_seed0',checkpoint_dir=str(root/'checkpoints'),log_dir=str(root/'logs'),result_dir=str(root/'learner'))
  if branch:
   root=branch_policy.OUT
   cfg['formal_candidate'].update(learner_updates=5,wandb_run_name=branch_policy.RUN_NAME,checkpoint_dir=str(root/'checkpoints'),log_dir=str(root/'logs'),result_dir=str(root/'learner'))
 diagnostic_values=(a.diagnose_updates,a.split_utd1,a.diagnostic_rng_from_step is not None,a.diagnostic_rng_key is not None,a.diagnostic_no_save,a.diagnostic_save_after_update,a.step_indexed_rng,a.max_resume_updates is not None,a.diagnostic_disable_wandb,a.diagnostic_stack_dumps)
 if a.profile!='diagnostic' and any(diagnostic_values):raise ValueError('diagnostic-only arguments require --profile diagnostic')
 if a.profile=='diagnostic' and not any(diagnostic_values):raise ValueError('--profile diagnostic requires at least one diagnostic-only argument')
 if a.formal_candidate and (a.profile!='smoke' or a.role!='learner' or (a.resume_from and not a.dry_run)):raise ValueError('formal candidate requires fresh learner --profile smoke; resume is dry-run only')
 if a.formal_candidate and any(diagnostic_values):raise ValueError('formal candidate forbids all diagnostic-only arguments')
 selected_profile=a.profile;bound_profile=None
 if a.resume_from:
  cp=a.resume_from.resolve();meta=cp.parent/f'{cp.name}_resume_metadata.json';resume_name=json.loads(meta.read_text()).get('wandb_name')
  profile_by_name={cfg['wandb']['smoke_run_name']:'smoke',cfg['wandb']['run_name']:'formal',cfg['formal_candidate']['wandb_run_name']:'candidate'}
  if branch:profile_by_name['peg_f0_exact_path_seed0']='acceptance'
  bound_profile=profile_by_name.get(resume_name)
  expected_profile='candidate' if a.formal_candidate else selected_profile
  if bound_profile is None or (selected_profile!='diagnostic' and bound_profile!=expected_profile):raise ValueError('checkpoint/profile mismatch')
  if selected_profile=='diagnostic':selected_profile='smoke' if bound_profile=='candidate' else bound_profile
 diagnostic_candidate_resume=a.profile=='diagnostic' and bound_profile=='candidate'
 smoke=selected_profile=='smoke';candidate=a.formal_candidate or (acceptance and a.role=='learner')
 bundle_cfg=copy.deepcopy(cfg)
 if candidate or diagnostic_candidate_resume or branch:bundle_cfg['wandb']['run_name']=cfg['formal_candidate']['wandb_run_name']
 elif smoke:bundle_cfg['wandb']['run_name']=cfg['wandb']['smoke_run_name']
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
   resume=load_bundle(a.resume_from,source_bundle_cfg if branch else bundle_cfg,MANIFEST,a.wandb_run_id)
   if a.role=='actor' and a.resume_actor_step!=resume['actor_steps']:raise ValueError('actor step mismatch')
  candidate_replay_path=None;candidate_result_dir=None;candidate_log_dir=None
  if candidate and not resume and not acceptance:
   source_cp=(ROOT/cfg['formal_candidate']['replay_checkpoint']).resolve();source_meta,source_replay=source_cp.parent/f'{source_cp.name}_resume_metadata.json',source_cp.parent/f'{source_cp.name}_online_replay.pkl'
   if not source_meta.is_file() or not source_replay.is_file():raise FileNotFoundError('formal candidate requires validated smoke replay sidecars')
   source_metadata=json.loads(source_meta.read_text())
   if source_metadata.get('online_replay_sha256')!=sha(source_replay) or int(source_metadata.get('online_replay_size',0))<cfg['training']['training_starts']:raise ValueError('formal candidate replay integrity/size failure')
   candidate_replay_path=source_replay
  if candidate:
   out=a.resume_from.resolve().parent if resume and not branch else safe_output(Path(cfg['formal_candidate']['checkpoint_dir']))
   candidate_log_dir=(ROOT/cfg['formal_candidate']['log_dir']).resolve()
   candidate_result_dir=(ROOT/cfg['formal_candidate']['result_dir']).resolve()
   if not resume or branch:
    safe_output(Path(cfg['formal_candidate']['log_dir']));safe_output(Path(cfg['formal_candidate']['result_dir']));candidate_result_dir.mkdir(parents=True,exist_ok=True)
  elif diagnostic_candidate_resume:
   out=a.resume_from.resolve().parent
  elif acceptance:
   out=safe_output(root/'actor')
  else:
   out_key=('smoke_' if smoke else '')+('checkpoint_dir' if a.role=='learner' else 'actor_log_dir')
   out=(ROOT/cfg['outputs'][out_key]).resolve() if resume else safe_output(Path(cfg['outputs'][out_key]))
  if resume and (ROOT not in out.parents or (a.role=='learner' and not branch and out!=a.resume_from.resolve().parent)):raise ValueError('resume output mismatch')
  if a.dry_run:print('PEG_TRAINING_DRY_RUN: PASS');env.close();return 0
  env.close()
  steps=cfg['formal_candidate']['learner_updates'] if candidate else (cfg['smoke']['actor_steps'] if smoke and a.role=='actor' else (cfg['smoke']['learner_updates'] if smoke else cfg['training']['max_steps']))
  if acceptance and a.role=='actor':steps=1500
  if resume and not branch:steps=max(0,steps-(resume['actor_steps'] if a.role=='actor' else resume['learner_update_steps']+1))
  if a.max_resume_updates is not None:
   if not resume or a.role!='learner' or a.max_resume_updates<1:raise ValueError('--max-resume-updates requires learner resume and a positive value')
   steps=min(steps,a.max_resume_updates);print('RESUME_UPDATE_CAP:',steps,flush=True)
  if sum((a.step_indexed_rng,a.diagnostic_rng_from_step is not None,a.diagnostic_rng_key is not None))>1:raise ValueError('diagnostic RNG policies are mutually exclusive')
  if a.diagnostic_save_after_update and not (resume and a.role=='learner' and a.diagnostic_no_save and (a.step_indexed_rng or a.diagnostic_rng_from_step is not None or a.diagnostic_rng_key is not None)):raise ValueError('--diagnostic-save-after-update requires learner resume, --diagnostic-no-save, and an explicit diagnostic RNG policy')
  checkpoint_period=0 if a.diagnostic_no_save else (cfg['formal_candidate']['checkpoint_period'] if candidate else (cfg['smoke']['checkpoint_period'] if smoke else cfg['training']['checkpoint_period']))
  run_name=cfg['formal_candidate']['wandb_run_name'] if candidate or diagnostic_candidate_resume else (cfg['wandb']['smoke_run_name'] if smoke else cfg['wandb']['run_name'])
  if acceptance:run_name=cfg['formal_candidate']['wandb_run_name']
  argv=[str(ENTRY),f'--{a.role}',f'--env={cfg["environment"]}',f'--seed={cfg["seed"]}',f'--max_steps={steps}',f'--random_steps={cfg["training"]["random_steps"]}',f'--training_starts={cfg["training"]["training_starts"]}',f'--batch_size={cfg["training"]["total_batch_size"]}',f'--critic_actor_ratio={cfg["training"]["critic_actor_ratio"]}',f'--utd_ratio={cfg["training"]["utd_ratio"]}',f'--replay_buffer_capacity={cfg["training"]["replay_capacity"]}',f'--checkpoint_period={checkpoint_period}',f'--checkpoint_path={out}',f'--demo_path={ROOT/cfg["demo"]["path"]}',f'--ip={a.ip}',f'--exp_name={run_name}']
  argv.extend([f'--learner_start_step={resume["checkpoint_step"]+1 if resume else 0}',f'--actor_start_step={resume["actor_steps"] if resume else 0}'])
  if candidate:argv.append('--log_period=1')
  spec=importlib.util.spec_from_file_location('peg_delivered_rlpd',ENTRY);mod=importlib.util.module_from_spec(spec);sys.argv=argv;spec.loader.exec_module(mod)
  if a.role=='actor':
   from training.peg_insertion.actor_shutdown import install_bounded_actor_shutdown
   install_bounded_actor_shutdown(mod,out/'actor_shutdown.json',timeout_seconds=10)
  if branch and a.role=='actor':
   original_actor=mod.actor
   def branch_actor(*args,**kwargs):
    print('BRANCH_ACTOR_START_STEP:',resume['actor_steps'],flush=True)
    return original_actor(*args,**kwargs)
   mod.actor=branch_actor
   original_client=mod.TrainerClient
   class BranchClient(original_client):
    def recv_network_callback(self,callback):
     def observed(params):
      print('BRANCH_ACTOR_NETWORK_RECEIVED',flush=True)
      return callback(params)
     return super().recv_network_callback(observed)
   mod.TrainerClient=BranchClient
  servers=[]
  from training.peg_insertion.sync_policy import block_until_ready as per_leaf_sync,JaxView
  if a.role=='learner':mod.jax=JaxView(jax)
  if acceptance or a.profile=='formal':
   original_server=mod.TrainerServer
   class AcceptanceServer(original_server):
    def __init__(self,*args,**kw):super().__init__(*args,**kw);servers.append(self)
   mod.TrainerServer=AcceptanceServer
  diag={'update_calls':0,'concat_calls':0}
  candidate_state={'records':[],'sample_seconds':None,'total_start':None}
  candidate_updates_path=candidate_result_dir/'updates.jsonl' if candidate else None
  candidate_result_path=candidate_result_dir/'result.json' if candidate else None
  def write_json_atomic(path,payload):
   tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
  if candidate:
   original_timer=mod.Timer
   class CandidateTimer(original_timer):
    def tick(self,key):
     if key=='sample_replay_buffer':candidate_state['total_start']=time.perf_counter();candidate_state['sample_start']=candidate_state['total_start']
     return super().tick(key)
    def tock(self,key):
     result=super().tock(key)
     if key=='sample_replay_buffer':candidate_state['sample_seconds']=time.perf_counter()-candidate_state['sample_start']
     return result
   mod.Timer=CandidateTimer
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
   if branch and a.role=='learner':branch_policy.audit_state(agent.state,'state_restore')
   if a.profile=='formal' and a.role=='learner':
    from training.peg_insertion.formal_monitor import FormalMonitor
    from training.peg_insertion.update_observer import UpdateObserver
    import uuid
    def replay_progress():
     replay=holder['online']
     with replay._lock:return int(replay._size),int(replay.project_actor_steps)
    monitor=FormalMonitor(ROOT/cfg['outputs']['log_dir']/('formal_monitor_'+uuid.uuid4().hex),
      resume['checkpoint_step']+1 if resume else 0,
      rng=lambda value:np.asarray(value.state.rng,dtype=np.uint32).tolist(),
      sync=per_leaf_sync,replay=replay_progress,
      observer=lambda directory,index:UpdateObserver(directory,index,jax))
    print('FORMAL_MONITOR_DIRECTORY:',str(monitor.directory),flush=True)
    cls=type(agent);original_formal_update=cls.update_high_utd
    def formal_update(self,*args,**kwargs):
     return monitor.update(lambda:original_formal_update(self,*args,**kwargs),self)
    cls.update_high_utd=formal_update
   if candidate:
    cls=type(agent);original_candidate_update=cls.update_high_utd
    def candidate_update(self,batch,*,utd_ratio,pmap_axis=None):
     index=len(candidate_state['records']);rng_before=np.asarray(self.state.rng,dtype=np.uint32).tolist();started=time.perf_counter()
     if branch and index==0:branch_policy.audit_state(self.state,'state_before_first_update')
     observer=None
     if a.observe_update1:
      from training.peg_insertion.update_observer import UpdateObserver
      observer=UpdateObserver(candidate_result_dir,index,jax)
      if index<2:observer.signature('state_before',self.state);observer.signature('batch',batch)
      observer.mark('before_update_high_utd',rng_before=rng_before)
     def timeout_update():
      failure={'gate':'FAIL','reason':'update_timeout','failed_update':index,'timeout_seconds':cfg['formal_candidate']['update_timeout_seconds'],'profile':'smoke','formal_candidate':True,'diagnostic_enabled':False,'pid':os.getpid()}
      try:write_json_atomic(candidate_result_path,failure)
      finally:
       print(f'PEG_FORMAL_CANDIDATE_UPDATE_TIMEOUT: update={index} limit={cfg["formal_candidate"]["update_timeout_seconds"]}s',flush=True);os._exit(124)
     watchdog=threading.Timer(cfg['formal_candidate']['update_timeout_seconds'],timeout_update);watchdog.daemon=True;watchdog.start()
     if observer:observer.arm()
     try:
      dispatch_start=time.perf_counter()
      new_agent,info=original_candidate_update(self,batch,utd_ratio=utd_ratio,pmap_axis=pmap_axis)
      if observer:observer.mark('update_high_utd_python_return',dispatch_seconds=time.perf_counter()-dispatch_start);observer.mark('before_full_synchronization')
      sync_start=time.perf_counter()
      per_leaf_sync(new_agent);per_leaf_sync(info)
      if observer:observer.mark('synchronization_complete',sync_seconds=time.perf_counter()-sync_start)
     finally:
      watchdog.cancel()
      if observer:observer.close()
     update_seconds=time.perf_counter()-started;sample_seconds=float(candidate_state['sample_seconds']);total_seconds=time.perf_counter()-candidate_state['total_start'];rng_after=np.asarray(new_agent.state.rng,dtype=np.uint32).tolist()
     if observer:
      if index<2:observer.signature('state_after',new_agent.state)
      observer.mark('after_update_state',rng_after=rng_after)
     if rng_before==rng_after:raise RuntimeError('formal candidate RNG did not advance')
     if candidate_state['records'] and candidate_state['records'][-1]['rng_after']!=rng_before:raise RuntimeError('formal candidate RNG chain discontinuity')
     flat={}
     def scalars(prefix,value):
      if isinstance(value,Mapping):
       for key,item in value.items():scalars(f'{prefix}.{key}' if prefix else str(key),item)
      else:
       array=np.asarray(jax.device_get(value))
       if array.size==1:flat[prefix]=float(array.reshape(()))
     scalars('',info);losses={key:value for key,value in flat.items() if 'loss' in key.lower()}
     if not losses or not all(np.isfinite(value) for value in losses.values()):raise RuntimeError('formal candidate loss is missing or non-finite')
     global_index=resume['checkpoint_step']+1+index if resume else index
     record={'update':global_index,'sample_seconds':sample_seconds,'update_seconds':update_seconds,'total_seconds':total_seconds,'losses':losses,'losses_finite':True,'rng_before':rng_before,'rng_after':rng_after,'pid':os.getpid()}
     if branch:
      candidate_state['last_state']=new_agent.state
      if index==4:branch_policy.write_evidence('final_state_fingerprint.json',branch_policy.state_fingerprint(new_agent.state))
     with candidate_updates_path.open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
     candidate_state['records'].append(record)
     print(f'FORMAL_CANDIDATE_UPDATE: {index+1}/{cfg["formal_candidate"]["learner_updates"]} sample_seconds={sample_seconds:.6f} update_seconds={update_seconds:.6f} total_seconds={total_seconds:.6f}',flush=True)
     info=dict(info);info['formal_candidate_timing']={'sample_seconds':sample_seconds,'update_seconds':update_seconds,'total_seconds':total_seconds}
     return new_agent,info
    cls.update_high_utd=candidate_update
   if a.diagnostic_rng_from_step is not None:
    if not (resume and a.role=='learner' and a.diagnostic_no_save):raise ValueError('diagnostic RNG override requires learner resume with --diagnostic-no-save')
    reference=restore_fresh_state(a.resume_from.parent,agent.state,a.diagnostic_rng_from_step)
    agent=agent.replace(state=agent.state.replace(rng=reference.rng))
    print(f'[DIAG] RNG_OVERRIDDEN_FROM_STEP={a.diagnostic_rng_from_step}',flush=True)
   if a.diagnostic_rng_key is not None:
    if not (resume and a.role=='learner' and a.diagnostic_no_save):raise ValueError('diagnostic RNG key requires learner resume with --diagnostic-no-save')
    key_parts=[int(part) for part in a.diagnostic_rng_key.split(',')]
    if len(key_parts)!=2 or any(part<0 or part>2**32-1 for part in key_parts):raise ValueError('--diagnostic-rng-key requires two comma-separated uint32 values')
    explicit_rng=jax.numpy.asarray(key_parts,dtype=jax.numpy.uint32)
    agent=agent.replace(state=agent.state.replace(rng=explicit_rng))
    print(f'[DIAG] RNG_OVERRIDDEN_WITH_KEY={key_parts}',flush=True)
   if (a.diagnose_updates or a.split_utd1 or a.step_indexed_rng or a.diagnostic_save_after_update or a.diagnostic_stack_dumps) and a.role=='learner':
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
      stack_timers=[]
      if a.diagnostic_stack_dumps:
       def dump_stack(delay):
        print(f'[DIAG][update={idx}] python_stack_after={delay}s',file=sys.stderr,flush=True);faulthandler.dump_traceback(file=sys.stderr,all_threads=True)
       for delay in (30,90):
        timer=threading.Timer(delay,dump_stack,args=(delay,));timer.daemon=True;timer.start();stack_timers.append(timer)
      try:
       if a.split_utd1:
        if utd_ratio!=1:raise ValueError('--split-utd1 requires utd_ratio=1')
        phase=time.perf_counter();new_agent,critic_info=original_base_update(self,batch,pmap_axis=pmap_axis,networks_to_update=frozenset({'critic'}));per_leaf_sync(new_agent)
        print(f'[DIAG][update={idx}] after_critic dt={time.perf_counter()-phase:.6f}s',flush=True)
        phase=time.perf_counter();new_agent,actor_info=original_base_update(new_agent,batch,pmap_axis=pmap_axis,networks_to_update=frozenset({'actor','temperature'}));per_leaf_sync(new_agent)
        print(f'[DIAG][update={idx}] after_actor_temperature dt={time.perf_counter()-phase:.6f}s',flush=True)
        critic_info={k:v for k,v in critic_info.items() if k not in {'actor','temperature'}}
        actor_info={k:v for k,v in actor_info.items() if k!='critic'}
        info={**critic_info,**actor_info}
       else:new_agent,info=original_update(self,batch,utd_ratio=utd_ratio,pmap_axis=pmap_axis)
       per_leaf_sync(new_agent)
       jax.tree_util.tree_map(lambda x:x.block_until_ready() if hasattr(x,'block_until_ready') else x,info)
      finally:
       for timer in stack_timers:timer.cancel()
      print(f'[DIAG][update={idx}] after_update dt={time.perf_counter()-t0:.6f}s',flush=True)
      if a.diagnostic_save_after_update:
       global_step=resume['checkpoint_step']+1+idx
       print(f'[DIAG][checkpoint={global_step}] before_post_update_checkpoint t={time.time():.6f}',flush=True)
       phase=time.perf_counter();save_bundle(out,new_agent.state,holder['online'],global_step,int(holder['online'].project_actor_steps),bundle_cfg,MANIFEST,active['id'],{'identifier':'peg-project-launcher-post-update','diagnostic_rng_from_step':a.diagnostic_rng_from_step,'diagnostic_rng_key':a.diagnostic_rng_key,'step_indexed_rng':a.step_indexed_rng})
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
    if candidate and not acceptance:restore_replay(b,candidate_replay_path)
    elif resume:
     restore_replay(b,resume['online_replay_path'])
     if branch:branch_policy.audit_replay(b)
    else:install_actor_step_tracking(b,0)
   return b
  mod.make_replay_buffer=make_replay
  active={'id':None if branch else a.wandb_run_id}
  class Logger:
   def __init__(self):
    import wandb
    log_dir=candidate_log_dir if candidate else (ROOT/cfg['outputs'][('smoke_' if smoke else '')+'log_dir']).resolve()
    log_dir.mkdir(parents=True,exist_ok=True)
    kw={'project':cfg['wandb']['project'],'name':run_name,'mode':'online','dir':str(log_dir)}
    if branch:
     kw.update(id=json.loads((root/'launch.json').read_text())['branch_run_id'],resume='never',config=branch_policy.provenance())
    elif resume:kw.update(id=a.wandb_run_id,resume='must')
    elif a.profile=='formal':
     import uuid
     kw.update(id=uuid.uuid4().hex[:8],resume='never')
    self.run=wandb.init(**kw);active['id']=self.run.id;active['entity']=self.run.entity;active['run']=self.run
    if a.profile=='formal':
     from training.peg_insertion.formal_preflight import publish_session
     publish_session(self.run.id,a.resume_from,a.ip)
   def log(self,payload,step=None):
    global_step=step
    if branch and global_step is not None and global_step>30:return # bounded run has no update31
    if branch:payload={**payload,'learner_update':global_step}
    self.run.log(payload,step=global_step)
  mod.make_wandb_logger=(lambda **kw:None) if a.diagnostic_disable_wandb else (lambda **kw:Logger())
  def save(directory,state,step,keep=20,**kw):
   if 'online' not in holder:raise RuntimeError('online replay unavailable')
   global_step=int(step)
   if a.diagnose_updates:print(f'[DIAG][checkpoint={global_step}] before_checkpoint t={time.time():.6f}',flush=True)
   source={'identifier':'peg-formal-candidate-continuous' if candidate else 'peg-project-launcher','profile':a.profile,'formal_candidate':candidate,'diagnostic_enabled':a.profile=='diagnostic'}
   if branch:source.update(branch_policy.provenance())
   if a.profile=='formal':
    gate=require_gate()
    source['formal_binding']={'source_hashes':gate['source_hashes'],'bindings':gate['bindings']}
   t0=time.perf_counter();result=str(save_bundle(Path(directory),state,holder['online'],global_step,int(holder['online'].project_actor_steps),bundle_cfg,MANIFEST,active['id'],source))
   if a.diagnose_updates:print(f'[DIAG][checkpoint={global_step}] after_checkpoint dt={time.perf_counter()-t0:.6f}s',flush=True)
   return result
  mod.checkpoints.save_checkpoint=save
  if candidate:
   original_learner=mod.learner
   def candidate_learner(*args,**kwargs):
    result=original_learner(*args,**kwargs);records=candidate_state['records'];last_saved=30 if branch else (25 if acceptance else 19)
    if branch:
     if len(records)!=5:raise RuntimeError('branch requires exactly five updates before final checkpoint')
     save(out,candidate_state['last_state'],30)
    final_cp=out/f'checkpoint_{last_saved}';meta=out/f'checkpoint_{last_saved}_resume_metadata.json';replay=out/f'checkpoint_{last_saved}_online_replay.pkl'
    if len(records)!=cfg['formal_candidate']['learner_updates'] or not all(path.is_file() for path in (final_cp,meta,replay)):raise RuntimeError('formal candidate completion/checkpoint failure')
    summary={'gate':'PENDING_VERIFICATION','profile':a.profile,'formal_candidate':True,'fresh_initialization':True,'single_learner_process':True,'learner_process_pid':os.getpid(),'process_launch_count':1,'normal_rng_mode':True,'diagnostic_enabled':False,'updates_completed':len(records),'updates_expected':cfg['formal_candidate']['learner_updates'],'update_timeout_seconds':cfg['formal_candidate']['update_timeout_seconds'],'max_update_seconds':max(item['update_seconds'] for item in records),'all_losses_finite':all(item['losses_finite'] for item in records),'checkpoint':str(final_cp),'wandb_entity':active.get('entity'),'wandb_project':cfg['wandb']['project'],'wandb_name':run_name,'wandb_run_id':active['id'],'replay_source':str(candidate_replay_path),'replay_source_sha256':sha(candidate_replay_path) if candidate_replay_path else None}
    if active.get('run') is not None:active['run'].finish()
    if branch:summary.update(branch_policy.provenance(),fresh_initialization=False,per_leaf_sync=True,source_actor_steps=resume['actor_steps'],final_actor_steps=int(holder['online'].project_actor_steps))
    write_json_atomic(candidate_result_path,summary);print('PEG_FORMAL_CANDIDATE_CONTINUOUS_RUN: COMPLETE',flush=True);return result
   mod.learner=candidate_learner
  print('TRAINING_PROFILE:',a.profile);print('FORMAL_CANDIDATE:',candidate);print('DIAGNOSTIC_ENABLED:',a.profile=='diagnostic');print('FRESH_INITIALIZATION:',not bool(resume));print('LOAD_LEGACY_CHECKPOINT: FALSE');print('RESUME_PROFILE:',selected_profile);print('COMMAND_ARGS:',json.dumps(argv[1:]))
  try:mod.app.run(mod.main)
  finally:
   for server in servers:server.stop()
   if a.profile=='formal' and active.get('run') is not None:active['run'].finish(exit_code=wandb_exit_code(sys.exc_info()[1]))
if __name__=='__main__':raise SystemExit(main())
