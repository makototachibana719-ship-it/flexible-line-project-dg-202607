#!/usr/bin/env python3
"""Stages A-G: C2 causal review and existing-data representation test."""
from __future__ import annotations
import hashlib,json,math,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,jax.numpy as jnp,numpy as np,optax
from async_sac_state_sim import concat_rlpd_batches
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_observation_gated_residual_c1_pilot as c1
import scripts.run_peg_f3_v2_progressive_frozen_residual_v2_c2_pilot as pilot
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture,restore,transition_record
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model,load_config,make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.common.optimizers import make_optimizer
from training.peg_insertion.observation_gated_residual_v1 import RESIDUAL_DEF,gate_value,residual_output
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate,progressive_mode,support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

PILOT=ROOT/'results/training/peg_f3_v2_progressive_frozen_residual_v2_c2_pilot_20260908_002'
PRE=ROOT/'results/preflight/peg/f3_v2_progressive_frozen_residual_v2_20260908_002'
CP=PILOT/'checkpoints/c2_learner_0250_residual_062.msgpack'
REPLAY=PILOT/'online_replay.pkl'
OUT=ROOT/'results/diagnostics/peg/f3_v2_c2_data_policy_capacity_review_20260908_003'
LIMIT=np.asarray([.002,.003,.05],np.float64);EPS=1e-6
FIT_RAW_MAE_MAX=.10;FIT_ACTION_MAE_MAX=.05;FIT_COSINE_MIN=.90;FIT_PROBE_REDUCTION=.50
TX=make_optimizer(learning_rate=3e-4,warmup_steps=2000)

def req(x,m):
 if not x:raise RuntimeError(m)
def jv(x):
 if isinstance(x,np.ndarray):return jv(x.tolist())
 if isinstance(x,(np.bool_,bool)):return bool(x)
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.floating):return float(x)
 if isinstance(x,dict):return {str(k):jv(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jv(v) for v in x]
 return x
def dump(p,x):
 with Path(p).open('x') as f:json.dump(jv(x),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def thash(t):
 h=hashlib.sha256()
 for p,x in jax.tree_util.tree_flatten_with_path(t)[0]:
  a=np.asarray(jax.device_get(x));h.update(jax.tree_util.keystr(p).encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def init_c1(common):
 r=RESIDUAL_DEF.init(jax.random.PRNGKey(260910),np.zeros((1,20),np.float32))['params'];return {'agent_state':common.state,'residual_params':r,'residual_opt_state':c1.RESIDUAL_TX.init(r)}
def init_c2(common):
 old=init_c1(common);r=RESIDUAL_DEF.init(jax.random.PRNGKey(260920),np.zeros((1,20),np.float32))['params'];return {'agent_state':old['agent_state'],'c1_params':old['residual_params'],'c1_opt_state':old['residual_opt_state'],'c2_params':r,'c2_opt_state':TX.init(r)}
def frozen_components(agent,c1p,obs):
 o=jnp.asarray(obs);base=agent.forward_policy(o,rng=None,train=False);loc=base.distribution.loc;g1=gate_value(o);c1r=residual_output(c1p,o);floc=loc.at[...,:6].add(g1[...,None]*c1r);return loc,floc,jnp.tanh(floc)
def demo_nearest(obs,demo_obs,demo_act,scale):
 d=np.sqrt(np.sum(((demo_obs-np.asarray(obs))/scale)**2,axis=1));i=int(np.argmin(d));return i,float(d[i]),demo_act[i].copy()
def episode_provenance(demos):
 out=[];ep=step=0
 for i,x in enumerate(demos):
  out.append((ep,step));step+=1
  if bool(x['dones']):ep+=1;step=0
 return out
def before_metrics(env):
 v=geo.pre_metrics(env);return {'position':float(np.linalg.norm(np.asarray(env.unwrapped._get_obs()['tcp_pos']) if False else np.zeros(3))), 'lateral':float(v[0]),'axial':float(v[1]),'orientation':float(v[2]),'success_margin':float(np.max(v/LIMIT))}
def branch(env_level,snap,action):
 e=make_train_env(env_level)
 try:
  restore(e,snap);n,r,t,tr,info=e.step(np.asarray(action,np.float32));rec=transition_record(e,1,n,action,info,t,tr);link,bad,pairs=contact_flags(e)
  return {k:rec[k] for k in ('lateral_error','axial_error','orientation_error','success_margin','force','table_ee_contact')}|{'position':float(np.linalg.norm(np.asarray(n)[13:16])),'unexpected_contact':bool(link or bad),'contact_pairs':pairs}
 finally:e.close()
def improvement(cur,exp):
 return bool(exp['success_margin']<cur['success_margin']-1e-6 and exp['position']<=cur['position']+1e-6 and exp['force']<=cur['force']+5.)
def fit_metrics(params,obs,target_raw,agent,c1p,support,scale,core,outer,target_action):
 pred=np.asarray(jax.device_get(residual_output(params,jnp.asarray(obs))),np.float32);raw_mae=float(np.mean(np.abs(pred-target_raw)));act=np.asarray(jax.device_get(progressive_mode(agent,c1p,params,jnp.asarray(obs),support,scale,core,outer)),np.float32)[:,:6];tar=target_action[:,:6];action_mae=float(np.mean(np.abs(act-tar)));den=np.linalg.norm(act,axis=1)*np.linalg.norm(tar,axis=1);cos=np.divide(np.sum(act*tar,axis=1),den,out=np.zeros_like(den),where=den>1e-9);return {'raw_residual_mae':raw_mae,'action_mae':action_mae,'action_cosine':float(np.mean(cos)),'per_axis_action_mae':np.mean(np.abs(act-tar),axis=0),'per_axis_raw_mae':np.mean(np.abs(pred-target_raw),axis=0)}
@jax.jit
def bc_step(params,opt_state,obs,target):
 def lossfn(p):return jnp.mean((residual_output(p,obs)-target)**2)
 loss,g=jax.value_and_grad(lossfn)(params);up,no=TX.update(g,opt_state,params);return optax.apply_updates(params,up),no,loss,optax.global_norm(g)

def main():
 req(Path.cwd()==ROOT and not OUT.exists() and jax.default_backend()=='gpu','fresh/cwd/GPU');OUT.mkdir(parents=True);(OUT/'checkpoints').mkdir();started=time.time()
 before={'production_demo':sha(p4.DEMO),'corrective':sha(p4.CORR),'local':sha(p4.LOCAL),'pilot_checkpoint':sha(CP),'pilot_replay':sha(REPLAY)};dump(OUT/'protected_before.json',before);req(before['production_demo']==p4.DSHA,'Demo SHA')
 result0=json.loads((PILOT/'result.json').read_text());pre=json.loads((PRE/'result.json').read_text());req(result0['PROGRESSIVE_FROZEN_RESIDUAL_V2']=='FAIL' and result0['PRIMARY_ROOT_CAUSE']=='C2_RESIDUAL_DOES_NOT_ACQUIRE_EARLY_APPROACH_POLICY','authoritative _002')
 cfg=load_config(2);demos=pickle.loads(p4.DEMO.read_bytes());demo_obs,dscale=distance_model(demos);demo_act=np.asarray([x['actions'] for x in demos],np.float32);prov=episode_provenance(demos)
 sup=pickle.loads((PRE/'C1_CONFIRMED_SUPPORT_V1.pkl').read_bytes());support=np.asarray(sup['observations'],np.float32);scale=np.asarray(sup['normalization_scale'],np.float32);core=float(pre['C1_CORE_THRESHOLD']);outer=float(pre['C1_SHELL_OUTER_THRESHOLD'])
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  common,_,_=geo.build_common_start(cfg,demos);loaded=flax.serialization.from_bytes(init_c2(common),CP.read_bytes());agent=common.replace(state=loaded['agent_state']);c1p=loaded['c1_params'];c2p=loaded['c2_params'];base_sha=geo.thash(agent.state.params['modules_actor']);c1sha=geo.thash(c1p);baseguard=thash(agent.state.params['modules_actor']);c1guard=thash(c1p);baseopt=thash(agent.state.opt_states['actor']);basetarget=thash(agent.state.target_params['modules_actor']);req(base_sha==pre['C0_BASE_SHA'] and c1sha==pre['C1_RESIDUAL_SHA'],'frozen stack')

  # A: reproduce the corrected final C2 trajectory and select immutable pre-contact probes.
  e=make_train_env('C2');rows=[];snaps=[]
  try:
   obs,_=e.reset(seed=100)
   for step in range(500):
    snap=capture(e,obs);v=geo.pre_metrics(e);pos=float(np.linalg.norm(np.asarray(obs)[13:16]));frozen=np.asarray(jax.device_get(frozen_components(agent,c1p,jnp.asarray(obs))[2]),np.float32);raw=np.asarray(jax.device_get(residual_output(c2p,jnp.asarray(obs))),np.float32);g=float(np.asarray(jax.device_get(progressive_gate(jnp.asarray(obs),support,scale,core,outer))));final=np.asarray(jax.device_get(progressive_mode(agent,c1p,c2p,jnp.asarray(obs),support,scale,core,outer)),np.float32);dd=p4.demo_distance(obs,demo_obs,dscale);n,reward,t,tr,info=e.step(final);link,bad,pairs=contact_flags(e);nd=p4.demo_distance(n,demo_obs,dscale)
    rows.append({'source_episode':100,'source_step':step,'observation':np.asarray(obs,np.float32),'g2':g,'frozen_action':frozen,'current_raw_c2_residual':raw,'final_physical_action':final,'position':pos,'lateral':float(v[0]),'axial':float(v[1]),'orientation':float(v[2]),'success_margin':float(np.max(v/LIMIT)),'demo_distance':dd,'next_demo_distance':nd,'next_position':float(np.linalg.norm(np.asarray(n)[13:16])),'unexpected_contact_after':bool(link or bad),'contact_pairs_after':pairs,'tags':[]});snaps.append(snap);obs=n
    if t or tr:break
  finally:e.close()
  div=next((i for i,x in enumerate(rows) if x['next_demo_distance']>p4.TH),len(rows)-1);contact=next((i for i,x in enumerate(rows) if x['unexpected_contact_after']),len(rows));pre_end=max(1,contact);best=min(range(pre_end),key=lambda i:rows[i]['position']);growth=max(range(1,pre_end),key=lambda i:rows[i]['next_position']-rows[i]['position']);ids={0,'reset'} if False else set()
  tags={0:['c2_reset'],max(0,div-2):['pre_first_divergence'],div:['first_divergence'],min(5,pre_end-1):['early_approach'],best:['best_position'],growth:['pre_bad_trajectory_growth'],max(0,contact-1):['pre_unexpected_contact']}
  selected=[];selected_snaps=[]
  for i in sorted(tags):
   if i>=pre_end:continue
   x=dict(rows[i]);x['tags']=tags[i];selected.append(x);selected_snaps.append(snaps[i])

  # B: existing frozen expert diagnostic labels and exact representability.
  relabeled=[]
  for x in selected:
   idx,dd,expert=demo_nearest(x['observation'],demo_obs,demo_act,dscale);_,floc,_=frozen_components(agent,c1p,jnp.asarray(x['observation']));floc=np.asarray(jax.device_get(floc),np.float32);g=x['g2'];target_loc=np.arctanh(np.clip(expert[:6],-.999999,.999999));full_gate_raw=target_loc-floc[:6];raw_exact=full_gate_raw/g if g>EPS else np.full(6,np.inf);raw_action=(expert[:6]-x['frozen_action'][:6])/g if g>EPS else np.full(6,np.inf);representable=bool(g>EPS and np.all(np.abs(raw_exact)<=1.+1e-6));full_representable=bool(np.all(np.abs(full_gate_raw)<=1.+1e-6));authority=bool(g>EPS and (not full_representable or representable));target=x['final_physical_action'].copy();target[:6]=np.tanh(floc[:6]+g*np.clip(raw_exact,-1,1));y=dict(x);y.update(expert_action=expert,expert_source_row=idx,expert_source_episode=prov[idx][0],expert_source_step=prov[idx][1],expert_demo_distance=dd,expert_binding='exact_C2_source_transition' if x['source_step']==0 and idx==42 else 'production_demo_nearest_normalized_Obs20',required_effective_delta=expert[:6]-x['frozen_action'][:6],required_raw_action_residual=raw_action,required_raw_c2_residual=raw_exact,required_full_gate_c2_residual=full_gate_raw,required_effective_delta_norm=float(np.linalg.norm(expert[:6]-x['frozen_action'][:6])),required_raw_c2_residual_norm=float(np.linalg.norm(raw_exact)),required_residual_representable=representable,full_gate_residual_representable=full_representable,gate_authority_sufficient=authority,expert_action_saturated=bool(np.any(np.abs(expert[:6])>=.999)),expert_composed_action=target);relabeled.append(y)
  probe_path=OUT/'C2_BEHAVIOR_CRITICAL_PROBE_V1.pkl';probe_path.write_bytes(pickle.dumps(relabeled,pickle.HIGHEST_PROTOCOL));probe_sha=sha(probe_path);represent='YES' if all(x['required_residual_representable'] for x in relabeled) else 'NO' if not any(x['required_residual_representable'] for x in relabeled) else 'MIXED';authority='YES' if all(x['gate_authority_sufficient'] for x in relabeled) else 'NO' if not any(x['gate_authority_sufficient'] for x in relabeled) else 'MIXED'
  dump(OUT/'C2_BEHAVIOR_CRITICAL_PROBE_V1_manifest.json',{'name':'C2_BEHAVIOR_CRITICAL_PROBE_V1','sha256':probe_sha,'rows':len(relabeled),'source':'deterministic replay of corrected _002 final checkpoint','pre_contact_only':True,'tags':[x['tags'] for x in relabeled]})

  # C: exact integration/Python-state one-step branches.
  branches=[]
  for x,snap in zip(relabeled,selected_snaps):
   cur=branch('C2',snap,x['final_physical_action']);exp=branch('C2',snap,x['expert_composed_action']);branches.append({'source_step':x['source_step'],'tags':x['tags'],'current':cur,'expert_correction':exp,'improves':improvement(cur,exp)})
  improve_fraction=float(np.mean([x['improves'] for x in branches]))

  # D: one ordinary current SAC C2 actor update, attribution only.
  replay=pickle.loads(REPLAY.read_bytes());rd=replay['dataset_dict'];rsize=int(replay['size']);rng=np.random.default_rng(260940);ri=rng.integers(0,rsize,32);di=rng.integers(0,len(demos),32);batch={k:jnp.asarray(np.concatenate((np.asarray(rd[k])[:rsize][ri],np.asarray([d[k] for d in demos])[di]),axis=0)) for k in ('observations','actions','next_observations','rewards','masks','dones')};before_actions=np.asarray([x['final_physical_action'] for x in relabeled],np.float32);agent2,c2after,_,ainfo=pilot.c2_update(agent,c1p,c2p,loaded['c2_opt_state'],support,scale,core,outer,batch);after_actions=np.asarray(jax.device_get(progressive_mode(agent2,c1p,c2after,np.asarray([x['observation'] for x in relabeled]),support,scale,core,outer)),np.float32);direction=after_actions[:,:6]-before_actions[:,:6];toexpert=np.asarray([x['expert_action'][:6] for x in relabeled])-before_actions[:,:6];den=np.linalg.norm(direction,axis=1)*np.linalg.norm(toexpert,axis=1);cos=np.divide(np.sum(direction*toexpert,axis=1),den,out=np.zeros_like(den),where=den>1e-12);sac_useful='YES' if float(np.median(cos))>0 and float(np.mean(cos>0))>=.75 else 'NO' if float(np.median(cos))<=0 and float(np.mean(cos>0))<=.25 else 'MIXED'
  actor_effect=[]
  for x,snap,a in zip(relabeled,selected_snaps,after_actions):
   cur=branch('C2',snap,x['final_physical_action']);aft=branch('C2',snap,a);actor_effect.append({'source_step':x['source_step'],'success_margin_change':aft['success_margin']-cur['success_margin'],'position_change':aft['position']-cur['position'],'force_change':aft['force']-cur['force']})

  # E: corrected replay coverage, using only actor-visible Obs20 geometry.
  ro=np.asarray(rd['observations'])[:rsize];rn=np.asarray(rd['next_observations'])[:rsize];rdd=np.min(np.sqrt(np.sum(((ro[:,None,:]-demo_obs[None,:,:])/dscale)**2,axis=2)),axis=1);rc1=np.asarray(jax.device_get(support_distance(ro,support,scale)));nc1=np.asarray(jax.device_get(support_distance(rn,support,scale)));rg=np.asarray(jax.device_get(progressive_gate(ro,support,scale,core,outer)));rpos=np.linalg.norm(ro[:,13:16],axis=1);near=(np.abs(ro[:,13])<=.020)&(np.linalg.norm(ro[:,14:16],axis=1)<=.010);useful=(rpos>=.075)&(rpos<=.110)&(nc1<rc1)&(rdd<=p4.TH);probe_obs=np.asarray([x['observation'] for x in relabeled]);prec=np.min(np.sqrt(np.sum(((ro[:,None,:]-probe_obs[None,:,:])/dscale)**2,axis=2)),axis=1)<=.10
  coverage={'rows':rsize,'pre_divergence_fraction':float(np.mean(rdd<=p4.TH)),'post_divergence_fraction':float(np.mean(rdd>p4.TH)),'c1_shell_like_fraction':float(np.mean((rc1>core)&(rc1<=outer))),'c1_core_like_fraction':float(np.mean(rc1<=core)),'production_demo_support_fraction':float(np.mean(rdd<=p4.TH)),'near_hole_fraction':float(np.mean(near)),'unexpected_contact_precursor_fraction':float(np.mean(prec)),'useful_corridor_count':int(np.sum(useful)),'useful_corridor_fraction':float(np.mean(useful)),'nearest_demo_distance_quantiles':{str(q):float(np.quantile(rdd,q)) for q in (0,.25,.5,.75,.95,1)}};data_coverage='SUFFICIENT' if int(np.sum(useful))>=64 else 'INSUFFICIENT' if int(np.sum(useful))<32 else 'MIXED'

  # F: immutable subset from production Demo only; fixed C2 position band and C1-directed motion.
  do=np.asarray([x['observations'] for x in demos],np.float32);dn=np.asarray([x['next_observations'] for x in demos],np.float32);dc=np.asarray(jax.device_get(support_distance(do,support,scale)));dnc=np.asarray(jax.device_get(support_distance(dn,support,scale)));dp=np.linalg.norm(do[:,13:16],axis=1);subset=[]
  for i in np.where((dp>=.075)&(dp<=.110)&(dnc<dc))[0]:
   g=float(np.asarray(jax.device_get(progressive_gate(do[i],support,scale,core,outer))));_,floc,_=frozen_components(agent,c1p,do[i]);floc=np.asarray(jax.device_get(floc));raw=(np.arctanh(np.clip(demo_act[i,:6],-.999999,.999999))-floc[:6])/g if g>EPS else np.full(6,np.inf);subset.append({'observations':do[i],'actions':demo_act[i],'next_observations':dn[i],'source_row':int(i),'source_episode':prov[i][0],'source_step':prov[i][1],'g2':g,'D_C1':float(dc[i]),'next_D_C1':float(dnc[i]),'required_raw_c2_residual':raw,'representable':bool(g>EPS and np.all(np.abs(raw)<=1.+1e-6))})
  subset_path=OUT/'C2_EXISTING_EXPERT_SUBSET_V1.pkl';subset_path.write_bytes(pickle.dumps(subset,pickle.HIGHEST_PROTOCOL));subset_sha=sha(subset_path);dump(OUT/'C2_EXISTING_EXPERT_SUBSET_V1_manifest.json',{'name':'C2_EXISTING_EXPERT_SUBSET_V1','sha256':subset_sha,'rows':len(subset),'source_episodes':sorted(set(x['source_episode'] for x in subset)),'selection':'0.075<=norm(obs[13:16])<=0.110 and next D_C1 < current D_C1','production_demo_sha256':p4.DSHA,'production_demo_modified':False})

  # G: fresh scratch fit. No behavior selection and no environment rollout.
  offline='NOT_RUN';fit_reports=[];best=None
  if represent=='YES' and authority=='YES' and subset and all(x['representable'] for x in subset):
   train=[x for x in subset if x['source_episode']<9];valid=[x for x in subset if x['source_episode']>=9];req(train and valid,'episode split');to=np.asarray([x['observations'] for x in train]);tt=np.asarray([x['required_raw_c2_residual'] for x in train]);vo=np.asarray([x['observations'] for x in valid]);vt=np.asarray([x['required_raw_c2_residual'] for x in valid]);va=np.asarray([x['actions'] for x in valid]);po=np.asarray([x['observation'] for x in relabeled]);pt=np.asarray([x['required_raw_c2_residual'] for x in relabeled]);pa=np.asarray([x['expert_action'] for x in relabeled]);params=RESIDUAL_DEF.init(jax.random.PRNGKey(260950),np.zeros((1,20),np.float32))['params'];opt=TX.init(params);rr=np.random.default_rng(260951);baseline_probe=fit_metrics(params,po,pt,agent,c1p,support,scale,core,outer,pa)['action_mae']
   for step in range(1,1001):
    ii=rr.integers(0,len(train),64);params,opt,loss,gn=bc_step(params,opt,jnp.asarray(to[ii]),jnp.asarray(tt[ii]))
    if step in (250,500,1000):
     vm=fit_metrics(params,vo,vt,agent,c1p,support,scale,core,outer,va);pm=fit_metrics(params,po,pt,agent,c1p,support,scale,core,outer,pa);fit_reports.append({'step':step,'loss':float(loss),'gradient_norm':float(gn),'validation':vm,'probe':pm})
   best=min(fit_reports,key=lambda x:x['validation']['action_mae']);v=best['validation'];offline='GOOD' if v['raw_residual_mae']<=FIT_RAW_MAE_MAX and v['action_mae']<=FIT_ACTION_MAE_MAX and v['action_cosine']>=FIT_COSINE_MIN and best['probe']['action_mae']<=FIT_PROBE_REDUCTION*baseline_probe else 'POOR'
  else:offline='NOT_AUTHORIZED'

  req(thash(agent.state.params['modules_actor'])==baseguard and thash(agent.state.opt_states['actor'])==baseopt and thash(agent.state.target_params['modules_actor'])==basetarget and thash(c1p)==c1guard,'frozen mutation')
  if represent!='YES' or authority!='YES':primary='C2_RESIDUAL_ACTION_AUTHORITY_LIMIT';nxt='C2_RESIDUAL_OUTPUT_PARAMETERIZATION_REVIEW'
  elif offline=='POOR':primary='C2_RESIDUAL_CAPACITY_FAILURE';nxt='ONE_MINIMAL_C2_ARCHITECTURE_REVIEW'
  elif offline=='GOOD':primary='C2_DATA_OR_OBJECTIVE_ACQUISITION_FAILURE';nxt='C2_EXISTING_EXPERT_RESIDUAL_BC_V1'
  else:primary='C2_EXISTING_DATA_TARGETS_NOT_REPRESENTABLE';nxt='C2_RESIDUAL_OUTPUT_PARAMETERIZATION_REVIEW'
  result={'C2_DATA_POLICY_CAPACITY_REVIEW':'PASS' if offline=='GOOD' else 'STOP','C2_PROBE_SHA':probe_sha,'C2_EXISTING_EXPERT_SUBSET_SHA':subset_sha,'C2_REQUIRED_ACTION_REPRESENTABLE':represent,'C2_GATE_AUTHORITY_SUFFICIENT':authority,'EXPERT_CORRECTION_ONE_STEP_IMPROVES':improve_fraction,'one_step_branches':branches,'ACTOR_UPDATE_TO_EXPERT_COSINE':{'mean':float(np.mean(cos)),'median':float(np.median(cos)),'worst_quartile':float(np.quantile(cos,.25)),'positive_fraction':float(np.mean(cos>0))},'STANDARD_SAC_DIRECTION_USEFUL':sac_useful,'actor_one_update_effect':actor_effect,'C2_REPLAY_USEFUL_CORRIDOR_FRACTION':coverage['useful_corridor_fraction'],'C2_EXISTING_DATA_COVERAGE':data_coverage,'replay_coverage':coverage,'C2_RESIDUAL_OFFLINE_FIT':offline,'offline_fit_thresholds':{'raw_mae_max':FIT_RAW_MAE_MAX,'action_mae_max':FIT_ACTION_MAE_MAX,'cosine_min':FIT_COSINE_MIN,'probe_mae_ratio_max':FIT_PROBE_REDUCTION},'offline_fit_reports':fit_reports,'PRIMARY_C2_FAILURE_MODE':primary,'C2_EXISTING_EXPERT_BC_USED':'NO','C2_ON_POLICY_CORRECTIVE_USED':'NO','C2_ACQUISITION':'NOT_RUN','C2_MATCHED_CONFIRMATION':'NOT_RUN','C2_SKILL_CONFIRMED':'NO','CANONICAL_INTERFACE_AUDIT':'NOT_RUN','PEG_SINGLE_TASK_CONVERGED':'NO','FIRST_BAD_GATE':None if offline=='GOOD' else 'C2_REPRESENTATION_OR_OFFLINE_FIT','PRIMARY_ROOT_CAUSE':primary,'NEXT':nxt,'C0_BASE_SHA':base_sha,'C1_RESIDUAL_SHA':c1sha,'PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':p4.TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','PROTECTED_ARTIFACTS_CHANGED':'NO','TRAINING_RUNNING':False,'RESIDUAL_PROCESSES':'NONE','elapsed_seconds':time.time()-started}
 after={'production_demo':sha(p4.DEMO),'corrective':sha(p4.CORR),'local':sha(p4.LOCAL),'pilot_checkpoint':sha(CP),'pilot_replay':sha(REPLAY)};req(after==before,'protected changed');dump(OUT/'protected_after.json',after);dump(OUT/'result.json',result);print(json.dumps(jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
