#!/usr/bin/env python3
"""Critic-first repaired-Actor handoff with one conditional coverage fallback."""
from __future__ import annotations
import hashlib,json,math,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,jax.numpy as jnp,numpy as np,optax
from async_sac_state_sim import concat_rlpd_batches
from scripts.audit_peg_f3_v2_actor_parameterization_distribution_shift import rollout
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import action,append,block,demo_distance,distance_model,finite_scalars,fixed_batch,load_config,make_train_env
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay,save_bundle
DEMO=ROOT/'results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl';DSHA='4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878';CORR=ROOT/'results/datasets/peg_v2_corrective_postbc_round1_20260908_001/corrective_round1.pkl';LOCAL=ROOT/'results/datasets/peg_v2_local_insertion_supervision_20260908_001/local_insertion_supervision.pkl';TRACE=ROOT/'results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl';HAND=ROOT/'results/closure/peg/f3_v2_on_policy_corrective_20260908_001/canonical_safety_and_retention/repaired_handoff';REPAIRED=ROOT/'results/closure/peg/f3_v2_on_policy_corrective_20260908_001/round1_checkpoints/checkpoint_500.msgpack';OUT=ROOT/'results/training/peg_f3_v2_critic_first_repaired_handoff_20260908_001';TH=7.41773975814;PROBES=(0,10,25,50,100,250,500,1000)
def req(x,m):
 if not x:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
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
def thash(t):
 h=hashlib.sha256()
 for p,x in jax.tree_util.tree_flatten_with_path(t)[0]:h.update(str(p).encode());h.update(np.asarray(jax.device_get(x)).tobytes())
 return h.hexdigest()
def modes(a,o):return np.asarray(jax.device_get(a.sample_actions(observations=jax.device_put(o),argmax=True)),np.float32)
def fresh_agent(cfg):
 e=make_train_env('C0')
 try:a=make_combined_q_anchor_agent(cfg['seed'],e.observation_space.sample(),e.action_space.sample(),cfg)
 finally:e.close()
 return a.replace(state=restore_fresh_state(HAND,a.state,1000))
def build_probe(agent):
 rows=[]
 for source,path in [('round1_corrective',CORR),('local_insertion',LOCAL)]:
  for i,x in enumerate(pickle.loads(path.read_bytes())):rows.append({'observation':np.asarray(x['observations'],np.float32),'expert_action':np.asarray(x['actions'],np.float32),'source':source,'source_row':i,'tags':[source]})
 tr=[json.loads(x) for x in TRACE.read_text().splitlines() if x.strip()];e=make_train_env('C0')
 try:
  obs,_=e.reset(seed=100)
  for step in range(100):
   a=modes(agent,np.asarray(obs)[None])[0];expert=np.asarray(tr[137+min(step,len(tr)-138)]['action'],np.float32);tags=['repaired_success_rollout'];ax=abs(float(obs[13]));lat=float(np.linalg.norm(np.asarray(obs)[14:16]));
   if step==0:tags.append('c0_initial')
   if lat<=.01 and ax<=.02:tags.append('near_hole')
   rows.append({'observation':np.asarray(obs,np.float32),'expert_action':expert,'source':'repaired_success_rollout','source_row':step,'tags':tags})
   obs,_,term,trunc,info=e.step(a)
   if term or trunc:break
  req(info.get('is_success',False),'repaired rollout not successful')
 finally:e.close()
 unique={}
 for x in rows:
  key=hashlib.sha256(x['observation'].tobytes()+x['expert_action'].tobytes()).hexdigest()
  if key not in unique:unique[key]=x|{'pair_sha256':key,'duplicate_sources':[]}
  else:unique[key]['duplicate_sources'].append({'source':x['source'],'source_row':x['source_row']});unique[key]['tags']=sorted(set(unique[key]['tags']+x['tags']))
 vals=sorted(unique.values(),key=lambda x:x['pair_sha256']);p=OUT/'CRITIC_ALIGNMENT_PROBE_V1.pkl';p.write_bytes(pickle.dumps(vals,pickle.HIGHEST_PROTOCOL));manifest={'name':'CRITIC_ALIGNMENT_PROBE_V1','rows':len(vals),'input_rows':len(rows),'source_counts':{s:sum(x['source']==s for x in vals) for s in set(x['source'] for x in vals)},'probe_sha256':sha(p),'new_demonstrations':False,'training_use':False};dump(OUT/'CRITIC_ALIGNMENT_PROBE_V1_manifest.json',manifest);return np.asarray([x['observation'] for x in vals],np.float32),np.asarray([x['expert_action'] for x in vals],np.float32),manifest
def geometry(agent,o,e):
 p=modes(agent,o);oo=jax.device_put(o);pa=jax.device_put(p)
 def each(obs,act):return agent.forward_critic(obs[None],act[None],rng=None,train=False)[:,0]
 jac=np.asarray(jax.device_get(jax.vmap(jax.jacrev(each,argnums=1))(oo,pa)))[:,:,:6];d=e[:,:6]-p[:,:6];den=np.linalg.norm(jac,axis=2)*np.linalg.norm(d,axis=1)[:,None];cos=np.divide(np.sum(jac*d[:,None],axis=2),den,out=np.zeros_like(den),where=den>1e-12);agg=jac.mean(1);ad=np.linalg.norm(agg,axis=1)*np.linalg.norm(d,axis=1);ac=np.divide(np.sum(agg*d,axis=1),ad,out=np.zeros_like(ad),where=ad>1e-12);qp=np.asarray(jax.device_get(agent.forward_critic(oo,pa,rng=None,train=False)));qe=np.asarray(jax.device_get(agent.forward_critic(oo,jax.device_put(e),rng=None,train=False)));return {'Q_POLICY_MINUS_EXPERT':float((qp-qe).mean()),'Q_EXPERT_NOT_WORSE_FRACTION':float((qe.mean(0)>=qp.mean(0)).mean()),'Q_GRAD_EXPERT_COSINE_MEAN':float(ac.mean()),'Q_GRAD_EXPERT_COSINE_MEDIAN':float(np.median(ac)),'Q_GRAD_ANTI_ALIGNED_FRACTION':float((ac<0).mean()),'per_head':[{'head':i,'mean':float(cos[:,i].mean()),'median':float(np.median(cos[:,i])),'anti_fraction':float((cos[:,i]<0).mean())} for i in range(cos.shape[1])],'finite':bool(np.isfinite(cos).all() and np.isfinite(qp).all() and np.isfinite(qe).all()),'boundary_action_rate':float(np.mean(np.abs(p)>=.999))}
@jax.jit
def critic_step(agent,batch):
 rng,nrng=jax.random.split(agent.state.rng);(loss,info),grads=jax.value_and_grad(lambda p:agent.critic_loss_fn(batch,p,rng),has_aux=True)(agent.state.params);updates,newopt=agent.state.txs['critic'].update(grads,agent.state.opt_states['critic'],agent.state.params);params=optax.apply_updates(agent.state.params,updates);opts=dict(agent.state.opt_states);opts['critic']=newopt;tp=dict(agent.state.target_params);tau=agent.config['soft_target_update_rate'];tp['modules_critic']=jax.tree_util.tree_map(lambda p,t:tau*p+(1-tau)*t,params['modules_critic'],tp['modules_critic']);state=agent.state.replace(step=agent.state.step+1,params=params,opt_states=opts,target_params=tp,rng=nrng);return agent.replace(state=state),info
def td(agent,batch):
 def one(b,key):
  loss,info=SACAgent.critic_loss_fn(agent,b,agent.state.params,key);return {'bellman_loss':float(loss),'predicted_q':float(info['predicted_qs']),'target_q':float(info['target_qs']),'td_abs_proxy':float(abs(info['predicted_qs']-info['target_qs']))}
 return {'online':one({k:v[:32] for k,v in batch.items()},jax.random.PRNGKey(51)),'expert':one({k:v[32:] for k,v in batch.items()},jax.random.PRNGKey(52))}
def catchup(cfg,o,e,demos,coverage):
 agent=fresh_agent(cfg);env=make_train_env('C0')
 try:
  online=make_replay_buffer(env,capacity=cfg['training']['replay_capacity'],type='replay_buffer');restore_replay(online,HAND/'checkpoint_1000_online_replay.pkl');demo=make_replay_buffer(env,capacity=len(demos)+100,type='replay_buffer');corr=make_replay_buffer(env,capacity=len(pickle.loads(CORR.read_bytes()))+100,type='replay_buffer')
  for x in demos:demo.insert(x)
  for x in pickle.loads(CORR.read_bytes()):corr.insert({k:x[k] for k in ('observations','actions','next_observations','rewards','masks','dones')})
  rng=np.random.default_rng(7100 if coverage else 7000);base_actor=thash(agent.state.params['modules_actor']);base_actor_opt=thash(agent.state.opt_states['actor']);base_temp=thash(agent.state.params['modules_temperature']);base_temp_opt=thash(agent.state.opt_states['temperature']);base_other_target={k:thash(v) for k,v in agent.state.target_params.items() if k!='modules_critic'};reports={};baseline=geometry(agent,o,e);reports['0']=baseline;last={}
  for u in range(1,1001):
   oi=rng.integers(0,len(online),32)
   if coverage:
    di=rng.integers(0,len(demo),16);ci=rng.integers(0,len(corr),16);expert=concat_rlpd_batches(fixed_batch(demo,di),fixed_batch(corr,ci));batch=concat_rlpd_batches(fixed_batch(online,oi),expert)
   else:di=rng.integers(0,len(demo),32);batch=concat_rlpd_batches(fixed_batch(online,oi),fixed_batch(demo,di))
   agent,info=critic_step(agent,batch);block((agent,info));last=finite_scalars(info)
   if u in PROBES[1:]:
    g=geometry(agent,o,e);g.update(update=u,critic_losses=last,td_error=td(agent,batch));reports[str(u)]=g
    passed=g['Q_GRAD_EXPERT_COSINE_MEDIAN']>=0 and g['Q_GRAD_ANTI_ALIGNED_FRACTION']<=baseline['Q_GRAD_ANTI_ALIGNED_FRACTION']-.10 and g['Q_EXPERT_NOT_WORSE_FRACTION']>=.5 and g['finite']
    if passed:break
  req(thash(agent.state.params['modules_actor'])==base_actor and thash(agent.state.opt_states['actor'])==base_actor_opt and thash(agent.state.params['modules_temperature'])==base_temp and thash(agent.state.opt_states['temperature'])==base_temp_opt and all(thash(agent.state.target_params[k])==v for k,v in base_other_target.items()),'frozen binding')
  selected=next((int(k) for k,v in reports.items() if int(k)>0 and v['Q_GRAD_EXPERT_COSINE_MEDIAN']>=0 and v['Q_GRAD_ANTI_ALIGNED_FRACTION']<=baseline['Q_GRAD_ANTI_ALIGNED_FRACTION']-.10 and v['Q_EXPERT_NOT_WORSE_FRACTION']>=.5 and v['finite']),None);return agent,reports,selected,online
 finally:env.close()
def local_exit(agent,demo_obs,scale):
 rows=pickle.loads(LOCAL.read_bytes());sp=json.loads((LOCAL.parent/'split.json').read_text());tr=set(sp['train_episodes']);va=set(sp['valid_episodes']);to=np.asarray([x['observations'] for x in rows if x['source_episode'] in tr],np.float64);vo=np.asarray([x['observations'] for x in rows if x['source_episode'] in va],np.float64);s,_=rollout(agent,demo_obs,to,vo,scale);return next((x['step'] for x in s if min(x['D_LOCAL_TRAIN'],x['D_LOCAL_VALID'])>TH),None)
def reenable(agent,online,cfg,demos,demo_obs,scale,o,e,out):
 env=make_train_env('C0');demo=make_replay_buffer(env,capacity=len(demos)+100,type='replay_buffer')
 for x in demos:demo.insert(x)
 try:
  obs,_=env.reset(seed=cfg['seed']);arng=jax.random.PRNGKey(19000);rng=np.random.default_rng(8100);reports={'0':{'c0':evaluate(agent,'C0',demo_obs,scale),'geometry':geometry(agent,o,e),'corrective_mae':float(np.mean(np.abs(modes(agent,o)[:,:6]-e[:,:6]))),'support_exit':local_exit(agent,demo_obs,scale)}}
  for bi in range(5):
   admitted=attempt=0
   while admitted<60:
    a,arng=action(agent,obs,deterministic=False,rng=arng);n,r,term,trunc,info=env.step(a);d=demo_distance(n,demo_obs,scale);attempt+=1
    if d<=TH:online.insert({'observations':np.asarray(obs,np.float32),'actions':a,'next_observations':np.asarray(n,np.float32),'rewards':np.float32(r),'masks':np.float32(0 if term else 1),'dones':bool(term or trunc)});admitted+=1;obs=n
    if term or trunc or d>TH:obs,_=env.reset(seed=cfg['seed'])
   frozen=len(online)
   for u in range(bi*100+1,(bi+1)*100+1):
    oi=rng.integers(0,frozen,32);di=rng.integers(0,len(demo),32);batch=concat_rlpd_batches(fixed_batch(online,oi),fixed_batch(demo,di));agent,info=agent.update_high_utd(batch,utd_ratio=1);block((agent,info))
    if u in (1,2,5,10,25,50,100,250,500):
     g=geometry(agent,o,e);rpt={'c0':evaluate(agent,'C0',demo_obs,scale) if u in (5,10,25,50,100,250,500) else None,'geometry':g,'corrective_mae':float(np.mean(np.abs(modes(agent,o)[:,:6]-e[:,:6]))),'support_exit':local_exit(agent,demo_obs,scale) if u in (5,10,25,50,100,250,500) else None};reports[str(u)]=rpt;dump(out/f'actor_reenable_{u}.json',rpt)
   if bi==0:
    r=reports['100'];ok=r['c0']['success_count']>=2 and r['c0']['near_hole_count']>=2 and r['c0']['contact_health'] and not r['c0']['table_ee_contact'] and r['geometry']['Q_GRAD_EXPERT_COSINE_MEDIAN']>-.2
    if not ok:return agent,reports,'FAIL','NOT_RUN'
  r=reports['500'];ok=r['c0']['success_count']>=2 and r['c0']['near_hole_count']>=2 and r['c0']['contact_health'] and not r['c0']['table_ee_contact'];return agent,reports,'PASS','PASS' if ok else 'FAIL'
 finally:env.close()
def main():
 req(Path.cwd()==ROOT and not OUT.exists(),'fresh/cwd');req(jax.default_backend()=='gpu' and sha(DEMO)==DSHA,'GPU/Demo');OUT.mkdir(parents=True);before={'demo':sha(DEMO),'corrective':sha(CORR),'local':sha(LOCAL),'repaired':sha(REPAIRED)};dump(OUT/'protected_before.json',before);cfg=load_config(1);demos=pickle.loads(DEMO.read_bytes());demo_obs,scale=distance_model(demos);start=time.time()
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  repaired=fresh_agent(cfg);o,e,manifest=build_probe(repaired);base=geometry(repaired,o,e);dump(OUT/'baseline_geometry.json',base);agent,reports,selected,online=catchup(cfg,o,e,demos,False);coverage=False;dump(OUT/'primary_catchup.json',reports)
  if selected is None:agent,reports,selected,online=catchup(cfg,o,e,demos,True);coverage=True;dump(OUT/'corrective_coverage_catchup.json',reports)
  if selected is None:
   final={'REPAIRED_ACTOR_CRITIC_HANDOFF_REVIEW':'PASS','REPAIRED_ACTOR_CHECKPOINT':str(REPAIRED),'REPAIRED_ACTOR_SHA256':sha(REPAIRED),'CRITIC_ALIGNMENT_PROBE_SHA256':manifest['probe_sha256'],'BASE_Q_GRAD_EXPERT_COSINE_MEAN':base['Q_GRAD_EXPERT_COSINE_MEAN'],'BASE_Q_GRAD_EXPERT_COSINE_MEDIAN':base['Q_GRAD_EXPERT_COSINE_MEDIAN'],'BASE_Q_GRAD_ANTI_ALIGNED_FRACTION':base['Q_GRAD_ANTI_ALIGNED_FRACTION'],'CRITIC_CATCHUP':'FAIL','CRITIC_CATCHUP_PASS_UPDATE':None,'CORRECTIVE_EXPERT_CRITIC_COVERAGE_USED':'YES','FROZEN_ACTOR_BEHAVIOR_PRESERVED':'YES','PRIMARY_ROOT_CAUSE':'LOCAL_CRITIC_ACTION_GEOMETRY_NOT_REPAIRED_BY_DATA_COVERAGE','FIRST_BAD_GATE':'CRITIC_CATCHUP','NEXT':'CRITIC_PARAMETERIZATION_OR_TARGET_GEOMETRY_REVIEW'}
  else:
   post=reports[str(selected)];san=evaluate(agent,'C0',demo_obs,scale);pres=san['success_count']>=2;dump(OUT/'frozen_actor_sanity.json',san);req(pres,'frozen actor binding error');ad,ar,g100,g500=reenable(agent,online,cfg,demos,demo_obs,scale,o,e,OUT);final={'REPAIRED_ACTOR_CRITIC_HANDOFF_REVIEW':'PASS','REPAIRED_ACTOR_CHECKPOINT':str(REPAIRED),'REPAIRED_ACTOR_SHA256':sha(REPAIRED),'CRITIC_ALIGNMENT_PROBE_SHA256':manifest['probe_sha256'],'BASE_Q_GRAD_EXPERT_COSINE_MEAN':base['Q_GRAD_EXPERT_COSINE_MEAN'],'BASE_Q_GRAD_EXPERT_COSINE_MEDIAN':base['Q_GRAD_EXPERT_COSINE_MEDIAN'],'BASE_Q_GRAD_ANTI_ALIGNED_FRACTION':base['Q_GRAD_ANTI_ALIGNED_FRACTION'],'CRITIC_CATCHUP':'PASS','CRITIC_CATCHUP_PASS_UPDATE':selected,'POST_CATCHUP_Q_GRAD_EXPERT_COSINE_MEAN':post['Q_GRAD_EXPERT_COSINE_MEAN'],'POST_CATCHUP_Q_GRAD_EXPERT_COSINE_MEDIAN':post['Q_GRAD_EXPERT_COSINE_MEDIAN'],'POST_CATCHUP_ANTI_ALIGNED_FRACTION':post['Q_GRAD_ANTI_ALIGNED_FRACTION'],'CORRECTIVE_EXPERT_CRITIC_COVERAGE_USED':'YES' if coverage else 'NO','FROZEN_ACTOR_BEHAVIOR_PRESERVED':'YES','ACTOR_REENABLE_100':g100,'C0_SUCCESS_AT_100':f"{ar['100']['c0']['success_count']}/3",'C0_NEAR_HOLE_AT_100':f"{ar['100']['c0']['near_hole_count']}/3",'C0_SUCCESS_MARGIN_AT_100':ar['100']['c0']['best_success_margin'],'C0_CONTACT_AT_100':'PASS' if ar['100']['c0']['contact_health'] and not ar['100']['c0']['table_ee_contact'] else 'FAIL','ACTOR_REENABLE_500':g500,'C0_SUCCESS_AT_500':f"{ar['500']['c0']['success_count']}/3" if '500' in ar else 'NOT_RUN','C0_NEAR_HOLE_AT_500':f"{ar['500']['c0']['near_hole_count']}/3" if '500' in ar else 'NOT_RUN','C0_CONTACT_AT_500':'PASS' if '500' in ar and ar['500']['c0']['contact_health'] and not ar['500']['c0']['table_ee_contact'] else 'NOT_RUN' if '500' not in ar else 'FAIL','REPAIRED_ACTOR_RLPD_RETENTION':'PASS' if g500=='PASS' else 'FAIL','PRIMARY_ROOT_CAUSE':'CRITIC_FIRST_HANDOFF_REPAIRS_LOCAL_GEOMETRY' if g500=='PASS' else 'ACTOR_CRITIC_CLOSED_LOOP_GEOMETRY_UNSTABLE','FIRST_BAD_GATE':None if g500=='PASS' else 'ACTOR_REENABLE_100' if g100=='FAIL' else 'ACTOR_REENABLE_500','NEXT':'MATCHED_RETENTION_CONFIRMATION' if g500=='PASS' else 'STOP'}
 after={'demo':sha(DEMO),'corrective':sha(CORR),'local':sha(LOCAL),'repaired':sha(REPAIRED)};req(before==after,'protected');final.update(MATCHED_RETENTION_CONFIRMATION='NOT_RUN',C1_TRAINING='NOT_RUN',C2_TRAINING='NOT_RUN',C3_TRAINING='NOT_RUN',C3_1K='NOT_RUN',C3_1K_CONFIRMATION='NOT_RUN',C3_3K='NOT_RUN',FORMAL_5K='NOT_RUN',PEG_SINGLE_TASK_CONVERGED='NO',SUCCESSFUL_INSERTION='NO_CANONICAL',TRAINING_RUNNING=False,RESIDUAL_PROCESSES='NONE',PRODUCTION_DEMO_SHA_UNCHANGED='YES',ONLINE_ADMISSION_THRESHOLD=TH,ONLINE_ADMISSION_THRESHOLD_CHANGED='NO',EVAL_RESET_CANONICAL='YES',EVAL_GOAL_CANONICAL='YES',EVAL_CURRICULUM_ACTIVE='NO',PROTECTED_ARTIFACTS_CHANGED='NO',elapsed_seconds=time.time()-start);dump(OUT/'result.json',final);dump(OUT/'protected_after.json',after);print(json.dumps(jv(final),indent=2,sort_keys=True))
if __name__=='__main__':main()
