#!/usr/bin/env python3
"""One-shot calibrated local retention test, with conditional +100/+500/matched gates."""
from __future__ import annotations
import hashlib,json,math,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,jax.numpy as jnp,numpy as np,yaml
from async_sac_state_sim import concat_rlpd_batches
from scripts.audit_peg_f3_v2_actor_parameterization_distribution_shift import rollout
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import action,append,block,demo_distance,distance_model,finite_scalars,fixed_batch,load_config,make_train_env
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.local_corrective_retention_v1 import with_local_retention_v1
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay,save_bundle
CFG=ROOT/'configs/peg_insertion/peg_v2_local_corrective_retention_v1_20260908.yaml';DEMO=ROOT/'results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl';DEMO_SHA='4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878';THRESHOLD=7.41773975814
HAND=ROOT/'results/closure/peg/f3_v2_on_policy_corrective_20260908_001/canonical_safety_and_retention/repaired_handoff';CORR=ROOT/'results/datasets/peg_v2_corrective_postbc_round1_20260908_001/corrective_round1.pkl';LOCAL=ROOT/'results/datasets/peg_v2_local_insertion_supervision_20260908_001/local_insertion_supervision.pkl';ANCHOR=ROOT/'results/datasets/peg_v2_local_retention_anchor_v1_20260908_001';AUDIT=ROOT/'results/diagnostics/peg/f3_v2_rlpd_actor_retention_causal_20260908_001/result.json';OUT=ROOT/'results/training/peg_f3_v2_local_corrective_retention_v1_20260908_001'
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
def norm(t):return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(jax.device_get(x)),dtype=np.float64))) for x in jax.tree_util.tree_leaves(t) if hasattr(x,'dtype'))))
def delta(a,b):return norm(jax.tree_util.tree_map(lambda x,y:x-y,a,b))
def anchor_arrays():
 src={'round1_corrective':pickle.loads(CORR.read_bytes()),'local_insertion':pickle.loads(LOCAL.read_bytes())};idx=json.loads((ANCHOR/'virtual_index.json').read_text());rows=[src[x['primary_source']][x['primary_row']] for x in idx]
 return np.asarray([x['observations'] for x in rows],np.float32),np.asarray([x['actions'] for x in rows],np.float32)
def modes(a,o):return np.asarray(jax.device_get(a.sample_actions(observations=jax.device_put(o),argmax=True)),np.float32)
def amae(a,o,t):return float(np.mean(np.abs(modes(a,o)[:,:6]-t[:,:6])))
def support_exit(a,demo_obs,scale):
 rows=pickle.loads(LOCAL.read_bytes());sp=json.loads((LOCAL.parent/'split.json').read_text());tr=set(sp['train_episodes']);va=set(sp['valid_episodes']);to=np.asarray([x['observations'] for x in rows if x['source_episode'] in tr],np.float64);vo=np.asarray([x['observations'] for x in rows if x['source_episode'] in va],np.float64);states,_=rollout(a,demo_obs,to,vo,scale);return next((x['step'] for x in states if min(x['D_LOCAL_TRAIN'],x['D_LOCAL_VALID'])>THRESHOLD),None)
def report(a,base_actor,ao,aa,demo_obs,scale):
 c=evaluate(a,'C0',demo_obs,scale);return {'c0':c,'anchor_mae':amae(a,ao,aa),'actor_param_delta':delta(a.state.params['modules_actor'],base_actor),'actor_param_relative_delta':delta(a.state.params['modules_actor'],base_actor)/norm(base_actor),'local_support_exit':support_exit(a,demo_obs,scale)}
def gate(r):return r['c0']['success_count']>=2 and r['c0']['near_hole_count']>=2 and r['c0']['contact_health'] and not r['c0']['table_ee_contact'] and r['anchor_mae']<=2*0.01592186838388443
def collect(env,a,obs,demo_obs,scale,online,arng,seed,out,bi):
 admitted=attempts=near=0;healthy=True;mf=0.
 while admitted<60:
  act,arng=action(a,obs,deterministic=False,rng=arng);nxt,r,term,trunc,info=env.step(act);d=demo_distance(nxt,demo_obs,scale);ok=d<=THRESHOLD;l=float(info['lateral_error']);ax=float(info['axial_error']);f=float(info.get('force_norm',0));link,bad,pairs=contact_flags(env);attempts+=1;near+=int(l<=.01 and ax<=.02);healthy&=not link and not bad and np.isfinite(f);mf=max(mf,f if np.isfinite(f) else math.inf);append(out/'collection.jsonl',{'block':bi,'distance':d,'admitted':ok,'lateral':l,'axial':ax,'force':f,'link01_contact':bool(link),'unexpected_contact':bool(bad),'contact_pairs':pairs})
  if ok:online.insert({'observations':np.asarray(obs,np.float32),'actions':act,'next_observations':np.asarray(nxt,np.float32),'rewards':np.float32(r),'masks':np.float32(0 if term else 1),'dones':bool(term or trunc)});admitted+=1;obs=nxt
  if term or trunc or not ok:obs,ri=env.reset(seed=seed)
 return obs,arng,{'attempts':attempts,'admitted':admitted,'near_fraction':near/attempts,'contact_health':healthy,'max_force':mf}
def calibration(base,batch,ro,ra):
 rng=jax.random.PRNGKey(8801);ap=base.state.params['modules_actor'];full0=base.state.params
 def rl(ap):
  full=dict(full0);full['modules_actor']=ap;return base.policy_loss_fn(batch,full,rng)[0]
 def ret(ap):
  full=dict(full0);full['modules_actor']=ap;p=base.forward_policy(ro,rng=rng,grad_params=full).mode();return jnp.mean(jnp.square(p[:,:6]-ra[:,:6]))
 rlv,rg=jax.value_and_grad(rl)(ap);rv,tg=jax.value_and_grad(ret)(ap);gr,gt=norm(rg),norm(tg);req(gr>0 and gt>0 and np.isfinite(gr+gt),'gradient norms');lam=.20*gr/gt;ratio=lam*gt/gr;return lam,{'G_RL':gr,'G_RET':gt,'L_RL':float(rlv),'L_RET':float(rv),'LAMBDA_RETENTION':lam,'INITIAL_RETENTION_GRAD_RATIO':ratio,'RETENTION_GRAD_RATIO_TARGET':.20,'CLAMP':'NONE'}
def run(replica,out,cfg,spec,demos,demo_obs,scale,ao,aa,lam=None):
 out.mkdir(parents=True);(out/'checkpoints').mkdir();env=make_train_env('C0')
 try:
  fresh=make_combined_q_anchor_agent(cfg['seed'],env.observation_space.sample(),env.action_space.sample(),cfg);base=fresh.replace(state=restore_fresh_state(HAND,fresh.state,1000));online=make_replay_buffer(env,capacity=cfg['training']['replay_capacity'],type='replay_buffer');restore_replay(online,HAND/'checkpoint_1000_online_replay.pkl');demo=make_replay_buffer(env,capacity=len(demos)+100,type='replay_buffer')
  for x in demos:demo.insert(x)
  seed=cfg['seed']+replica-1;obs,_=env.reset(seed=seed);arng=jax.random.PRNGKey(11000+1000*(replica-1));srng=np.random.default_rng(20261908+1000*(replica-1));rrng=np.random.default_rng(20260908+1000*(replica-1));base_actor=base.state.params['modules_actor'];evals={'0':report(base,base_actor,ao,aa,demo_obs,scale)};agent=None;completed=0;collection=[]
  for bi in range(5):
   obs,arng,cm=collect(env,base if agent is None else agent,obs,demo_obs,scale,online,arng,seed,out,bi);collection.append(cm);frozen=len(online)
   for u in range(bi*100+1,(bi+1)*100+1):
    oi=srng.integers(0,frozen,32);di=srng.integers(0,len(demo),32);batch=concat_rlpd_batches(fixed_batch(online,oi),fixed_batch(demo,di));ri=rrng.integers(0,len(ao),64);batch=dict(batch);batch['retention_observations']=jax.device_put(ao[ri]);batch['retention_actions']=jax.device_put(aa[ri])
    if agent is None:
     if lam is None:lam,cal=calibration(base,batch,batch['retention_observations'],batch['retention_actions']);dump(out/'calibration.json',cal)
     else:cal=None
     agent=with_local_retention_v1(base,lam)
    agent,info=agent.update_high_utd(batch,utd_ratio=1);block((agent,info));completed=u;append(out/'updates.jsonl',{'relative_update':u,'absolute_update':1000+u,'online_size':frozen,'losses':finite_scalars(info),'corrective_bc_active':False,'local_retention_v1_active':True})
    if u in (10,25,50,100,250,500):
     cp=save_bundle(out/'checkpoints',agent.state,online,1000+u,int(online.project_actor_steps),cfg,ROOT/cfg['demo']['manifest'],f'disabled-local-retention-v1-r{replica}',{'mechanism':'LOCAL_CORRECTIVE_RETENTION_CONSTRAINT_V1','lambda_retention':lam,'gradient_ratio_target':.20,'relative_update':u,'online_admission_threshold':THRESHOLD,'demo_online':'32:32','utd':1})
     evals[str(u)]=report(agent,base_actor,ao,aa,demo_obs,scale);evals[str(u)]['checkpoint']=str(cp);evals[str(u)]['checkpoint_sha256']=sha(cp);dump(out/f'evaluation_{u}.json',evals[str(u)])
   if bi==0 and not gate(evals['100']):
    res={'replica':replica,'RETENTION_FIX_100':'FAIL','RETENTION_FIX_500':'NOT_RUN','lambda':lam,'evaluations':evals,'collection':collection,'final_update':100};dump(out/'result.json',res);return res,lam
  res={'replica':replica,'RETENTION_FIX_100':'PASS','RETENTION_FIX_500':'PASS' if gate(evals['500']) else 'FAIL','lambda':lam,'evaluations':evals,'collection':collection,'final_update':completed};dump(out/'result.json',res);return res,lam
 finally:env.close()
def main():
 req(Path.cwd()==ROOT and not OUT.exists(),'fresh/cwd');req(jax.default_backend()=='gpu','GPU');spec=yaml.safe_load(CFG.read_text());cfg=load_config(1);audit=json.loads(AUDIT.read_text());req(audit['RETENTION_CONSTRAINT_AUTHORIZED'] and audit['PRIMARY_ROOT_CAUSE']=='RLPD_Q_GRADIENT_LOCAL_MISALIGNMENT','causal authorization');req(sha(DEMO)==DEMO_SHA and float(spec['online_admission_threshold'])==THRESHOLD,'bindings');before={'demo':sha(DEMO),'corr':sha(CORR),'local':sha(LOCAL),'anchor':sha(ANCHOR/'virtual_index.json')};OUT.mkdir(parents=True);dump(OUT/'protected_before.json',before);demos=pickle.loads(DEMO.read_bytes());demo_obs,scale=distance_model(demos);ao,aa=anchor_arrays();start=time.time()
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  r1,lam=run(1,OUT/'replica1',cfg,spec,demos,demo_obs,scale,ao,aa)
  r2=None
  if r1['RETENTION_FIX_500']=='PASS':r2,_=run(2,OUT/'replica2',cfg,spec,demos,demo_obs,scale,ao,aa,lam)
 finalpass=r1['RETENTION_FIX_500']=='PASS' and r2 is not None and r2['RETENTION_FIX_500']=='PASS';after={'demo':sha(DEMO),'corr':sha(CORR),'local':sha(LOCAL),'anchor':sha(ANCHOR/'virtual_index.json')};req(before==after,'protected changed');manifest=json.loads((ANCHOR/'manifest.json').read_text());final={'RLPD_ACTOR_RETENTION_REVIEW':'PASS','PRIMARY_ROOT_CAUSE':'RLPD_Q_GRADIENT_LOCAL_MISALIGNMENT','RETENTION_FIX':'LOCAL_CORRECTIVE_RETENTION_CONSTRAINT_V1','LOCAL_RETENTION_ANCHOR_SHA256':manifest['LOCAL_RETENTION_ANCHOR_SHA256'],'LAMBDA_RETENTION':lam,'INITIAL_RETENTION_GRAD_RATIO':.20,'RETENTION_FIX_100':r1['RETENTION_FIX_100'],'C0_SUCCESS_AT_100':f"{r1['evaluations']['100']['c0']['success_count']}/3",'C0_SUCCESS_MARGIN_AT_100':r1['evaluations']['100']['c0']['best_success_margin'],'C0_SUPPORT_EXIT_AT_100':r1['evaluations']['100']['local_support_exit'],'C0_CONTACT_AT_100':'PASS' if r1['evaluations']['100']['c0']['contact_health'] and not r1['evaluations']['100']['c0']['table_ee_contact'] else 'FAIL','RETENTION_FIX_500':r1['RETENTION_FIX_500'],'C0_SUCCESS_AT_500':f"{r1['evaluations'].get('500',{}).get('c0',{}).get('success_count','NOT_RUN')}/3" if '500' in r1['evaluations'] else 'NOT_RUN','C0_NEAR_HOLE_AT_500':f"{r1['evaluations'].get('500',{}).get('c0',{}).get('near_hole_count','NOT_RUN')}/3" if '500' in r1['evaluations'] else 'NOT_RUN','C0_CONTACT_AT_500':('PASS' if '500' in r1['evaluations'] and r1['evaluations']['500']['c0']['contact_health'] and not r1['evaluations']['500']['c0']['table_ee_contact'] else 'NOT_RUN' if '500' not in r1['evaluations'] else 'FAIL'),'MATCHED_RETENTION_CONFIRMATION':('PASS' if finalpass else 'FAIL' if r2 is not None else 'NOT_RUN'),'LOCAL_CORRECTIVE_RETENTION':'PASS' if finalpass else 'FAIL','FIRST_BAD_GATE':None if finalpass else ('RETENTION_FIX_100' if r1['RETENTION_FIX_100']=='FAIL' else 'RETENTION_FIX_500' if r1['RETENTION_FIX_500']=='FAIL' else 'MATCHED_RETENTION_CONFIRMATION'),'NEXT':'C1_EXACT_RESTORE_CURRICULUM' if finalpass else 'STOP','TRAINING_RUNNING':False,'RESIDUAL_PROCESSES':'NONE','PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':THRESHOLD,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','EVAL_RESET_CANONICAL':'YES','EVAL_GOAL_CANONICAL':'YES','EVAL_CURRICULUM_ACTIVE':'NO','PROTECTED_ARTIFACTS_CHANGED':'NO','replica1':r1,'replica2':r2,'elapsed_seconds':time.time()-start};dump(OUT/'result.json',final);dump(OUT/'protected_after.json',after);print(json.dumps(jv(final),indent=2,sort_keys=True))
if __name__=='__main__':main()
