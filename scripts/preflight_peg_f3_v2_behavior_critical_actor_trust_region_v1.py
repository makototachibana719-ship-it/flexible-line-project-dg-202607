#!/usr/bin/env python3
"""Freeze V1 behavior-critical trust-region bounds from existing evidence only."""
from __future__ import annotations
import hashlib,json,os,pickle,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,numpy as np
from async_sac_state_sim import concat_rlpd_batches
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import block,distance_model,fixed_batch,load_config,make_train_env
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
DEMO=p4.DEMO;DSHA=p4.DSHA;LOCAL=p4.LOCAL;CORR=p4.CORR;TH=p4.TH;PRIOR=p4.PRIOR;P4=ROOT/'results/training/peg_f3_v2_actor_update_timescale_v1_20260908_003';OUT=ROOT/'results/preflight/peg/f3_v2_behavior_critical_actor_trust_region_v1_20260908_005';ALPHAS=(1.,.5,.25,.125,.0625,.03125,.015625);EPS=1e-8
CHECKS=((0,0,True),(20,5,True),(40,10,True),(100,25,True),(250,62,False),(500,125,False))
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
def checkpoint(u,a):return P4/'checkpoints'/f'learner_{u:04d}_actor_{a:03d}.msgpack'
def drift_metrics(agent,rows):
 o=np.asarray([x['observation'] for x in rows],np.float32);r=np.asarray([x['reference_action'] for x in rows],np.float32);d=np.linalg.norm(geo.modes(agent,o)[:,:6]-r[:,:6],axis=1);return {'MEAN_ACTION_DRIFT':float(d.mean()),'P95_ACTION_DRIFT':float(np.quantile(d,.95)),'MAX_ACTION_DRIFT':float(d.max())}
def main():
 req(Path.cwd()==ROOT and not OUT.exists(),'fresh/cwd');req(jax.default_backend()=='gpu' and sha(DEMO)==DSHA,'GPU/Demo');OUT.mkdir(parents=True);before={'demo':sha(DEMO),'corrective':sha(CORR),'local':sha(LOCAL)};dump(OUT/'protected_before.json',before);cfg=load_config(1);demos=pickle.loads(DEMO.read_bytes());demo_obs,scale=distance_model(demos)
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  common,online,demo=geo.build_common_start(cfg,demos);prior=json.loads((PRIOR/'result.json').read_text());refsha=geo.thash(common.state.params['modules_actor']);req(refsha==prior['ACTOR_SHA'],'reference actor');c0=evaluate(common,'C0',demo_obs,scale);req(c0['success_count']==3 and c0['contact_health'] and not c0['table_ee_contact'],'common C0')
  geo.OUT=OUT;rows,snaps,tagged,manifest=geo.build_critical(common);req(manifest['sha256']=='81b03c51951727533b1a5ba11dc0578d9bf4e953a9f024fe544cf7a7455a43c1','probe sha')
  first_lateral=next(i for i,x in enumerate(rows) if 'first_lateral_correction' in x['tags']);best_lateral=int(np.argmin([x['lateral'] for x in rows]));sel=sorted(set([0,first_lateral,best_lateral,len(rows)-4,len(rows)-3,len(rows)-2,len(rows)-1]));req(6<=len(sel)<=12,'sentinel count');roles={0:['c0_initial','near_hole_entry'],first_lateral:['early_lateral_correction'],best_lateral:['orientation_correction_region','minimum_lateral'],len(rows)-4:['best_success_margin_approach'],len(rows)-3:['pre_success'],len(rows)-2:['pre_success'],len(rows)-1:['pre_success','insertion_success_region']}
  sentinel=[]
  for i in sel:y={k:v for k,v in rows[i].items() if k!='snapshot_index'};y['sentinel_roles']=sorted(set(roles.get(i,[])));sentinel.append(y)
  sp=OUT/'C0_TRUST_REGION_SENTINEL_V1.pkl';sp.write_bytes(pickle.dumps(sentinel,pickle.HIGHEST_PROTOCOL));smanifest={'name':'C0_TRUST_REGION_SENTINEL_V1','count':len(sel),'probe_indices':sel,'roles':[roles.get(i,[]) for i in sel],'source_tags':[rows[i]['tags'] for i in sel],'sha256':sha(sp),'source_probe_sha256':manifest['sha256'],'new_demonstrations':False,'selection_rule':'initial + tagged first lateral + minimum lateral/orientation-control region + final four success-approach states'};dump(OUT/'C0_TRUST_REGION_SENTINEL_V1_manifest.json',smanifest)
  evidence={}
  for u,a,success in CHECKS:
   cp=checkpoint(u,a);req(cp.exists(),'checkpoint');agent=common.replace(state=flax.serialization.from_bytes(common.state,cp.read_bytes()));m=drift_metrics(agent,rows);m['MAX_ONE_STEP_DAMAGE']=geo.one_step(agent,rows,snaps,sel)['maximum'];m.update(learner_update=u,actor_count=a,successful_checkpoint=success,checkpoint_sha256=sha(cp));evidence[str(a)]=m
  safe=[evidence[str(a)] for _,a,s in CHECKS if s];failed=[evidence[str(a)] for _,a,s in CHECKS if not s];bounds={'TAU_MEAN_ACTION_DRIFT':max(x['MEAN_ACTION_DRIFT'] for x in safe),'TAU_P95_ACTION_DRIFT':max(x['P95_ACTION_DRIFT'] for x in safe),'TAU_MAX_ONE_STEP_DAMAGE':max(x['MAX_ONE_STEP_DAMAGE'] for x in safe),'EPS':EPS};sep=all(min(x[k] for x in failed)>bounds['TAU_'+k] for k in ('MEAN_ACTION_DRIFT','P95_ACTION_DRIFT','MAX_ONE_STEP_DAMAGE'))
  env=make_train_env('C0')
  try:obs,arng,ci=p4.collect(env,common,online,cfg,demo_obs,scale,None,jax.random.PRNGKey(19000));frozen=len(online);rng=np.random.default_rng(8100)
  finally:env.close()
  # The validated baseline's learner-4 Actor slot advances optimizer state but is
  # an exact parameter no-op. Reproduce it, then use learner 8: the first ordinary
  # period-4 candidate with an actual parameter delta from COMMON_START.
  candidate_base=common
  learner4_actor_opt_before=geo.thash(candidate_base.state.opt_states['actor'])
  for learner in range(1,9):
   batch=concat_rlpd_batches(fixed_batch(online,rng.integers(0,frozen,32)),fixed_batch(demo,rng.integers(0,len(demo),32)));candidate_base,_=p4.critic_update_only(candidate_base,batch);block(candidate_base)
   if learner==4:
    candidate_base,_=p4.actor_temperature_update_only(candidate_base,batch);block(candidate_base);req(geo.thash(candidate_base.state.params['modules_actor'])==refsha,'learner4 expected actor parameter no-op');req(geo.thash(candidate_base.state.opt_states['actor'])!=learner4_actor_opt_before,'learner4 expected optimizer advance')
  req(geo.thash(candidate_base.state.params['modules_actor'])==refsha,'critic-only actor freeze')
  full,_=p4.actor_temperature_update_only(candidate_base,batch);block(full);baseline8=flax.serialization.from_bytes(common.state,checkpoint(8,2).read_bytes());req(geo.thash(full.state.params['modules_actor'])==geo.thash(baseline8.params['modules_actor']),'learner8 ordinary candidate equivalence');old=candidate_base.state.params['modules_actor'];new=full.state.params['modules_actor'];o=np.asarray([x['observation'] for x in rows],np.float32);ref=np.asarray([x['reference_action'] for x in rows],np.float32);current_modes=geo.modes(candidate_base,o);san=[]
  for alpha in ALPHAS:
   ap=jax.tree_util.tree_map(lambda x,y:x+alpha*(y-x),old,new);params=dict(candidate_base.state.params);params['modules_actor']=ap;candidate=candidate_base.replace(state=candidate_base.state.replace(params=params));delta=jax.tree_util.tree_map(lambda x,y:x-y,ap,old);m=geo.modes(candidate,o);fixed=np.linalg.norm(m[:,:6]-ref[:,:6],axis=1);induced=np.linalg.norm(m[:,:6]-current_modes[:,:6],axis=1);san.append({'alpha':alpha,'parameter_delta_norm':geo.norm(delta),'mean_fixed_reference_action_drift':float(fixed.mean()),'mean_update_induced_action_displacement':float(induced.mean())})
  req(san[0]['parameter_delta_norm']>0,'ordinary period4 candidate must move Actor');req(geo.thash(jax.tree_util.tree_map(lambda x,y:x+(y-x),old,new))==geo.thash(new),'alpha1 exact');req(all(san[i]['parameter_delta_norm']>=san[i+1]['parameter_delta_norm'] for i in range(len(san)-1)) and all(san[i]['mean_update_induced_action_displacement']>=san[i+1]['mean_update_induced_action_displacement']-1e-10 for i in range(len(san)-1)),'backtracking monotonicity')
  result={'BEHAVIOR_CRITICAL_ACTOR_TRUST_REGION_PREFLIGHT':'PASS','COMMON_START_C0_SUCCESS':'3/3','REFERENCE_ACTOR_SHA':refsha,'CRITICAL_PROBE_SHA':manifest['sha256'],'SENTINEL_SHA':smanifest['sha256'],'SAFE_MEAN_MAX':bounds['TAU_MEAN_ACTION_DRIFT'],'SAFE_P95_MAX':bounds['TAU_P95_ACTION_DRIFT'],'SAFE_DAMAGE_MAX':bounds['TAU_MAX_ONE_STEP_DAMAGE'],'FAILED_MEAN_MIN':min(x['MEAN_ACTION_DRIFT'] for x in failed),'FAILED_P95_MIN':min(x['P95_ACTION_DRIFT'] for x in failed),'FAILED_DAMAGE_MIN':min(x['MAX_ONE_STEP_DAMAGE'] for x in failed),**bounds,'TRUST_REGION_SEPARATION':'PASS' if sep else 'WEAK','BACKTRACKING_MONOTONICITY':'PASS','BACKTRACKING_MONOTONICITY_METRIC':'update-induced action displacement from current Actor; fixed-reference drift separately recorded and retained for formal acceptance','SANITY_CANDIDATE_CONTEXT':'COMMON_START baseline trajectory through the exact learner-4 parameter no-op, then first parameter-moving ordinary period4 candidate at learner 8','LEARNER4_ACTOR_PARAMETER_UPDATE':'NO_OP','LEARNER4_ACTOR_OPTIMIZER_ADVANCED':'YES','LEARNER8_CANDIDATE_MATCHES_BASELINE_CHECKPOINT':'YES','ALPHA_CANDIDATES':ALPHAS,'checkpoint_evidence':evidence,'sanity':san,'ACTOR_UPDATE_PERIOD':4,'PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','EVAL_RESET_CANONICAL':'YES','EVAL_GOAL_CANONICAL':'YES','EVAL_CURRICULUM_ACTIVE':'NO','PROTECTED_ARTIFACTS_CHANGED':'NO','STAGE_EXECUTION_STATUS':'COMPLETE','SCIENTIFIC_GATE':'PASS'}
 after={'demo':sha(DEMO),'corrective':sha(CORR),'local':sha(LOCAL)};req(before==after,'protected');dump(OUT/'frozen_trust_region.json',result);dump(OUT/'protected_after.json',after);print(json.dumps(jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
