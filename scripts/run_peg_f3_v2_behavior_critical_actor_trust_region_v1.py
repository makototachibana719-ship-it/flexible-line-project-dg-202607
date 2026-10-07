#!/usr/bin/env python3
"""Behavior-critical Actor trust region V1, period-4 R1 retention run."""
from __future__ import annotations
import hashlib,json,os,pickle,sys,time
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

DEMO=p4.DEMO;DSHA=p4.DSHA;LOCAL=p4.LOCAL;CORR=p4.CORR;HAND=p4.HAND;PRIOR=p4.PRIOR;TH=p4.TH
FROZEN_DIR=ROOT/'results/preflight/peg/f3_v2_behavior_critical_actor_trust_region_v1_20260908_005'
FROZEN=FROZEN_DIR/'frozen_trust_region.json';SENTINEL=FROZEN_DIR/'C0_TRUST_REGION_SENTINEL_V1.pkl';SMANIFEST=FROZEN_DIR/'C0_TRUST_REGION_SENTINEL_V1_manifest.json'
OUT=ROOT/'results/training/peg_f3_v2_behavior_critical_actor_trust_region_v1_r1_20260908_001'
ALPHAS=(1.,.5,.25,.125,.0625,.03125,.015625);EPS=1e-8;PERIOD=4;PRIMARY=100;RETENTION=500;MONITORS=(0,20,40,60,80,100,250,500)

def req(x,m):
 if not x:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p4.dump(p,x)
def save_state(agent,u,count):
 p=OUT/'checkpoints'/f'learner_{u:04d}_candidate_{count:03d}.msgpack'
 with p.open('xb') as f:f.write(flax.serialization.to_bytes(agent.state))
 restored=flax.serialization.from_bytes(agent.state,p.read_bytes());req(geo.thash(restored.params)==geo.thash(agent.state.params),'checkpoint restore')
 return {'path':str(p),'sha256':sha(p),'state_step':int(agent.state.step)}
def action_metrics(agent,obs,ref):
 d=np.linalg.norm(geo.modes(agent,obs)[:,:6]-ref[:,:6],axis=1)
 return {'MEAN_ACTION_DRIFT':float(d.mean()),'P95_ACTION_DRIFT':float(np.quantile(d,.95)),'MAX_ACTION_DRIFT':float(d.max())}
def actor_candidate(agent,batch,obs,ref,rows,snaps,sel,bounds,learner):
 pre=agent;full,info=p4.actor_temperature_update_only(pre,batch);block((full,info));old=pre.state.params['modules_actor'];new=full.state.params['modules_actor'];delta=jax.tree_util.tree_map(lambda x,y:x-y,new,old);full_norm=geo.norm(delta);trials=[];accepted=None
 for alpha in ALPHAS:
  ap=jax.tree_util.tree_map(lambda x,y:x+alpha*(y-x),old,new);params=dict(full.state.params);params['modules_actor']=ap;cand=full.replace(state=full.state.replace(params=params));m=action_metrics(cand,obs,ref);within_action=m['MEAN_ACTION_DRIFT']<=bounds['TAU_MEAN_ACTION_DRIFT']+EPS and m['P95_ACTION_DRIFT']<=bounds['TAU_P95_ACTION_DRIFT']+EPS;damage=None
  if within_action:damage=float(geo.one_step(cand,rows,snaps,sel)['maximum'])
  ok=within_action and damage<=bounds['TAU_MAX_ONE_STEP_DAMAGE']+EPS;trial={'alpha':alpha,**m,'MAX_ONE_STEP_DAMAGE':damage,'within_frozen_bounds':bool(ok)};trials.append(trial)
  if ok:accepted=(alpha,cand,trial);break
 fullm=action_metrics(full,obs,ref)
 if accepted is None:return pre,{'learner_update':learner,'full_candidate_parameter_delta_norm':full_norm,'full_candidate_action_drift':fullm,'accepted':False,'accepted_alpha':None,'trials':trials,'temperature_updated':False,'post_accept':action_metrics(pre,obs,ref)}
 alpha,cand,trial=accepted
 return cand,{'learner_update':learner,'full_candidate_parameter_delta_norm':full_norm,'full_candidate_action_drift':fullm,'accepted':True,'accepted_alpha':alpha,'trials':trials,'temperature_updated':True,'post_accept':trial}
def gate(report):
 c=report['c0'];return c['success_count']>=2 and c['near_hole_count']==3 and c['contact_health'] and not c['table_ee_contact']
def summarize(logs,limit):
 q=[x for x in logs if x['learner_update']<=limit];a=[x for x in q if x['accepted']];nz=[x for x in a if x['full_candidate_parameter_delta_norm']>0]
 return {'ACTOR_CANDIDATE_COUNT':len(q),'ACTOR_ACCEPT_COUNT':len(a),'ACTOR_REJECT_COUNT':len(q)-len(a),'NONZERO_CANDIDATE_ACCEPT_COUNT':len(nz),'MEAN_ACCEPTED_ALPHA':float(np.mean([x['accepted_alpha'] for x in a])) if a else None,'MIN_ACCEPTED_ALPHA':float(min(x['accepted_alpha'] for x in a)) if a else None,'EFFECTIVE_FULL_STEP_SUM':float(sum(x['accepted_alpha'] for x in a))}
def main():
 req(Path.cwd()==ROOT and not OUT.exists(),'fresh/cwd');req(jax.default_backend()=='gpu' and sha(DEMO)==DSHA,'GPU/Demo');frozen=json.loads(FROZEN.read_text());sm=json.loads(SMANIFEST.read_text());req(frozen['SCIENTIFIC_GATE']=='PASS' and tuple(frozen['ALPHA_CANDIDATES'])==ALPHAS and frozen['ACTOR_UPDATE_PERIOD']==PERIOD,'frozen preflight');req(sha(SENTINEL)==frozen['SENTINEL_SHA']==sm['sha256'] and frozen['CRITICAL_PROBE_SHA']==sm['source_probe_sha256'],'frozen artifacts')
 OUT.mkdir(parents=True);(OUT/'checkpoints').mkdir();start=time.time();p4.OUT=OUT;before={'demo':sha(DEMO),'corrective':sha(CORR),'local':sha(LOCAL)};dump(OUT/'protected_before.json',before);dump(OUT/'static_preflight.json',{'R1':'YES','FROZEN_TRUST_REGION_PATH':str(FROZEN),'FROZEN_TRUST_REGION_SHA256':sha(FROZEN),'REFERENCE_ACTOR_SHA':frozen['REFERENCE_ACTOR_SHA'],'CRITICAL_PROBE_SHA':frozen['CRITICAL_PROBE_SHA'],'SENTINEL_SHA':frozen['SENTINEL_SHA'],'TAU_MEAN_ACTION_DRIFT':frozen['TAU_MEAN_ACTION_DRIFT'],'TAU_P95_ACTION_DRIFT':frozen['TAU_P95_ACTION_DRIFT'],'TAU_MAX_ONE_STEP_DAMAGE':frozen['TAU_MAX_ONE_STEP_DAMAGE'],'EPS':EPS,'ALPHA_CANDIDATES':ALPHAS,'ACTOR_UPDATE_PERIOD':PERIOD,'CRITIC_OBJECTIVE_UNCHANGED':'YES','ACTOR_OBJECTIVE_UNCHANGED':'YES','TEMPERATURE_ONLY_ON_ACCEPT':'YES','ACTOR_OPTIMIZER_SEMANTICS':'accepted candidate keeps one ordinary optimizer-state advance; rejected candidate keeps pre-slot optimizer state','ONLINE_ADMISSION_THRESHOLD':TH})
 cfg=load_config(1);demos=pickle.loads(DEMO.read_bytes());demo_obs,scale=distance_model(demos);reports={};checkpoints={};collections=[];candidate_logs=[]
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  agent,online,demo=geo.build_common_start(cfg,demos);prior=json.loads((PRIOR/'result.json').read_text());hashes={'COMMON_START_ACTOR_SHA':geo.thash(agent.state.params['modules_actor']),'COMMON_START_CRITIC_SHA':geo.thash(agent.state.params['modules_critic']),'COMMON_START_TARGET_CRITIC_SHA':geo.thash(agent.state.target_params['modules_critic']),'COMMON_START_REPLAY_SHA':sha(HAND/'checkpoint_1000_online_replay.pkl')};req(hashes['COMMON_START_ACTOR_SHA']==frozen['REFERENCE_ACTOR_SHA']==prior['ACTOR_SHA'] and hashes['COMMON_START_CRITIC_SHA']==prior['CRITIC_SHA'] and hashes['COMMON_START_TARGET_CRITIC_SHA']==prior['TARGET_CRITIC_SHA'] and hashes['COMMON_START_REPLAY_SHA']==prior['REPLAY_SHA'],'COMMON_START hashes')
  initial=evaluate(agent,'C0',demo_obs,scale);req(initial['success_count']==3 and initial['contact_health'] and not initial['table_ee_contact'],'COMMON_START behavior');geo.OUT=OUT;rows,snaps,_,manifest=geo.build_critical(agent);req(manifest['sha256']==frozen['CRITICAL_PROBE_SHA'],'probe sha');sel=sm['probe_indices'];sent=pickle.loads(SENTINEL.read_bytes());req(len(sel)==len(sent) and all(np.array_equal(np.asarray(rows[i]['observation']),np.asarray(s['observation'])) for i,s in zip(sel,sent)),'sentinel identity')
  obs_probe=np.asarray([x['observation'] for x in rows],np.float32);ref=np.asarray([x['reference_action'] for x in rows],np.float32);bounds={k:frozen[k] for k in ('TAU_MEAN_ACTION_DRIFT','TAU_P95_ACTION_DRIFT','TAU_MAX_ONE_STEP_DAMAGE')};reports['0']=p4.probe(agent,0,0,rows,snaps,sel,demo_obs,scale);checkpoints['0']=save_state(agent,0,0)
  arng=jax.random.PRNGKey(19000);rng=np.random.default_rng(8100);obs=None;completed=0;stop=None;env=make_train_env('C0')
  for u in range(1,RETENTION+1):
   if (u-1)%100==0:obs,arng,ci=p4.collect(env,agent,online,cfg,demo_obs,scale,obs,arng);ci.update(block=(u-1)//100+1,replay_size=len(online));collections.append(ci);frozen_online=len(online)
   batch=concat_rlpd_batches(fixed_batch(online,rng.integers(0,frozen_online,32)),fixed_batch(demo,rng.integers(0,len(demo),32)));pre_actor=geo.thash(agent.state.params['modules_actor']);pre_opt=geo.thash(agent.state.opt_states['actor']);pre_temp=geo.thash(agent.state.params['modules_temperature']);agent,ci=p4.critic_update_only(agent,batch);block((agent,ci))
   if u%PERIOD==0:agent,cl=actor_candidate(agent,batch,obs_probe,ref,rows,snaps,sel,bounds,u);candidate_logs.append(cl)
   else:req(geo.thash(agent.state.params['modules_actor'])==pre_actor and geo.thash(agent.state.opt_states['actor'])==pre_opt and geo.thash(agent.state.params['modules_temperature'])==pre_temp,'period4 freeze')
   completed=u
   if u in MONITORS:reports[str(u)]=p4.probe(agent,u,len(candidate_logs),rows,snaps,sel,demo_obs,scale);checkpoints[str(u)]=save_state(agent,u,len(candidate_logs))
   if u==100 and not gate(reports['100']):stop='TRUST_REGION_100_GATE';break
   if u==500 and (not gate(reports['500']) or summarize(candidate_logs,500)['ACTOR_ACCEPT_COUNT']==0):stop='TRUST_REGION_500_GATE'
  env.close();dump(OUT/'candidate_updates.json',candidate_logs);dump(OUT/'monitoring.json',reports);dump(OUT/'checkpoints.json',checkpoints);dump(OUT/'collections.json',collections)
  s100=summarize(candidate_logs,100);s500=summarize(candidate_logs,500);pass100=completed>=100 and gate(reports['100']);pass500=completed>=500 and gate(reports['500']) and s500['ACTOR_ACCEPT_COUNT']>0;accepted_inside=all(x['post_accept']['MEAN_ACTION_DRIFT']<=bounds['TAU_MEAN_ACTION_DRIFT']+EPS and x['post_accept']['P95_ACTION_DRIFT']<=bounds['TAU_P95_ACTION_DRIFT']+EPS and x['post_accept']['MAX_ONE_STEP_DAMAGE']<=bounds['TAU_MAX_ONE_STEP_DAMAGE']+EPS for x in candidate_logs if x['accepted'])
  if not pass500 and completed>=500:rootcause='STANDARD_SAC_ACTOR_OBJECTIVE_HAS_NO_USEFUL_STEP_INSIDE_LOCAL_SAFE_BASIN' if s500['ACTOR_ACCEPT_COUNT']==0 else 'BEHAVIOR_CRITICAL_PROBE_NOT_SUFFICIENT_TO_DEFINE_SAFE_POLICY_BASIN';nexttask='LOCAL_POLICY_OR_RESIDUAL_POLICY_ARCHITECTURE_REVIEW' if s500['ACTOR_ACCEPT_COUNT']==0 else 'LOCAL_POLICY_PARAMETERIZATION_OR_HIERARCHICAL_CONTROL_REVIEW'
  else:rootcause='NONE' if pass500 else 'TRUST_REGION_100_FAILED';nexttask='MATCHED_R2' if pass500 else 'STOP'
  def c0(u):return f"{reports[str(u)]['c0']['success_count']}/3" if str(u) in reports else 'NOT_RUN'
  result={'BEHAVIOR_CRITICAL_ACTOR_TRUST_REGION_REVIEW':'PASS' if pass500 else 'FAIL','RUN':'R1',**hashes,'COMMON_START_C0_SUCCESS':'3/3','REFERENCE_ACTOR_SHA':frozen['REFERENCE_ACTOR_SHA'],'CRITICAL_PROBE_SHA':frozen['CRITICAL_PROBE_SHA'],'SENTINEL_SHA':frozen['SENTINEL_SHA'],'SAFE_MEAN_MAX':frozen['SAFE_MEAN_MAX'],'SAFE_P95_MAX':frozen['SAFE_P95_MAX'],'SAFE_DAMAGE_MAX':frozen['SAFE_DAMAGE_MAX'],**bounds,'EPS':EPS,'TRUST_REGION_SEPARATION':frozen['TRUST_REGION_SEPARATION'],'BACKTRACKING_MONOTONICITY':frozen['BACKTRACKING_MONOTONICITY'],'ACTOR_UPDATE_PERIOD':PERIOD,'TRUST_REGION_100':'PASS' if pass100 and accepted_inside else 'FAIL','C0_SUCCESS_100':c0(100),'C0_NEAR_HOLE_100':f"{reports['100']['c0']['near_hole_count']}/3" if '100' in reports else 'NOT_RUN','C0_SUCCESS_MARGIN_100':reports['100']['c0']['best_success_margin'] if '100' in reports else 'NOT_RUN','C0_SUPPORT_EXIT_100':reports['100']['FIRST_LOCAL_SUPPORT_EXIT_STEP'] if '100' in reports else 'NOT_RUN','C0_CONTACT_100':'PASS' if '100' in reports and reports['100']['c0']['contact_health'] and not reports['100']['c0']['table_ee_contact'] else 'FAIL' if '100' in reports else 'NOT_RUN',**{k+'_100':v for k,v in s100.items()},'TRUST_REGION_500':'PASS' if pass500 and accepted_inside else 'FAIL' if completed>=500 else 'NOT_RUN','C0_SUCCESS_500':c0(500),'C0_NEAR_HOLE_500':f"{reports['500']['c0']['near_hole_count']}/3" if '500' in reports else 'NOT_RUN','C0_SUCCESS_MARGIN_500':reports['500']['c0']['best_success_margin'] if '500' in reports else 'NOT_RUN','C0_SUPPORT_EXIT_500':reports['500']['FIRST_LOCAL_SUPPORT_EXIT_STEP'] if '500' in reports else 'NOT_RUN','C0_CONTACT_500':'PASS' if '500' in reports and reports['500']['c0']['contact_health'] and not reports['500']['c0']['table_ee_contact'] else 'FAIL' if '500' in reports else 'NOT_RUN',**{k+'_500':v for k,v in s500.items()},'ALL_ACCEPTED_CANDIDATES_INSIDE_FROZEN_BOUNDS':'YES' if accepted_inside else 'NO','R1_RETENTION':'PASS' if pass500 else 'FAIL' if completed>=500 else 'NOT_RUN','R2_RETENTION':'NOT_RUN','BEHAVIOR_CRITICAL_TRUST_REGION_RETENTION':'PENDING_R2' if pass500 else 'FAIL','CANONICAL_SAFETY_CHECK':'NOT_RUN','C1_TRAINING':'NOT_RUN','C2_TRAINING':'NOT_RUN','C3_TRAINING':'NOT_RUN','PEG_SINGLE_TASK_CONVERGED':'NO','COMPLETED_LEARNER_UPDATES':completed,'FIRST_BAD_GATE':stop,'PRIMARY_ROOT_CAUSE':rootcause,'NEXT':nexttask,'PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','EVAL_RESET_CANONICAL':'YES','EVAL_GOAL_CANONICAL':'YES','EVAL_CURRICULUM_ACTIVE':'NO','PROTECTED_ARTIFACTS_CHANGED':'NO','TRAINING_RUNNING':False,'RESIDUAL_PROCESSES':'NONE','elapsed_seconds':time.time()-start}
  after={'demo':sha(DEMO),'corrective':sha(CORR),'local':sha(LOCAL)};req(before==after,'protected');dump(OUT/'result.json',result);dump(OUT/'protected_after.json',after);print(json.dumps(p4.jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
