#!/usr/bin/env python3
"""Stage M: exact-state C1 with the frozen C0 behavior trust region."""
from __future__ import annotations
import json,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,numpy as np
from async_sac_state_sim import concat_rlpd_batches
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_behavior_critical_actor_trust_region_v1 as tr
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import block,distance_model,evaluate,fixed_batch,load_config,make_train_env
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay,snapshot_replay
R2=ROOT/'results/training/peg_f3_v2_behavior_critical_actor_trust_region_v1_r2_20260908_002';CP=R2/'checkpoints/learner_0500_candidate_125.msgpack';RP=R2/'online_replay.pkl';SAFETY=ROOT/'results/diagnostics/peg/f3_v2_behavior_critical_actor_trust_region_v1_canonical_safety_20260908_001/result.json';OUT=ROOT/'results/training/peg_f3_v2_behavior_critical_actor_trust_region_v1_c1_20260908_001';FROZEN=tr.FROZEN;SMANIFEST=tr.SMANIFEST;TOTAL=1000;MON=(0,500,1000)
def req(x,m):
 if not x:raise RuntimeError(m)
def progress(a,b):
 return b['success_count']>a['success_count'] or b['near_hole_rate']>a['near_hole_rate'] or b['median_min_lateral']<a['median_min_lateral']-1e-9 or b['median_min_position']<a['median_min_position']-1e-9 or b['median_first_divergence']>a['median_first_divergence']
def save(agent,u,n):
 p=OUT/'checkpoints'/f'c1_learner_{u:04d}_candidate_{n:03d}.msgpack'
 with p.open('xb') as f:f.write(flax.serialization.to_bytes(agent.state))
 return {'path':str(p),'sha256':p4.sha(p)}
def main():
 req(Path.cwd()==ROOT and not OUT.exists() and jax.default_backend()=='gpu','fresh/cwd/GPU');r2=json.loads((R2/'result.json').read_text());safety=json.loads(SAFETY.read_text());frozen=json.loads(FROZEN.read_text());sm=json.loads(SMANIFEST.read_text());req(r2['R2_RETENTION']=='PASS' and safety['CANONICAL_SAFETY_CHECK']=='PASS' and p4.sha(CP)==safety['SOURCE_R2_CHECKPOINT_SHA256'],'handoff gates');req(tuple(frozen['ALPHA_CANDIDATES'])==tr.ALPHAS and frozen['ACTOR_UPDATE_PERIOD']==4,'frozen trust region')
 OUT.mkdir(parents=True);(OUT/'checkpoints').mkdir();before={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};p4.dump(OUT/'protected_before.json',before);cfg=load_config(1);demos=pickle.loads(p4.DEMO.read_bytes());demo_obs,scale=distance_model(demos);started=time.time();logs=[];collections=[];checkpoints={}
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  common,_,demo=geo.build_common_start(cfg,demos);geo.OUT=OUT;rows,snaps,_,manifest=geo.build_critical(common);req(manifest['sha256']==frozen['CRITICAL_PROBE_SHA'],'probe');sel=sm['probe_indices'];obs_probe=np.asarray([x['observation'] for x in rows],np.float32);ref=np.asarray([x['reference_action'] for x in rows],np.float32);bounds={k:frozen[k] for k in ('TAU_MEAN_ACTION_DRIFT','TAU_P95_ACTION_DRIFT','TAU_MAX_ONE_STEP_DAMAGE')};agent=common.replace(state=flax.serialization.from_bytes(common.state,CP.read_bytes()));env=make_train_env('C1');online=make_replay_buffer(env,capacity=cfg['training']['replay_capacity'],type='replay_buffer');ri=restore_replay(online,RP);req(ri['sha256']==r2['FINAL_ONLINE_REPLAY_SHA256'],'replay')
  initial_c1=evaluate(agent,'C1',demo_obs,scale);initial_c0=evaluate(agent,'C0',demo_obs,scale);p4.dump(OUT/'initial_c1.json',initial_c1);p4.dump(OUT/'initial_c0.json',initial_c0);checkpoints['0']=save(agent,0,0);arng=jax.random.PRNGKey(19500);rng=np.random.default_rng(20261008);obs=None
  for u in range(1,TOTAL+1):
   if (u-1)%100==0:obs,arng,ci=p4.collect(env,agent,online,cfg,demo_obs,scale,obs,arng);ci.update(block=(u-1)//100+1,replay_size=len(online));collections.append(ci);frozen_online=len(online)
   batch=concat_rlpd_batches(fixed_batch(online,rng.integers(0,frozen_online,32)),fixed_batch(demo,rng.integers(0,len(demo),32)));agent,ci=p4.critic_update_only(agent,batch);block((agent,ci))
   if u%4==0:agent,cl=tr.actor_candidate(agent,batch,obs_probe,ref,rows,snaps,sel,bounds,u);logs.append(cl)
   if u in (500,1000):checkpoints[str(u)]=save(agent,u,len(logs));p4.dump(OUT/f'c1_eval_{u:04d}.json',evaluate(agent,'C1',demo_obs,scale));p4.dump(OUT/f'c0_diag_{u:04d}.json',evaluate(agent,'C0',demo_obs,scale))
  env.close();final_c1=json.loads((OUT/'c1_eval_1000.json').read_text());final_c0=json.loads((OUT/'c0_diag_1000.json').read_text());canonical=evaluate(agent,'C3',demo_obs,scale);replay_info=snapshot_replay(online,OUT/'online_replay.pkl');p4.dump(OUT/'canonical_evaluation.json',canonical);p4.dump(OUT/'candidate_updates.json',logs);p4.dump(OUT/'collections.json',collections);p4.dump(OUT/'checkpoints.json',checkpoints)
  accepted=[x for x in logs if x['accepted']];inside=all(x['post_accept']['MEAN_ACTION_DRIFT']<=bounds['TAU_MEAN_ACTION_DRIFT']+tr.EPS and x['post_accept']['P95_ACTION_DRIFT']<=bounds['TAU_P95_ACTION_DRIFT']+tr.EPS and x['post_accept']['MAX_ONE_STEP_DAMAGE']<=bounds['TAU_MAX_ONE_STEP_DAMAGE']+tr.EPS for x in accepted);c0ok=final_c0['success_count']>=2 and final_c0['contact_health'];c1prog=progress(initial_c1,final_c1);passed=c1prog and c0ok and final_c1['contact_health'] and inside
  result={'C1_TRAINING':'PASS' if passed else 'FAIL','FIRST_BAD_CURRICULUM_LEVEL':None if passed else 'C1','START_R2_CHECKPOINT_SHA256':p4.sha(CP),'START_REPLAY_SHA256':p4.sha(RP),'FINAL_CHECKPOINT':checkpoints['1000'],'FINAL_REPLAY':str(OUT/'online_replay.pkl'),'FINAL_REPLAY_SHA256':replay_info['sha256'],'FINAL_REPLAY_SIZE':replay_info['size'],'C1_PROGRESS':'PASS' if c1prog else 'FAIL','INITIAL_C1':initial_c1,'FINAL_C1':final_c1,'C0_DIAGNOSTIC_SUCCESS':f"{final_c0['success_count']}/3",'C0_DIAGNOSTIC_CONTACT':'PASS' if final_c0['contact_health'] else 'FAIL','ACTOR_CANDIDATE_COUNT':len(logs),'ACTOR_ACCEPT_COUNT':len(accepted),'ACTOR_REJECT_COUNT':len(logs)-len(accepted),'ALL_ACCEPTED_CANDIDATES_INSIDE_FROZEN_BOUNDS':'YES' if inside else 'NO','CRITICAL_PROBE_SHA':manifest['sha256'],'SENTINEL_SHA':frozen['SENTINEL_SHA'],'CANONICAL_EVALUATION':canonical,'PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':p4.TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','EVAL_RESET_CANONICAL':canonical['EVAL_RESET_CANONICAL'],'EVAL_GOAL_CANONICAL':canonical['EVAL_GOAL_CANONICAL'],'EVAL_CURRICULUM_ACTIVE':canonical['EVAL_CURRICULUM_ACTIVE'],'NEXT':'C2' if passed else 'STOP','elapsed_seconds':time.time()-started}
  after={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};req(before==after,'protected');p4.dump(OUT/'result.json',result);p4.dump(OUT/'protected_after.json',after);print(json.dumps(p4.jv({k:v for k,v in result.items() if k not in ('INITIAL_C1','FINAL_C1','CANONICAL_EVALUATION')}),indent=2,sort_keys=True))
if __name__=='__main__':main()
