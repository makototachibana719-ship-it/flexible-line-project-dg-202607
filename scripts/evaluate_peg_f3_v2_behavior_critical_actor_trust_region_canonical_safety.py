#!/usr/bin/env python3
"""Stage L: read-only canonical safety evaluation of retained R2 Actor."""
from __future__ import annotations
import json,os,pickle,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,numpy as np
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model,evaluate,load_config
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
R2=ROOT/'results/training/peg_f3_v2_behavior_critical_actor_trust_region_v1_r2_20260908_001';CP=R2/'checkpoints/learner_0500_candidate_125.msgpack';OUT=ROOT/'results/diagnostics/peg/f3_v2_behavior_critical_actor_trust_region_v1_canonical_safety_20260908_001'
def req(x,m):
 if not x:raise RuntimeError(m)
def main():
 req(Path.cwd()==ROOT and not OUT.exists() and jax.default_backend()=='gpu','fresh/cwd/GPU');r2=json.loads((R2/'result.json').read_text());req(r2['R2_RETENTION']=='PASS' and r2['TRUST_REGION_500']=='PASS','R2 gate');cfg=load_config(1);demos=pickle.loads(p4.DEMO.read_bytes());demo_obs,scale=distance_model(demos)
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  common,_,_=geo.build_common_start(cfg,demos);state=flax.serialization.from_bytes(common.state,CP.read_bytes());agent=common.replace(state=state);canonical=evaluate(agent,'C3',demo_obs,scale)
 finite=all(np.isfinite(float(canonical[k])) for k in ('max_force','mean_action_saturation','median_approach_reduction','median_min_position','median_min_lateral','median_min_axial','median_min_orientation'))
 no_unsafe_contact=canonical['contact_health'] and all(not x['link01_contact'] and not x['unexpected_contact'] for x in canonical['episode_results'])
 approach=canonical['median_approach_reduction']>0 and canonical['median_min_position']<0.2
 gross_orientation=np.isfinite(canonical['median_min_orientation']) and canonical['median_min_orientation']<0.2
 workspace=finite and canonical['mean_action_saturation']<1.0
 passed=canonical['EVAL_RESET_CANONICAL']=='YES' and canonical['EVAL_GOAL_CANONICAL']=='YES' and canonical['EVAL_CURRICULUM_ACTIVE']=='NO' and no_unsafe_contact and approach and gross_orientation and workspace
 result={'CANONICAL_SAFETY_CHECK':'PASS' if passed else 'FAIL','SOURCE_R2_CHECKPOINT':str(CP),'SOURCE_R2_CHECKPOINT_SHA256':p4.sha(CP),'NO_TRAINING':'YES','NO_EXPERT':'YES','CANONICAL_RESET':'YES','CANONICAL_GOAL':'YES','APPROACH_BEHAVIOR':'PASS' if approach else 'FAIL','FORCE_CONTACT':'PASS' if no_unsafe_contact else 'FAIL','GROSS_ORIENTATION':'PASS' if gross_orientation else 'FAIL','WORKSPACE_SAFETY':'PASS' if workspace else 'FAIL','canonical_evaluation':canonical,'PRODUCTION_DEMO_SHA_UNCHANGED':'YES' if p4.sha(p4.DEMO)==p4.DSHA else 'NO','ONLINE_ADMISSION_THRESHOLD':p4.TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','EVAL_RESET_CANONICAL':canonical['EVAL_RESET_CANONICAL'],'EVAL_GOAL_CANONICAL':canonical['EVAL_GOAL_CANONICAL'],'EVAL_CURRICULUM_ACTIVE':canonical['EVAL_CURRICULUM_ACTIVE'],'NEXT':'C1' if passed else 'STOP'}
 OUT.mkdir(parents=True);p4.dump(OUT/'canonical_evaluation.json',canonical);p4.dump(OUT/'result.json',result);print(json.dumps(p4.jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
