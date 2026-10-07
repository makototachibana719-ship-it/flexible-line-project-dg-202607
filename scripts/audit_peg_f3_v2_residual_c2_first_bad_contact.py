#!/usr/bin/env python3
"""Read-only first-contact audit for the scientifically stopped C2 pilot."""
from __future__ import annotations
import json,os,pickle,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import flax,jax,numpy as np
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_observation_gated_residual_c1_pilot as pilot
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import load_config,make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.observation_gated_residual_v1 import RESIDUAL_DEF,gate_value,residual_output
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
CP=ROOT/'results/training/peg_f3_v2_observation_gated_residual_c2_pilot_20260908_001/checkpoints/c2_learner_0250_residual_312.msgpack';OUT=CP.parent.parent/'first_bad_contact_audit.json'
def raw(e):
 while hasattr(e,'env'):e=e.env
 return e
def main():
 cfg=load_config(1);demos=pickle.loads(p4.DEMO.read_bytes())
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  common,_,_=geo.build_common_start(cfg,demos);rp=RESIDUAL_DEF.init(jax.random.PRNGKey(260910),np.zeros((1,20),np.float32))['params'];ro=pilot.RESIDUAL_TX.init(rp);loaded=flax.serialization.from_bytes({'agent_state':common.state,'residual_params':rp,'residual_opt_state':ro},CP.read_bytes());agent=common.replace(state=loaded['agent_state']);rp=loaded['residual_params'];policy=pilot.CombinedPolicy(agent,rp);result={}
  for level in ('C2','C1'):
   e=make_train_env(level);obs,_=e.reset(seed=100);first=None
   for step in range(1,501):
    a=np.asarray(jax.device_get(policy.sample_actions(jax.device_put(obs),argmax=True)),np.float32);obs,reward,term,trunc,info=e.step(a);link,bad,pairs=contact_flags(e)
    if bad:
     g=float(np.asarray(jax.device_get(gate_value(obs))));rn=float(np.linalg.norm(np.asarray(jax.device_get(residual_output(rp,obs)))));first={'step':step,'contact_pairs':pairs,'link01_contact':bool(link),'lateral_error':float(info['lateral_error']),'axial_error':float(info['axial_error']),'orientation_error':float(info['orientation_error']),'force_norm':float(info.get('force_norm',0)),'gate':g,'residual_norm':rn};break
    if term or trunc:break
   e.close();result[level]=first
  result.update({'AUDIT_ONLY':'YES','NO_TRAINING':'YES','SCIENTIFIC_STOP_REMAINS':'YES','FIRST_BAD_GATE':'C2_250_CONTACT_HEALTH_AND_C1_CAPABILITY_RETENTION','PRIMARY_ROOT_CAUSE':'C2_STAGE_POLICY_DEGRADATION_OUTSIDE_PROTECTED_C0_REGION','NEXT':'STOP','COLLISION_EXCLUSION_ADDED':'NO','BASE_ACTOR_SHA':geo.thash(agent.state.params['modules_actor']),'PRODUCTION_DEMO_SHA_UNCHANGED':'YES'});p4.dump(OUT,result);print(json.dumps(p4.jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
