#!/usr/bin/env python3
"""Stages C/D: zero-init and randomized C0 invariance proof; no training."""
from __future__ import annotations
import json,os,pickle,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import jax,numpy as np
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import load_config
from training.peg_insertion.observation_gated_residual_v1 import RESIDUAL_DEF,combined_mode,gate_value,residual_output
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
AUDIT=ROOT/'results/preflight/peg/f3_v2_observation_gated_residual_feasibility_20260908_001';OUT=ROOT/'results/preflight/peg/f3_v2_observation_gated_residual_architecture_20260908_001'
def req(x,m):
 if not x:raise RuntimeError(m)
def maxdelta(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def main():
 req(Path.cwd()==ROOT and not OUT.exists() and jax.default_backend()=='gpu','fresh/cwd/GPU');audit=json.loads((AUDIT/'result.json').read_text());req(audit['SCIENTIFIC_GATE']=='PASS','gate audit');req(p4.sha(p4.DEMO)==p4.DSHA,'Demo');OUT.mkdir(parents=True);before={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};p4.dump(OUT/'protected_before.json',before);demos=pickle.loads(p4.DEMO.read_bytes());cfg=load_config(1)
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  base,_,_=geo.build_common_start(cfg,demos);base_sha=geo.thash(base.state.params['modules_actor']);geo.OUT=OUT;rows,_,_,manifest=geo.build_critical(base);req(base_sha==audit['BASE_ACTOR_SHA'] and manifest['sha256']==audit['CRITICAL_PROBE_SHA'],'base/probe');c0=np.asarray([x['observation'] for x in rows],np.float32);c1rows=json.loads((AUDIT/'c1_trace.json').read_text());c1=np.asarray([x['obs'] for x in c1rows if x['step']<=50],np.float32)
  params=RESIDUAL_DEF.init(jax.random.PRNGKey(260908),np.zeros((1,20),np.float32))['params'];base_c0=np.asarray(base.sample_actions(observations=jax.device_put(c0),argmax=True));combined_c0=np.asarray(combined_mode(base,params,c0));base_c1=np.asarray(base.sample_actions(observations=jax.device_put(c1),argmax=True));combined_c1=np.asarray(combined_mode(base,params,c1));initial=max(maxdelta(base_c0,combined_c0),maxdelta(base_c1,combined_c1));zero_res=float(np.max(np.abs(np.asarray(residual_output(params,c1)))))
  keys=iter(jax.random.split(jax.random.PRNGKey(260909),len(jax.tree_util.tree_leaves(params))));perturbed=jax.tree_util.tree_map(lambda x:x+0.5*jax.random.normal(next(keys),x.shape),params);pert_c0=np.asarray(combined_mode(base,perturbed,c0));pert_c1=np.asarray(combined_mode(base,perturbed,c1));invariance=maxdelta(base_c0,pert_c0);outside_change=maxdelta(base_c1,pert_c1);g0=np.asarray(gate_value(c0));g1=np.asarray(gate_value(c1));after_sha=geo.thash(base.state.params['modules_actor']);passed=initial<=1e-7 and invariance<=1e-7 and outside_change>1e-7 and np.all(g0==0) and np.any(g1>0) and base_sha==after_sha
  result={'RESIDUAL_ARCHITECTURE_STATIC_PROOF':'PASS' if passed else 'FAIL','NO_TRAINING':'YES','BASE_ACTOR_SHA':base_sha,'BASE_ACTOR_SHA_UNCHANGED':'YES' if after_sha==base_sha else 'NO','GATE_SOURCE':audit['GATE_SOURCE'],'GATE_INNER_BOUND':audit['GATE_INNER_BOUND'],'GATE_OUTER_BOUND':audit['GATE_OUTER_BOUND'],'RESIDUAL_ARCHITECTURE':'Obs20 -> Dense256/ReLU -> Dense256/ReLU -> zero-init Dense6 -> tanh bounded pre-tanh location shift; base scale and gripper loc unchanged','RESIDUAL_PARAMETER_COUNT':int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params))),'BASE_DISTRIBUTION_SCALE_SEMANTICS':'stop-gradient existing base scale_diag, no residual std/log_std','GRIPPER_SEMANTICS':'base pre-tanh loc and scale at a6 unchanged','INITIAL_RESIDUAL_OUTPUT_MAX':zero_res,'MAX_INITIAL_COMBINED_BASE_ACTION_DELTA_C0':maxdelta(base_c0,combined_c0),'MAX_INITIAL_COMBINED_BASE_ACTION_DELTA_ALL_TESTED':initial,'C0_BASE_INVARIANCE_MAX_ACTION_DELTA':invariance,'PERTURBED_C1_COMBINED_ACTION_DELTA':outside_change,'C0_GATE_ZERO_FRACTION':float(np.mean(g0==0)),'C1_GATE_NONZERO_FRACTION':float(np.mean(g1>0)),'CRITICAL_PROBE_SHA':manifest['sha256'],'SCIENTIFIC_GATE':'PASS' if passed else 'NOT_EVALUATED_ENGINEERING_INTERRUPTION','NEXT':'C1_RESIDUAL_250_PILOT' if passed else 'FIX_ARCHITECTURE_IN_NEW_FILES'}
  after={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};req(before==after,'protected');p4.dump(OUT/'result.json',result);p4.dump(OUT/'protected_after.json',after);print(json.dumps(p4.jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
