#!/usr/bin/env python3
"""Stages A/B: no-training Obs20 residual-gate feasibility proof."""
from __future__ import annotations
import json,os,pickle,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import jax,numpy as np
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import action,distance_model,load_config,make_peg_state_env,make_train_env
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
OUT=ROOT/'results/preflight/peg/f3_v2_observation_gated_residual_feasibility_20260908_001';TOL=1e-7;INNER=1.0
def req(x,m):
 if not x:raise RuntimeError(m)
def obs_errors(obs):
 x=np.asarray(obs,np.float64);return float(np.linalg.norm(x[14:16])),float(abs(x[13]))
def radius(obs):
 l,a=obs_errors(obs);return max(l/.010,a/.020)
def gate(obs,outer):
 r=radius(obs)
 if r<=INNER:return 0.0
 if r>=outer:return 1.0
 x=(r-INNER)/(outer-INNER);return float(x*x*(3-2*x))
def trace(level,agent,seeds=(100,101,102)):
 env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=500) if level=='C3' else make_train_env(level);rows=[]
 try:
  for seed in seeds:
   obs,_=env.reset(seed=seed);rows.append({'level':level,'seed':seed,'step':0,'obs':np.asarray(obs,np.float32)})
   for step in range(1,501):
    a,_=action(agent,obs,deterministic=True);n,r,t,tr,info=env.step(a);ol,oa=obs_errors(n);el=float(info['lateral_error']);ea=float(info['axial_error']);rows.append({'level':level,'seed':seed,'step':step,'obs':np.asarray(n,np.float32),'obs_lateral':ol,'obs_axial':oa,'env_lateral':el,'env_axial':ea,'lateral_abs_error':abs(ol-el),'axial_abs_error':abs(oa-ea),'predicate_obs':ol<=.010 and oa<=.020,'predicate_env':el<=.010 and ea<=.020,'success':bool(info.get('is_success',False))});obs=n
    if t or tr:break
  return rows
 finally:env.close()
def main():
 req(Path.cwd()==ROOT and not OUT.exists() and jax.default_backend()=='gpu','fresh/cwd/GPU');req(p4.sha(p4.DEMO)==p4.DSHA,'Demo');OUT.mkdir(parents=True);before={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};p4.dump(OUT/'protected_before.json',before);demos=pickle.loads(p4.DEMO.read_bytes());cfg=load_config(1);demo_obs,scale=distance_model(demos)
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  base,_,_=geo.build_common_start(cfg,demos);base_sha=geo.thash(base.state.params['modules_actor']);req(base_sha=='a4986a9a703f0629cc49e5092bbf7b245bc7766ab4096596a2ec60749d06d265','base');geo.OUT=OUT;probe,snaps,sel,manifest=geo.build_critical(base);req(manifest['sha256']=='81b03c51951727533b1a5ba11dc0578d9bf4e953a9f024fe544cf7a7455a43c1','probe')
  traces={k:trace(k,base) for k in ('C0','C1','C3')};measured=[x for rows in traces.values() for x in rows if x['step']>0];maxlat=max(x['lateral_abs_error'] for x in measured);maxax=max(x['axial_abs_error'] for x in measured);predicate_exact=all(x['predicate_obs']==x['predicate_env'] for x in measured);derivable=max(maxlat,maxax)<=TOL and predicate_exact
  successful_obs=[]
  for d in demos:successful_obs.extend((np.asarray(d['observations'],np.float32),np.asarray(d['next_observations'],np.float32)))
  outside=sorted(r for r in map(radius,successful_obs) if r>INNER+1e-12);req(outside,'transition support');outer=float(outside[0]);req(outer>INNER,'outer')
  probe_g=np.asarray([gate(x['observation'],outer) for x in probe]);c1early=[x for x in traces['C1'] if x['step']<=50];c1g=np.asarray([gate(x['obs'],outer) for x in c1early]);c0zero=float(np.mean(probe_g==0));c1nonzero=float(np.mean(c1g>0));static=c0zero==1.0 and c1nonzero>0
  result={'RESIDUAL_GATE_FEASIBILITY_AUDIT':'PASS' if derivable and static else 'FAIL','NO_TRAINING':'YES','BASE_ACTOR_SHA':base_sha,'NEAR_HOLE_GATE_OBS_DERIVABLE':'YES' if derivable else 'NO','OBS20_MAPPING':{'axial_error':'abs(obs[13])','lateral_error':'norm(obs[14:16])'},'FLOATING_POINT_TOLERANCE':TOL,'MAX_LATERAL_RECONSTRUCTION_ERROR':maxlat,'MAX_AXIAL_RECONSTRUCTION_ERROR':maxax,'NEAR_HOLE_PREDICATE_EXACT_FRACTION':float(np.mean([x['predicate_obs']==x['predicate_env'] for x in measured])),'GATE_SOURCE':'OBS20_NEAR_HOLE' if derivable else 'NOT_SELECTED','GATE_INNER_BOUND':{'normalized_radius':INNER,'lateral':.010,'axial':.020},'GATE_OUTER_BOUND':outer,'GATE_OUTER_DERIVATION':'minimum normalized radius >1 among immutable successful production-Demo observation/next-observation support','GATE_FORM':'hard exact zero for radius<=1; smoothstep to one over (1, outer); one at/above outer','C0_GATE_ZERO_FRACTION':c0zero,'C1_GATE_NONZERO_FRACTION':c1nonzero,'C1_GATE_MEAN_EARLY':float(c1g.mean()),'C1_GATE_P95_EARLY':float(np.quantile(c1g,.95)),'C0_SUCCESS_COUNTS_BY_TRACE':int(sum(any(x.get('success',False) for x in traces['C0'] if x['seed']==s) for s in (100,101,102))),'TRACE_COUNTS':{k:len(v) for k,v in traces.items()},'CRITICAL_PROBE_SHA':manifest['sha256'],'STATIC_PROOF':'PASS' if static else 'FAIL','SCIENTIFIC_GATE':'PASS' if derivable and static else 'FAIL','NEXT':'RESIDUAL_ARCHITECTURE_STATIC_PROOF' if derivable and static else 'STOP','PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':p4.TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','PROTECTED_ARTIFACTS_CHANGED':'NO'}
  for k,v in traces.items():p4.dump(OUT/f'{k.lower()}_trace.json',v)
  after={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};req(before==after,'protected');p4.dump(OUT/'result.json',result);p4.dump(OUT/'protected_after.json',after);print(json.dumps(p4.jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
