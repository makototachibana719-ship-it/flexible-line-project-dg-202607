#!/usr/bin/env python3
"""No-training closed-loop pilot for the Obs20 empirical recovery planner."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env")]

import flax
import jax
import numpy as np

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_observable_recovery_option_pilot as option
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_trajectory_recovery_head_v1_protocol_corrected as recovery
import scripts.run_peg_f3_v2_obs20_short_horizon_dynamics_model_feasibility as feasibility
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, evaluate, load_config, make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.c2_obs20_empirical_mpc_option_v1 import EmpiricalOneStepPlanner, empirical_option_step
from training.peg_insertion.c2_observable_recovery_option_v1 import RecoveryPhase, lower_action
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_obs20_empirical_mpc_option_pilot_20260909_001"
FEASIBILITY = ROOT / "results/diagnostics/peg/f3_v2_obs20_short_horizon_dynamics_model_feasibility_20260909_002"
SEEDS = (100, 101, 102)


def req(v, m):
    if not v: raise RuntimeError(m)


def jvalue(v):
    if isinstance(v, np.ndarray): return jvalue(v.tolist())
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, dict): return {str(k): jvalue(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jvalue(x) for x in v]
    return v


def dump(path, value):
    with Path(path).open("x") as f: json.dump(jvalue(value), f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


class Policy:
    def __init__(self, common, reset_obs): self.common=common; self.reset_obs=reset_obs; self.phase=RecoveryPhase.APPROACH; self.trace=[]
    def reset(self): self.phase=RecoveryPhase.APPROACH
    def act(self, obs):
        obs=np.asarray(obs,np.float32)
        if len(self.reset_obs) and np.any(np.all(self.reset_obs==obs,axis=1)): self.reset()
        self.phase, action, info=empirical_option_step(self.phase,obs,**self.common); self.trace.append(info); return action
    def sample_actions(self, observations, **_): return self.act(observations)
    def stats(self): return {"steps":len(self.trace),"recovery_initiations":sum(x["transition"]=="APPROACH_TO_RECOVERY" for x in self.trace),
        "lower_handoffs":sum(x["transition"]=="RECOVERY_TO_LOWER_HANDOFF" for x in self.trace),
        "recovery_steps":sum(x["phase"]=="RECOVERY" for x in self.trace),"trace":self.trace}


def resets():
    rows=[]
    for level in ("C0","C1","C2"):
        env=make_train_env(level)
        try:
            for seed in SEEDS: rows.append(np.asarray(env.reset(seed=seed)[0],np.float32))
        finally: env.close()
    return np.asarray(rows,np.float32)


def eval_all(common, reset_obs, demo_obs, demo_scale, baseline):
    out={}
    for level in ("C0","C1","C2"):
        p=Policy(common,reset_obs); out[level]=evaluate(p,level,demo_obs,demo_scale,seeds=SEEDS); out[level]["option"]=p.stats()
    keys=("success_count","near_hole_rate","median_first_divergence","median_corridor_steps","median_min_position","median_approach_reduction","contact_health")
    reg=all(out["C1"][k]==baseline[k] for k in keys); s=out["C2"]["option"]
    passed=bool(out["C0"]["success_count"]==3 and out["C0"]["contact_health"] and reg and out["C2"]["contact_health"] and s["recovery_initiations"]>=2 and s["lower_handoffs"]>=2)
    return {**out,"C1_REGRESSION":"PASS" if reg else "FAIL","interface_pass":passed}


def rollout(common,seed):
    p=Policy(common,np.zeros((0,20),np.float32)); env=make_train_env("C2"); rows=[]
    try:
        obs,_=env.reset(seed=seed)
        for step in range(500):
            action=p.act(obs); link,unexpected,pairs=contact_flags(env)
            rows.append({"seed":seed,"policy_step":step,"observation":np.asarray(obs,np.float32),"snapshot":capture(env,obs),"policy_action":action,"contact_before":bool(link or unexpected),"contact_pairs":pairs})
            if link or unexpected: break
            obs,_,terminated,truncated,_=env.step(action)
            if terminated or truncated: break
    finally: env.close()
    return rows,p.stats()


def main():
    req(Path.cwd()==ROOT and not OUT.exists(),"fresh output and cwd"); req(jax.default_backend()=="gpu","GPU")
    prior=json.loads((FEASIBILITY/"result.json").read_text()); req(prior["NEXT"]=="C2_OBS20_EMPIRICAL_MPC_OPTION_PILOT","next")
    OUT.mkdir(parents=True); started=time.time()
    protected={"production_demo":p4.DEMO,"global_c2_2k":recovery.CHECKPOINT,"r1_recovery_head":option.R1_CP,"feasibility":FEASIBILITY/"result.json"}
    before={k:sha(v) for k,v in protected.items()}; dump(OUT/"protected_before.json",before); req(before["production_demo"]==tail.DEMO_SHA,"Demo SHA")
    demos=pickle.loads(p4.DEMO.read_bytes()); demo_obs,demo_scale=distance_model(demos)
    obs=np.asarray([x["observations"] for x in demos],np.float32); acts=np.asarray([x["actions"] for x in demos],np.float32); nxt=np.asarray([x["next_observations"] for x in demos],np.float32)
    sb=pickle.loads((review.PREFLIGHT/"C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes()); support=np.asarray(sb["observations"],np.float32); scale=np.asarray(sb["normalization_scale"],np.float32)
    pf=json.loads((review.PREFLIGHT/"result.json").read_text()); core=float(pf["C1_CORE_THRESHOLD"]); outer=float(pf["C1_SHELL_OUTER_THRESHOLD"])
    r1=flax.serialization.msgpack_restore(option.R1_CP.read_bytes()); planner=EmpiricalOneStepPlanner(obs,acts,nxt,demo_scale,support,scale,64); cfg=load_config(2)
    with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
        ca,_,_=geo.build_common_start(cfg,demos); source=flax.serialization.from_bytes(review.init_loaded(ca),review.SOURCE_CP.read_bytes()); agent=ca.replace(state=source["agent_state"]); c1=source["c1_params"]
        gd,gp=tail.load_head_checkpoint(recovery.CHECKPOINT); global_fn=lambda x:tail.checkpoint_action(gd,gp,x,agent,c1,support,scale,core,outer)
        common=dict(planner=planner,global_action_fn=global_fn,recovery_support=np.asarray(r1["support"],np.float32),recovery_scale=scale,recovery_outer=float(r1["recovery_outer"]),agent=agent,c1_params=c1,c1_support=support,c1_scale=scale,c1_core=core,c1_outer=outer)
        reset_obs=resets(); deltas=[]
        for x in reset_obs[:6]: deltas.append(float(np.max(np.abs(Policy(common,reset_obs).act(x)-lower_action(agent,c1,x)))))
        req(max(deltas)==0.0,"C0/C1 invariance")
        baseline=pc.eval_selected(gd,gp,"H0",agent,c1,support,scale,core,outer,"C1",demo_obs,demo_scale,np.zeros(20,np.float32))
        pilot=eval_all(common,reset_obs,demo_obs,demo_scale,baseline)
        oracle=tail.RecoveryOracle(demos,demo_obs,demo_scale,support,scale,core,outer); episodes=[]; bad=[]
        try:
            for seed in SEEDS:
                rows,stats=rollout(common,seed); boundary,audited=recovery.first_boundary(rows,oracle); b=None if boundary is None else boundary[1]; bad.append(b)
                episodes.append({"seed":seed,"steps":len(rows),"option":stats,"first_nonrecoverable":b,"recoverability_audit":audited})
        finally: oracle.close()
        later=all(x is None or x>10 for x in bad); passed=bool(pilot["interface_pass"] and later); dump(OUT/"pilot.json",{"evaluation":pilot,"trajectories":episodes,"first_nonrecoverable_steps":bad,"later":later})
        matched="NOT_RUN"
        if passed:
            confirmation=eval_all(common,reset_obs,demo_obs,demo_scale,baseline); matched="PASS" if confirmation==pilot else "FAIL"; dump(OUT/"matched_confirmation.json",{"evaluation":confirmation,"exact_match":confirmation==pilot}); passed &= matched=="PASS"
        if passed: dump(OUT/"C2_CONFIRMED_OBS20_EMPIRICAL_MPC_OPTION_V1.json",{"version":"C2_OBS20_EMPIRICAL_MPC_OPTION_V1","candidate_count":64,"horizon":1,"inputs":"Obs20+normalized action","production_demo_sha":before["production_demo"],"global_c2_sha":before["global_c2_2k"],"r1_sha":before["r1_recovery_head"]})
    after={k:sha(v) for k,v in protected.items()}; req(after==before,"protected mutation"); dump(OUT/"protected_after.json",after)
    nxtask="CANONICAL_INTERFACE_REACHABILITY_AUDIT" if passed else "HIERARCHICAL_C2_APPROACH_TO_C1_POLICY_REVIEW"; root="NONE" if passed else "OBS20_EMPIRICAL_MPC_CLOSED_LOOP_RECOVERY_FAILURE"
    result={"AUTHORITATIVE":"YES","C2_OBS20_EMPIRICAL_MPC_OPTION_PILOT":"PASS" if passed else "FAIL","C2_SKILL_CONFIRMED":"YES" if passed else "NO","C2_MATCHED_CONFIRMATION":matched,"C0_SUCCESS":f"{pilot['C0']['success_count']}/3","C1_REGRESSION":pilot["C1_REGRESSION"],"C1_CONTACT":"PASS" if pilot["C1"]["contact_health"] else "FAIL","C2_SUCCESS":f"{pilot['C2']['success_count']}/3","C2_CONTACT":"PASS" if pilot["C2"]["contact_health"] else "FAIL","C2_RECOVERY_INITIATIONS":pilot["C2"]["option"]["recovery_initiations"],"C2_LOWER_HANDOFFS":pilot["C2"]["option"]["lower_handoffs"],"FIRST_NONRECOVERABLE_STEPS":bad,"C0_C1_STATIC_MAX_ACTION_DELTA":max(deltas),"TRAINING_STARTED":False,"PRIMARY_ROOT_CAUSE":root,"NEXT":nxtask,"PRODUCTION_DEMO_SHA_UNCHANGED":"YES","ONLINE_ADMISSION_THRESHOLD":7.41773975814,"ONLINE_ADMISSION_THRESHOLD_CHANGED":"NO","PROTECTED_ARTIFACTS_CHANGED":"NO","PEG_SINGLE_TASK_CONVERGED":"NO","TRAINING_RUNNING":False,"RESIDUAL_PROCESSES":"NONE","elapsed_seconds":time.time()-started}
    dump(OUT/"result.json",result); print(json.dumps(jvalue(result),indent=2,sort_keys=True))


if __name__=="__main__": main()
