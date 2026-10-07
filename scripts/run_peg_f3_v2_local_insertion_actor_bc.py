#!/usr/bin/env python3
"""One bounded actor-only local insertion BC run with protected gates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

ROOT=Path("/home/makoto/flexible_line_project")
sys.path[:0]=[str(ROOT),str(ROOT/"external/serl/serl_launcher"),str(ROOT/"external/serl/franka_sim"),
 str(ROOT/"external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
 str(ROOT/"external/packages/peg_insertion/peg_insertion/code/env")]

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from peg_env_adapter import make_peg_state_env
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    action as sample_action, demo_distance, distance_model, load_config, make_train_env, raw_env,
)
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from serl_launcher.common.optimizers import make_optimizer

CONFIG=ROOT/"configs/peg_insertion/peg_v2_local_insertion_actor_supervision_20260908.yaml"
LOCAL_DIR=ROOT/"results/datasets/peg_v2_local_insertion_supervision_20260908_001"
LOCAL=LOCAL_DIR/"local_insertion_supervision.pkl"
OUTPUT=ROOT/"results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001"
SOURCE_RUN=ROOT/"results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002"
SOURCE_CKPT=SOURCE_RUN/"checkpoints"
PROD=ROOT/"results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
TRACE=ROOT/"results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
DEMO_SHA="4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
LOCAL_SHA="e3dc6738696dd53e1757d60523669f3a180475946d4ba21714f160848d0629a0"
THRESHOLD=7.41773975814
LIMITS=np.asarray([.002,.003,.05],np.float64)
HIST_MARGIN=2.9458398747637045

def require(v,m):
    if not v: raise RuntimeError(m)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def sha_tree(tree):
    h=hashlib.sha256()
    for path,leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a=np.asarray(jax.device_get(leaf)); h.update(jax.tree_util.keystr(path).encode()); h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def jv(v):
    if isinstance(v,np.ndarray):return jv(v.tolist())
    if isinstance(v,(np.bool_,bool)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    if isinstance(v,dict):return {str(k):jv(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jv(x) for x in v]
    return v

def dump(path,value):
    with Path(path).open("x") as f:json.dump(jv(value),f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def protected(cfg):
    paths={"production_demo":PROD,"source_checkpoint":SOURCE_CKPT/"checkpoint_1000",
      "source_result":SOURCE_RUN/"result.json","exact_restore":ROOT/"training/peg_insertion/exact_state_curriculum.py",
      "curriculum_config":ROOT/"configs/peg_insertion/peg_v2_exact_restore_curriculum_rlpd_20260907.yaml"}
    return {k:sha(v) for k,v in paths.items()}

def metrics(agent,obs,acts):
    pred=np.asarray(jax.device_get(agent.forward_policy(jnp.asarray(obs),rng=None,train=False).mode()))[:,:6]
    target=np.asarray(acts)[:,:6]; delta=pred-target
    denom=np.linalg.norm(pred,axis=1)*np.linalg.norm(target,axis=1)
    cosine=np.divide(np.sum(pred*target,axis=1),denom,out=np.zeros_like(denom),where=denom>1e-9)
    return {"mae_a0_a5":float(np.mean(np.abs(delta))),"cosine_a0_a5":float(np.mean(cosine)),
            "max_error_a0_a5":float(np.max(np.abs(delta))),"mse_a0_a5":float(np.mean(delta**2))}

def is_descendant(model,child,ancestor):
    while child>0:
        if child==ancestor:return True
        child=int(model.body_parentid[child])
    return child==ancestor

def table_ee(env):
    r=raw_env(env);m=r.model;d=r.data
    tables={i for i in range(m.nbody) if "table" in (m.body(i).name or "").lower()}
    anchors={i for i in range(m.nbody) if any(x in (m.body(i).name or "").lower() for x in ("ft_sensor_mount","gripper","assemble_axle","end_effector"))}
    ee={i for i in range(m.nbody) if any(is_descendant(m,i,a) for a in anchors)}
    return any({int(m.geom_bodyid[d.contact[i].geom1]),int(m.geom_bodyid[d.contact[i].geom2])}&tables and
               {int(m.geom_bodyid[d.contact[i].geom1]),int(m.geom_bodyid[d.contact[i].geom2])}&ee for i in range(int(d.ncon)))

def evaluate(agent,level,demo_obs,scale,seeds=(100,101,102)):
    env=make_train_env(level) if level=="C0" else make_peg_state_env(render_mode="rgb_array",max_episode_steps=500)
    episodes=[]
    try:
      for seed in seeds:
        obs,_=env.reset(seed=seed);initial=float(np.linalg.norm(np.asarray(obs)[13:16]));first_div=None;corridor=0;near=False;best=math.inf
        mins={"lateral":math.inf,"axial":math.inf,"orientation":math.inf,"position":initial};max_force=0.;unexpected=False;table=False
        for step in range(1,501):
          act,_=sample_action(agent,obs,deterministic=True);obs,reward,term,trunc,info=env.step(act)
          distance=demo_distance(obs,demo_obs,scale);corridor+=int(distance<=THRESHOLD)
          if distance>THRESHOLD and first_div is None:first_div=step
          vals=np.asarray([info["lateral_error"],info["axial_error"],info["orientation_error"]],np.float64)
          margin=float(np.max(vals/LIMITS));best=min(best,margin);near|=bool(vals[0]<=.010 and vals[1]<=.020)
          mins["lateral"]=min(mins["lateral"],float(vals[0]));mins["axial"]=min(mins["axial"],float(vals[1]));mins["orientation"]=min(mins["orientation"],float(vals[2]));mins["position"]=min(mins["position"],float(np.linalg.norm(np.asarray(obs)[13:16])))
          force=float(info.get("force_norm",0.));max_force=max(max_force,force if np.isfinite(force) else math.inf)
          _,bad,_=contact_flags(env);unexpected|=bool(bad);table|=table_ee(env)
          if term or trunc:break
        episodes.append({"seed":seed,"success":bool(info.get("is_success",term and not trunc)),"near_hole":near,
          "best_success_margin":best,"first_divergence":first_div or step+1,"corridor_steps":corridor,"max_force":max_force,
          "unexpected_contact":unexpected,"table_ee_contact":table,"contact_health":not unexpected and not table and np.isfinite(max_force),
          "steps":step,"approach_reduction":initial-mins["position"],**{f"min_{k}":v for k,v in mins.items()}})
    finally:env.close()
    med=lambda k:float(np.median([x[k] for x in episodes]))
    return {"level":level,"episodes":episodes,"success_count":sum(x["success"] for x in episodes),
      "near_hole_count":sum(x["near_hole"] for x in episodes),"contact_health":all(x["contact_health"] for x in episodes),
      "table_ee_contact":any(x["table_ee_contact"] for x in episodes),"best_success_margin":min(x["best_success_margin"] for x in episodes),
      "median_first_divergence":med("first_divergence"),"max_force":max(x["max_force"] for x in episodes),
      "median_approach_reduction":med("approach_reduction"),"median_min_position":med("min_position"),
      "median_min_lateral":med("min_lateral"),"median_min_axial":med("min_axial"),"median_min_orientation":med("min_orientation")}

def main():
    require(not OUTPUT.exists(),"fresh BC output required");OUTPUT.mkdir(parents=True,exist_ok=False);(OUTPUT/"checkpoints").mkdir()
    config=yaml.safe_load(CONFIG.read_text());cfg=load_config(1)
    require(config["online_admission_threshold"]==THRESHOLD and sha(PROD)==DEMO_SHA and sha(LOCAL)==LOCAL_SHA,"frozen bindings")
    before=protected(cfg);dump(OUTPUT/"protected_before.json",before)
    local=pickle.loads(LOCAL.read_bytes());split=json.loads((LOCAL_DIR/"split.json").read_text())
    train_set=set(split["train_episodes"]);valid_set=set(split["valid_episodes"])
    train=[x for x in local if x["source_episode"] in train_set];valid=[x for x in local if x["source_episode"] in valid_set]
    require(len(train)==270 and len(valid)==60,"split rows")
    train_obs=np.asarray([x["observations"] for x in train],np.float32);train_act=np.asarray([x["actions"] for x in train],np.float32)
    valid_obs=np.asarray([x["observations"] for x in valid],np.float32);valid_act=np.asarray([x["actions"] for x in valid],np.float32)
    demos=pickle.loads(PROD.read_bytes());demo_obs,scale=distance_model(demos)
    trace=[json.loads(x) for x in TRACE.read_text().splitlines() if x.strip()]
    expert_first=np.asarray(trace[137]["action"],np.float32)

    with portable_peg_runtime(ROOT):
      env=make_train_env("C0");first_obs,_=env.reset(seed=100)
      fresh=make_combined_q_anchor_agent(cfg["seed"],env.observation_space.sample(),env.action_space.sample(),cfg);env.close()
      agent=fresh.replace(state=restore_fresh_state(SOURCE_CKPT,fresh.state,1000))
      old_first=np.asarray(jax.device_get(agent.sample_actions(observations=jax.device_put(first_obs),argmax=True)),np.float32)
      old_metrics=metrics(agent,valid_obs,valid_act);old_delta=float(np.linalg.norm(old_first[:6]-expert_first[:6]))
      hashes_before={"actor":sha_tree(agent.state.params["modules_actor"]),"critic":sha_tree(agent.state.params["modules_critic"]),
        "temperature":sha_tree(agent.state.params["modules_temperature"]),"target":sha_tree(agent.state.target_params)}
      actor_params=agent.state.params["modules_actor"]
      tx=make_optimizer(learning_rate=float(config["actor_only_bc"]["actor_learning_rate"]),warmup_steps=int(config["actor_only_bc"]["actor_warmup_steps"]))
      opt_state=tx.init(actor_params);rng=np.random.default_rng(int(config["actor_only_bc"]["seed"]));reports=[]
      apply_fn=agent.state.apply_fn
      @jax.jit
      def bc_step(params,opt_state,obs,targets):
        def loss_fn(p):
          dist=apply_fn({"params":{"modules_actor":p}},obs,name="actor",train=False)
          pred=dist.mode();return jnp.mean(jnp.square(pred[:,:6]-targets[:,:6]))
        loss,grads=jax.value_and_grad(loss_fn)(params);updates,new_opt=tx.update(grads,opt_state,params);new_params=optax.apply_updates(params,updates)
        norm=optax.global_norm(grads);return new_params,new_opt,loss,norm
      checkpoints=set(config["actor_only_bc"]["checkpoints"])
      for step in range(1,int(config["actor_only_bc"]["updates"])+1):
        idx=rng.integers(0,len(train),size=int(config["actor_only_bc"]["batch_size"]));actor_params,opt_state,loss,gnorm=bc_step(actor_params,opt_state,jnp.asarray(train_obs[idx]),jnp.asarray(train_act[idx]))
        if step in checkpoints:
          params=dict(agent.state.params);params["modules_actor"]=actor_params;candidate=agent.replace(state=agent.state.replace(params=params))
          report=metrics(candidate,valid_obs,valid_act);first=np.asarray(jax.device_get(candidate.sample_actions(observations=jax.device_put(first_obs),argmax=True)),np.float32)
          report.update(step=step,loss=float(loss),gradient_norm=float(gnorm),first_action=first,
                        first_action_l2_delta=float(np.linalg.norm(first[:6]-expert_first[:6])))
          reports.append(report)
          blob=flax.serialization.to_bytes(candidate.state)
          with (OUTPUT/"checkpoints"/f"bc_checkpoint_{step}.msgpack").open("xb") as f:f.write(blob)
      require(all(r["gradient_norm"]>0 and np.isfinite(r["gradient_norm"]) for r in reports),"active finite actor gradients")
      best_report=min(reports,key=lambda x:(x["mae_a0_a5"],x["first_action_l2_delta"]));best_step=best_report["step"]
      blob=(OUTPUT/"checkpoints"/f"bc_checkpoint_{best_step}.msgpack").read_bytes();template=agent.state
      best_state=flax.serialization.from_bytes(template,blob);best_agent=agent.replace(state=best_state)
      offline_pass=(best_report["mae_a0_a5"]<=.8*old_metrics["mae_a0_a5"] and best_report["first_action_l2_delta"]<=.8*old_delta)
      result={"ACTOR_ONLY_BC":"PASS" if offline_pass else "SCIENTIFIC_FAIL","BEST_BC_CHECKPOINT":str(OUTPUT/"checkpoints"/f"bc_checkpoint_{best_step}.msgpack"),
        "OLD_C0_FIRST_ACTION":old_first,"REPAIRED_C0_FIRST_ACTION":best_report["first_action"],"EXPERT_C0_FIRST_ACTION":expert_first,
        "OLD_ACTION_L2_DELTA":old_delta,"REPAIRED_ACTION_L2_DELTA":best_report["first_action_l2_delta"],
        "LOCAL_VALIDATION_ACTION_MAE":best_report["mae_a0_a5"],"LOCAL_VALIDATION_ACTION_COSINE":best_report["cosine_a0_a5"],
        "old_validation":old_metrics,"checkpoint_reports":reports,"actor_gradients_active":True,"supervised_dimensions":[0,1,2,3,4,5],"a6_supervised":False}
      hashes_after={"actor":sha_tree(best_agent.state.params["modules_actor"]),"critic":sha_tree(best_agent.state.params["modules_critic"]),
        "temperature":sha_tree(best_agent.state.params["modules_temperature"]),"target":sha_tree(best_agent.state.target_params)}
      result["frozen_networks_exact"]=(hashes_before["critic"]==hashes_after["critic"] and hashes_before["temperature"]==hashes_after["temperature"] and hashes_before["target"]==hashes_after["target"])
      require(result["frozen_networks_exact"] and hashes_before["actor"]!=hashes_after["actor"],"actor-only parameter change")
      if not offline_pass:
        result.update(LOCAL_INSERTION_POLICY_SUPERVISION="SCIENTIFIC_FAIL",LOCAL_BC_CATASTROPHIC_FORGETTING="NOT_RUN",
          ACTOR_OPTIMIZER_HANDOFF="NOT_RUN",LOCAL_BC_RETENTION="NOT_RUN",PRIMARY_ROOT_CAUSE="LOCAL_ACTOR_SUPERVISION_INEFFECTIVE",NEXT="STOP")
      else:
        c0=evaluate(best_agent,"C0",demo_obs,scale);result["c0_post_bc"]=c0
        c0_pass=(c0["success_count"]>=2 and c0["near_hole_count"]==3 and c0["contact_health"] and not c0["table_ee_contact"] and
                 c0["best_success_margin"]<.8*HIST_MARGIN and c0["median_first_divergence"]>3)
        result["LOCAL_INSERTION_POLICY_SUPERVISION"]="PASS" if c0_pass else "SCIENTIFIC_FAIL"
        if not c0_pass:
          result.update(LOCAL_BC_CATASTROPHIC_FORGETTING="NOT_RUN",ACTOR_OPTIMIZER_HANDOFF="NOT_RUN",LOCAL_BC_RETENTION="NOT_RUN",
            PRIMARY_ROOT_CAUSE="LOCAL_ACTOR_SUPERVISION_DID_NOT_CLOSE_C0",NEXT="STOP")
        else:
          old_canonical=evaluate(agent,"C3",demo_obs,scale);new_canonical=evaluate(best_agent,"C3",demo_obs,scale)
          catastrophic=(new_canonical["median_approach_reduction"]<.5*old_canonical["median_approach_reduction"] or
                        new_canonical["median_min_position"]>old_canonical["median_min_position"]+.05 or not np.isfinite(new_canonical["max_force"]))
          result["canonical_old"]=old_canonical;result["canonical_repaired"]=new_canonical
          result["LOCAL_BC_CATASTROPHIC_FORGETTING"]="YES" if catastrophic else "NO"
          if catastrophic:
            result.update(ACTOR_OPTIMIZER_HANDOFF="NOT_RUN",LOCAL_BC_RETENTION="NOT_RUN",PRIMARY_ROOT_CAUSE="LOCAL_BC_CATASTROPHIC_FORGETTING",NEXT="STOP")
          else:
            # New experiment-local full state.  Only the actor optimizer is
            # reinitialized using its unchanged source transformation.
            opt_states=dict(best_agent.state.opt_states);opt_states["actor"]=best_agent.state.txs["actor"].init(best_agent.state.params)
            repaired_state=best_agent.state.replace(opt_states=opt_states)
            repaired_blob=flax.serialization.to_bytes(repaired_state);repaired_path=OUTPUT/"checkpoints/repaired_c0_rlpd_state.msgpack"
            with repaired_path.open("xb") as f:f.write(repaired_blob)
            result.update(ACTOR_OPTIMIZER_HANDOFF="REINITIALIZED_SAME_HYPERPARAMETERS",REPAIRED_C0_TRAINING_CHECKPOINT=str(repaired_path),
                          LOCAL_BC_RETENTION="READY",PRIMARY_ROOT_CAUSE="LOCAL_ACTOR_MULTI_AXIS_CONTROL_FAILURE_REPAIRED_AT_C0",NEXT="C0_RLPD_RETENTION_PLUS_500")
      result.update(LOCAL_SUPERVISION_DATASET=str(LOCAL),LOCAL_SUPERVISION_EPISODES=11,LOCAL_SUPERVISION_TRANSITIONS=330,
        LOCAL_SUPERVISION_SHA256=LOCAL_SHA,LOCAL_BC_TRAIN_EPISODES=9,LOCAL_BC_VALID_EPISODES=2,
        LOCAL_BC_SPLIT_SHA256=json.loads((LOCAL_DIR/"manifest.json").read_text())["LOCAL_BC_SPLIT_SHA256"],
        ONLINE_ADMISSION_THRESHOLD=THRESHOLD,ONLINE_ADMISSION_THRESHOLD_CHANGED="NO",EVAL_RESET_CANONICAL="YES",EVAL_GOAL_CANONICAL="YES",EVAL_CURRICULUM_ACTIVE="NO",TRAINING_RUNNING=False)
    after=protected(cfg);require(before==after,"protected artifact changed");result["PRODUCTION_DEMO_SHA_UNCHANGED"]="YES";result["PROTECTED_ARTIFACTS_CHANGED"]="NO"
    dump(OUTPUT/"protected_after.json",after);dump(OUTPUT/"result.json",result)
    print(json.dumps(jv(result),indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":main()
