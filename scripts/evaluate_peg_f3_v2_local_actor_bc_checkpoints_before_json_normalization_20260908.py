#!/usr/bin/env python3
"""Read-only C0 selection across checkpoints from the single actor-BC run."""
from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
ROOT=Path("/home/makoto/flexible_line_project")
sys.path[:0]=[str(ROOT),str(ROOT/"external/serl/serl_launcher"),str(ROOT/"external/serl/franka_sim"),str(ROOT/"external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),str(ROOT/"external/packages/peg_insertion/peg_insertion/code/env")]
import flax,jax,numpy as np
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model,load_config,make_train_env
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate,metrics,sha
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

RUN=ROOT/"results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001"
OUT=ROOT/"results/diagnostics/peg/f3_v2_local_actor_bc_checkpoint_eval_20260908_001"
SOURCE=ROOT/"results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002/checkpoints"
LOCAL=ROOT/"results/datasets/peg_v2_local_insertion_supervision_20260908_001/local_insertion_supervision.pkl"
PROD=ROOT/"results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"

def main():
  assert not OUT.exists();OUT.mkdir(parents=True,exist_ok=False)
  base=json.loads((RUN/"result.json").read_text());split=json.loads((LOCAL.parent/"split.json").read_text());valid_set=set(split["valid_episodes"])
  local=pickle.loads(LOCAL.read_bytes());valid=[x for x in local if x["source_episode"] in valid_set]
  vo=np.asarray([x["observations"] for x in valid],np.float32);va=np.asarray([x["actions"] for x in valid],np.float32)
  demos=pickle.loads(PROD.read_bytes());demo_obs,scale=distance_model(demos);cfg=load_config(1)
  with portable_peg_runtime(ROOT):
    env=make_train_env("C0");fresh=make_combined_q_anchor_agent(cfg["seed"],env.observation_space.sample(),env.action_space.sample(),cfg);env.close()
    agent=fresh.replace(state=restore_fresh_state(SOURCE,fresh.state,1000));reports=[]
    for item in base["checkpoint_reports"]:
      step=item["step"]
      if item["mae_a0_a5"]>.8*base["old_validation"]["mae_a0_a5"] or item["first_action_l2_delta"]>.8*base["OLD_ACTION_L2_DELTA"]:continue
      state=flax.serialization.from_bytes(agent.state,(RUN/"checkpoints"/f"bc_checkpoint_{step}.msgpack").read_bytes());candidate=agent.replace(state=state)
      reports.append({"bc_step":step,"offline":metrics(candidate,vo,va),"c0":evaluate(candidate,"C0",demo_obs,scale)})
  reports.sort(key=lambda x:(-x["c0"]["success_count"],not x["c0"]["contact_health"],x["c0"]["best_success_margin"],-x["c0"]["median_first_divergence"]))
  best=reports[0];passed=best["c0"]["success_count"]>=2 and best["c0"]["contact_health"] and not best["c0"]["table_ee_contact"] and best["c0"]["best_success_margin"]<.8*2.9458398747637045 and best["c0"]["median_first_divergence"]>3
  result={"status":"PASS" if passed else "SCIENTIFIC_FAIL","evaluated_single_run_checkpoints":[x["bc_step"] for x in reports],"ranked":reports,"BEST_BC_CHECKPOINT":str(RUN/"checkpoints"/f"bc_checkpoint_{best['bc_step']}.msgpack"),"best_bc_step":best["bc_step"],"LOCAL_INSERTION_POLICY_SUPERVISION":"PASS" if passed else "SCIENTIFIC_FAIL","TRAINING_STARTED":False,"PRODUCTION_DEMO_SHA_UNCHANGED":"YES" if sha(PROD)=="4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878" else "NO"}
  with (OUT/"result.json").open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
  print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
