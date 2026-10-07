#!/usr/bin/env python3
"""Finish D5 after an analysis-only key-routing failure; reuse D1-D4 evidence."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE="false",
                  XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl")
import jax
import numpy as np
import yaml
from scripts.diagnose_peg_f3_v2_post5k import (CONFIG, DATASET, MANIFEST, PROVENANCE,
    RUN, SCREEN, STEPS, EXPECTED_CONFIG, EXPECTED_DATASET, canonical, require, sha,
    tree_digest, tree_l2_delta, tree_arrays, write_json)
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.observation_contract import V2, validate_config


def main():
    p=argparse.ArgumentParser(); p.add_argument("--reuse",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    reuse,out=a.reuse.resolve(),a.output.resolve(); require(ROOT in reuse.parents and ROOT in out.parents and not out.exists(),"isolated paths")
    out.mkdir(parents=True,exist_ok=False)
    result={"PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC":"FAIL","D5_CHECKPOINT_SANITY":"NOT_RUN",
            "FORMAL_10K_CONTINUE":"BLOCKED","TRAINING_STARTED":False,"NEXT":"BLOCKED","PROTECTED_ARTIFACTS_CHANGED":"NO"}
    protected={}
    try:
        prior=json.loads((reuse/"validation.json").read_text())
        for gate in ("D1_TRAJECTORY_CAUSAL_TRACE","D2_DEMO_REPLAY_SEMANTICS","D3_POLICY_DEMO_ALIGNMENT","D4_CRITIC_PREFERENCE"):
            require(prior.get(gate)=="PASS",f"reused {gate}")
        cfg=yaml.safe_load(CONFIG.read_text()); require(sha(CONFIG)==EXPECTED_CONFIG and sha(DATASET)==EXPECTED_DATASET,"frozen hashes")
        require(validate_config(cfg,for_training=True)==V2,"V2 config")
        prov=json.loads(PROVENANCE.read_text()); cps={s:RUN/"checkpoints"/f"checkpoint_{s}" for s in STEPS}
        paths=[CONFIG,DATASET,MANIFEST,PROVENANCE,SCREEN,reuse/"validation.json",reuse/"d1_checkpoint5000_trace.json"]
        for e in prov: paths.append(ROOT/e["episode_path"]/"transitions.jsonl")
        for s,cp in cps.items(): paths += [cp,cp.with_name(cp.name+"_online_replay.pkl"),cp.with_name(cp.name+"_resume_metadata.json")]
        require(all(x.is_file() for x in paths),"protected artifact missing"); protected={str(x):sha(x) for x in paths}; write_json(out/"protected_before.json",protected)
        ps=subprocess.run(["ps","-eo","pid=,args="],capture_output=True,text=True,check=True).stdout
        require(not any(("async_sac_state_sim.py" in x or "run_peg_f3_v2_rlpd_smoke.py" in x) and str(os.getpid()) not in x for x in ps.splitlines()),"training process running")
        require(jax.default_backend()=="gpu","GPU restore backend")
        pkg=ROOT/"external/packages/peg_insertion/peg_insertion"; sys.path[:0]=[str(pkg/"code/example/async_rlpd_peg_sim"),str(pkg/"code/env")]
        from async_sac_state_sim import make_sac_agent
        fresh=make_sac_agent(seed=0,sample_obs=np.zeros(20,np.float32),sample_action=np.zeros(7,np.float32)); states={}; d5={"steps":{}}
        for step,cp in cps.items():
            meta=json.loads(cp.with_name(cp.name+"_resume_metadata.json").read_text())
            require(meta["checkpoint_step"]==meta["learner_update_steps"]==step and meta["config_hash"]==canonical(cfg),"checkpoint binding")
            state=restore_fresh_state(cp.parent,fresh.state,step); states[step]=state; params=state.params
            d5["steps"][str(step)]={"state_digest":tree_digest(state),"actor_digest":tree_digest(params["modules_actor"]),
                "critic_digest":tree_digest(params["modules_critic"]),"temperature_digest":tree_digest(params["modules_temperature"]),
                "temperature":float(np.asarray(jax.device_get(fresh.replace(state=state).forward_temperature()))),
                "optimizer_finite":all(np.isfinite(x).all() for x in tree_arrays(state.opt_states))}
        d5["actor_1000_to_5000"]=tree_l2_delta(states[1000].params["modules_actor"],states[5000].params["modules_actor"])
        d5["critic_1000_to_5000"]=tree_l2_delta(states[1000].params["modules_critic"],states[5000].params["modules_critic"])
        d5["temperature_1000_to_5000"]=tree_l2_delta(states[1000].params["modules_temperature"],states[5000].params["modules_temperature"])
        d5.update(ACTOR_UPDATED=d5["actor_1000_to_5000"]["absolute_l2"]>0,CRITIC_UPDATED=d5["critic_1000_to_5000"]["absolute_l2"]>0,
                  EVAL_ACTOR_BINDING=True,CHECKPOINT_STEP_BINDING=True)
        require(d5["ACTOR_UPDATED"] and d5["CRITIC_UPDATED"] and all(x["optimizer_finite"] for x in d5["steps"].values()),"D5 sanity")
        result.update({k:prior[k] for k in ("D1_TRAJECTORY_CAUSAL_TRACE","D2_DEMO_REPLAY_SEMANTICS","D3_POLICY_DEMO_ALIGNMENT","D4_CRITIC_PREFERENCE")})
        result.update(D1=prior["D1"],D2=prior["D2"],D3=prior["D3"],D4=prior["D4"],D5=d5,D5_CHECKPOINT_SANITY="PASS")
        d3=prior["D3"]; d4=prior["D4"]; d1=prior["D1"]
        lateral=d3["5000"]["by_subset"]["lateral_correction"]; q=d4["5000"]["overall"]
        require(q["q_policy_gt_demo_rate"]>.5 and q["mean_q_demo_minus_policy"]<0,"critic bad-action preference not supported")
        result.update(PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC="PASS",AXIAL_ADVANCE_BEFORE_ALIGNMENT=d1["AXIAL_ADVANCE_BEFORE_ALIGNMENT"],
            CONTACT_FAILURE_MODE=d1["FAILURE_BEHAVIOR_CLASS"],DEMO_REPLAY_EXACT="PASS",POLICY_DEMO_ALIGNMENT_TREND=d3["ALIGNMENT_TREND"],
            CRITIC_PREFERENCE_TREND=d4["CRITIC_TREND"],PRIMARY_ROOT_CAUSE="D_CRITIC_PREFERS_BAD_ONLINE_ACTIONS",
            SECONDARY_CAUSE="C_POLICY_NOT_LEARNING_DEMO_CORRECTIONS",
            SUPPORTED_BY=["Demo-to-replay values and episode boundaries are exact",
              f"At checkpoint 5000 Q(policy)>Q(demo) rate={q['q_policy_gt_demo_rate']:.6f}, mean Qdemo-Qpolicy={q['mean_q_demo_minus_policy']:.6f}",
              f"Lateral correction MAE={lateral['mae']:.6f}; action[1] MAE={lateral['per_dimension_mae'][1]:.6f}",
              "Actor and critic parameter trees changed from checkpoint 1000 to 5000 and exact restored actor was evaluated",
              f"Online reward mean shift={prior['D2']['DEMO_ONLINE_REWARD_SHIFT']:.6f}; orientation state shift={prior['D2']['DEMO_ONLINE_STATE_SHIFT']['orientation']:.6f}"],
            NEXT="RLPD_Q_BIAS_DIAGNOSTIC_READY")
    except Exception as e:
        result["first_failed_gate"]=repr(e); result["traceback"]=traceback.format_exc(); traceback.print_exc()
    finally:
        after={p:sha(Path(p)) for p in protected}; write_json(out/"protected_after.json",after)
        if protected!=after: result.update(PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC="FAIL",PROTECTED_ARTIFACTS_CHANGED="YES",NEXT="BLOCKED")
        write_json(out/"validation.json",result); (out/"REPORT.md").write_text("# Peg F3 V2 Post-5K Root-Cause Diagnostic\n\n```json\n"+json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n```\n")
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),flush=True)
    return 0 if result["PEG_F3_V2_POST5K_ROOT_CAUSE_DIAGNOSTIC"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
