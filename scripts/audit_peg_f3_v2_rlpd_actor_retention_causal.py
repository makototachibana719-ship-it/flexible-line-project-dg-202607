#!/usr/bin/env python3
"""Read-only causal audit plus a fresh 100-update microscopic RLPD replay."""
from __future__ import annotations
import hashlib, json, math, os, pickle, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_FLAGS","--xla_gpu_enable_triton_gemm=false")
ROOT=Path("/home/makoto/flexible_line_project")
sys.path[:0]=[str(ROOT),str(ROOT/"external/serl/serl_launcher"),str(ROOT/"external/serl/franka_sim"),str(ROOT/"external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),str(ROOT/"external/packages/peg_insertion/peg_insertion/code/env")]
import flax, jax, jax.numpy as jnp, numpy as np
from async_sac_state_sim import concat_rlpd_batches
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import action,block,demo_distance,distance_model,finite_scalars,fixed_batch,load_config,make_train_env
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay

DEMO=ROOT/"results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA="4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD=7.41773975814
SOURCE=ROOT/"results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002/checkpoints"
POSTBC=ROOT/"results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001/checkpoints/bc_checkpoint_1000.msgpack"
ROUND1=ROOT/"results/closure/peg/f3_v2_on_policy_corrective_20260908_001/round1_checkpoints/checkpoint_500.msgpack"
HANDOFF=ROOT/"results/closure/peg/f3_v2_on_policy_corrective_20260908_001/canonical_safety_and_retention/repaired_handoff"
CORR=ROOT/"results/datasets/peg_v2_corrective_postbc_round1_20260908_001/corrective_round1.pkl"
LOCAL=ROOT/"results/datasets/peg_v2_local_insertion_supervision_20260908_001/local_insertion_supervision.pkl"
TRACE=ROOT/"results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
OUT=ROOT/"results/diagnostics/peg/f3_v2_rlpd_actor_retention_causal_20260908_001"
MICRO=(0,1,2,5,10,25,50,100)

def req(x,m):
 if not x: raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8388608),b""):h.update(b)
 return h.hexdigest()
def jv(x):
 if isinstance(x,np.ndarray):return jv(x.tolist())
 if isinstance(x,(np.bool_,bool)):return bool(x)
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.floating):return float(x)
 if isinstance(x,dict):return {str(k):jv(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jv(v) for v in x]
 return x
def dump(p,x):
 with Path(p).open("x") as f:json.dump(jv(x),f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def tree_norm(t):return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(jax.device_get(x)),dtype=np.float64))) for x in jax.tree_util.tree_leaves(t) if hasattr(x,"dtype"))))
def tree_delta(a,b):return tree_norm(jax.tree_util.tree_map(lambda x,y:x-y,a,b))
def tree_hash(t):
 h=hashlib.sha256()
 for p,x in jax.tree_util.tree_flatten_with_path(t)[0]:h.update(str(p).encode());h.update(np.asarray(jax.device_get(x)).tobytes())
 return h.hexdigest()
def modes(agent,obs):return np.asarray(jax.device_get(agent.sample_actions(observations=jax.device_put(obs),argmax=True)),np.float32)
def actor_stats(agent,obs,expert,base=None):
 pred=modes(agent,obs); d=pred[:,:6]-expert[:,:6]
 dist=agent.forward_policy(jax.device_put(obs),rng=jax.random.PRNGKey(991),train=False)
 logstd=np.log(np.asarray(jax.device_get(dist.distribution.scale_diag),np.float64))
 out={"mae":float(np.mean(np.abs(d))),"mean_l2":float(np.mean(np.linalg.norm(d,axis=1))),"log_std_mean":float(np.mean(logstd)),"log_std_std":float(np.std(logstd))}
 if base is not None:out["mode_delta_from_repaired"]=float(np.mean(np.linalg.norm(pred[:,:6]-base[:,:6],axis=1)))
 return out
def opt_summary(opt):
 rows=[];mu2=nu2=0.;counts=[];layers={}
 for path,x in jax.tree_util.tree_flatten_with_path(opt)[0]:
  name=str(path);a=np.asarray(jax.device_get(x));n=float(np.linalg.norm(a.astype(np.float64)))
  if "mu" in name:mu2+=n*n
  if "nu" in name:nu2+=n*n
  if "count" in name and a.size==1:counts.append(int(a.reshape(())))
  if ("mu" in name or "nu" in name) and "modules_actor" in name:layers[name]=n
  rows.append({"path":name,"shape":list(a.shape),"norm":n})
 return {"sha256":tree_hash(opt),"adam_step_counters":counts,"m_global_norm":math.sqrt(mu2),"v_global_norm":math.sqrt(nu2),"actor_layer_moment_norms":layers,"leaves":rows}
def make_agents(cfg):
 env=make_train_env("C0")
 try:fresh=make_combined_q_anchor_agent(cfg["seed"],env.observation_space.sample(),env.action_space.sample(),cfg)
 finally:env.close()
 pre=fresh.replace(state=restore_fresh_state(SOURCE,fresh.state,1000))
 post=pre.replace(state=flax.serialization.from_bytes(pre.state,POSTBC.read_bytes()))
 rep=pre.replace(state=flax.serialization.from_bytes(pre.state,ROUND1.read_bytes()))
 hand=fresh.replace(state=restore_fresh_state(HANDOFF,fresh.state,1000))
 return fresh,pre,post,rep,hand
def c0_initial_expert():
 env=make_train_env("C0")
 try:o,_=env.reset(seed=100)
 finally:env.close()
 tr=[json.loads(x) for x in TRACE.read_text().splitlines() if x.strip()]
 return np.asarray(o,np.float32)[None],np.asarray(tr[137]["action"],np.float32)[None]
def q_stats(agent,obs,expert):
 pa=modes(agent,obs);qp=np.asarray(jax.device_get(agent.forward_critic(jax.device_put(obs),jax.device_put(pa),rng=None,train=False)))
 qe=np.asarray(jax.device_get(agent.forward_critic(jax.device_put(obs),jax.device_put(expert),rng=None,train=False)))
 return {"policy_q_mean":float(qp.mean()),"expert_q_mean":float(qe.mean()),"expert_minus_policy":float((qe-qp).mean())}
def nearest(obs,ref,scale):
 vals=[]
 for s in range(0,len(obs),64):vals.append(np.min(np.sqrt(np.sum(np.square((obs[s:s+64,None]-ref[None])/scale),axis=2)),axis=1))
 return np.concatenate(vals)
def checkpoint_report(agent,repaired_actor,anchors,demo_obs,scale,losses):
 rep={"actor_param_delta":tree_delta(agent.state.params["modules_actor"],repaired_actor),"actor_param_relative_delta":tree_delta(agent.state.params["modules_actor"],repaired_actor)/tree_norm(repaired_actor),"temperature":float(np.asarray(jax.device_get(agent.forward_temperature()))) if np.asarray(jax.device_get(agent.forward_temperature())).size==1 else tree_norm(agent.state.params["modules_temperature"]),"losses":losses}
 for k,(o,a,b) in anchors.items():rep[k]=actor_stats(agent,o,a,b)
 rep["corrective_q"]=q_stats(agent,*anchors["corrective"][:2]);return rep
def evaluate_micro(agent,demo_obs,scale):return evaluate(agent,"C0",demo_obs,scale)
def counterfactual(agent,batch,anchors):
 actual,_=agent.update(batch,networks_to_update=frozenset({"actor"}))
 opts=dict(agent.state.opt_states);opts["actor"]=agent.state.txs["actor"].init(agent.state.params)
 fresh=agent.replace(state=agent.state.replace(opt_states=opts));fresh_after,_=fresh.update(batch,networks_to_update=frozenset({"actor"}))
 def one(after):
  out={"PARAM_DELTA_NORM":tree_delta(after.state.params["modules_actor"],agent.state.params["modules_actor"]),"PARAM_RELATIVE_DELTA":tree_delta(after.state.params["modules_actor"],agent.state.params["modules_actor"])/tree_norm(agent.state.params["modules_actor"])}
  for k,(o,a,b) in anchors.items():out[k]={"before":actor_stats(agent,o,a),"after":actor_stats(after,o,a),"ACTION_MODE_DELTA":float(np.mean(np.linalg.norm(modes(after,o)[:,:6]-modes(agent,o)[:,:6],axis=1)))}
  return out
 return {"actual":one(actual),"fresh":one(fresh_after),"post_actor_param_delta":tree_delta(actual.state.params["modules_actor"],fresh_after.state.params["modules_actor"])}
def components(agent,batch,anchors):
 obs=batch["observations"]
 def losses(ap,rng):
  full=dict(agent.state.params);full["modules_actor"]=ap
  _,pr,sr,cr=jax.random.split(rng,4);dist=agent.forward_policy(obs,rng=pr,grad_params=full);acts,lp=dist.sample_and_log_prob(seed=sr)
  qs=agent.forward_critic(obs,acts,rng=cr);temp=agent.forward_temperature()
  q=-jnp.mean(qs.mean(axis=0));ent=jnp.mean(temp*lp)
  demo_n=int(agent.config["demo_anchor_batch_size"]);dd=agent.forward_policy(obs[-demo_n:],rng=rng,grad_params=full).mode();other=float(agent.config["demo_anchor_lambda"])*jnp.mean(jnp.square(dd-batch["actions"][-demo_n:]))
  return q,ent,other
 ap=agent.state.params["modules_actor"];rng=jax.random.PRNGKey(4401);names=("Q","ENTROPY","OTHER");out={};p_norm=tree_norm(ap)
 for i,name in enumerate(names):
  fn=lambda p:losses(p,rng)[i];val,g=jax.value_and_grad(fn)(ap);gn=tree_norm(g);step=1e-5*p_norm/max(gn,1e-30);nap=jax.tree_util.tree_map(lambda p,x:p-step*x,ap,g);full=dict(agent.state.params);full["modules_actor"]=nap;after=agent.replace(state=agent.state.replace(params=full))
  effects={}
  labels=[]
  for k,(o,a,_) in anchors.items():
   before=actor_stats(agent,o,a)["mae"];aft=actor_stats(after,o,a)["mae"];delta=aft-before;effects[k]=delta;labels.append("TOWARD" if delta < -1e-10 else "AWAY" if delta > 1e-10 else "NEUTRAL")
  label=labels[0] if len(set(labels))==1 else "MIXED"
  out[name]={"loss":float(val),"gradient_norm":gn,"diagnostic_param_step_norm":1e-5*p_norm,"anchor_mae_delta":effects,"alignment":label}
 return out
def critic_geometry(agent,obs,expert):
 pa=jax.device_put(modes(agent,obs));oo=jax.device_put(obs);ee=np.asarray(expert)
 def each(o,a):return agent.forward_critic(o[None],a[None],rng=None,train=False)[:,0]
 jac=np.asarray(jax.device_get(jax.vmap(jax.jacrev(each,argnums=1))(oo,pa)))[:,:,:6];d=ee[:,:6]-np.asarray(pa)[:,:6]
 den=np.linalg.norm(jac,axis=2)*np.linalg.norm(d,axis=1)[:,None];cos=np.divide(np.sum(jac*d[:,None,:],axis=2),den,out=np.zeros_like(den),where=den>1e-12)
 agg=jac.mean(axis=1);aden=np.linalg.norm(agg,axis=1)*np.linalg.norm(d,axis=1);acos=np.divide(np.sum(agg*d,axis=1),aden,out=np.zeros_like(aden),where=aden>1e-12)
 ts=(0.,.05,.1,.25,.5,1.);scan={}
 for t in ts:
  a=np.asarray(pa).copy();a[:,:6]+=t*d;q=np.asarray(jax.device_get(agent.forward_critic(oo,jax.device_put(a),rng=None,train=False)));scan[str(t)]=float(q.mean())
 med=float(np.median(acos));label="ALIGNED" if med>.2 else "ANTI_ALIGNED" if med<-.2 else "ORTHOGONAL" if np.all(np.abs(acos)<=.2) else "MIXED"
 return {"per_head_cosine":cos,"ensemble_aggregate_cosine":acos,"median_aggregate_cosine":med,"label":label,"directional_q_scan":scan}
def main():
 req(Path.cwd()==ROOT,"cwd");req(not OUT.exists(),"fresh output");req(jax.default_backend()=="gpu","GPU");req(sha(DEMO)==DEMO_SHA,"Demo")
 OUT.mkdir(parents=True);(OUT/"checkpoints").mkdir();started=time.time();cfg=load_config(1);demos=pickle.loads(DEMO.read_bytes());demo_obs,scale=distance_model(demos)
 corr=pickle.loads(CORR.read_bytes());local=pickle.loads(LOCAL.read_bytes());co=np.asarray([x["observations"] for x in corr],np.float32);ca=np.asarray([x["actions"] for x in corr],np.float32);lo=np.asarray([x["observations"] for x in local],np.float32);la=np.asarray([x["actions"] for x in local],np.float32);io,ia=c0_initial_expert()
 all_demo_o=np.asarray([x["observations"] for x in demos],np.float32);all_demo_a=np.asarray([x["actions"] for x in demos],np.float32);approach_idx=np.where(np.linalg.norm(all_demo_o[:,13:16],axis=1)>.05)[0][:128];global_idx=np.linspace(0,len(demos)-1,128,dtype=int)
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  fresh,pre,post,rep,agent=make_agents(cfg);req(tree_delta(rep.state.params["modules_actor"],agent.state.params["modules_actor"])==0,"handoff actor")
  anchors={"c0_initial":(io,ia,modes(agent,io)),"corrective":(co,ca,modes(agent,co)),"local":(lo,la,modes(agent,lo)),"approach":(all_demo_o[approach_idx],all_demo_a[approach_idx],modes(agent,all_demo_o[approach_idx])),"global_demo":(all_demo_o[global_idx],all_demo_a[global_idx],modes(agent,all_demo_o[global_idx]))}
  hashes={"ACTOR_SHA":tree_hash(agent.state.params["modules_actor"]),"CRITIC_SHA":tree_hash(agent.state.params["modules_critic"]),"TARGET_CRITIC_SHA":tree_hash(agent.state.target_params["modules_critic"]),"TEMPERATURE_SHA":tree_hash(agent.state.params["modules_temperature"])}
  forensic={"pre_c0":opt_summary(pre.state.opt_states["actor"]),"post_bc_embedded":opt_summary(post.state.opt_states["actor"]),"round1_embedded":opt_summary(rep.state.opt_states["actor"]),"retention_start":opt_summary(agent.state.opt_states["actor"]),"fresh_same_hyperparameters":opt_summary(agent.state.txs["actor"].init(agent.state.params)),"actor_parameter_global_norm":tree_norm(agent.state.params["modules_actor"])}
  dump(OUT/"optimizer_forensics.json",forensic)
  env=make_train_env("C0")
  try:
   online=make_replay_buffer(env,capacity=cfg["training"]["replay_capacity"],type="replay_buffer");restore_replay(online,HANDOFF/"checkpoint_1000_online_replay.pkl");demo=make_replay_buffer(env,capacity=len(demos)+100,type="replay_buffer")
   for x in demos:demo.insert(x)
   obs,ri=env.reset(seed=cfg["seed"]);sched=ri["curriculum_schedule_sha256"];arng=jax.random.PRNGKey(11000);attempt=admit=0
   while admit<60:
    act,arng=action(agent,obs,deterministic=False,rng=arng);nxt,r,term,trunc,info=env.step(act);dist=demo_distance(nxt,demo_obs,scale);attempt+=1
    if dist<=THRESHOLD:
     online.insert({"observations":np.asarray(obs,np.float32),"actions":act,"next_observations":np.asarray(nxt,np.float32),"rewards":np.float32(r),"masks":np.float32(0 if term else 1),"dones":bool(term or trunc)});admit+=1;obs=nxt
    if term or trunc or dist>THRESHOLD:obs,ri=env.reset(seed=cfg["seed"]);req(ri["curriculum_schedule_sha256"]==sched,"schedule")
  finally:env.close()
  rng=np.random.default_rng(20261908);batches={};coverage=[];reports={};states={};losslog={};repaired_actor=agent.state.params["modules_actor"]
  states[0]=flax.serialization.to_bytes(agent.state);(OUT/"checkpoints/checkpoint_0.msgpack").write_bytes(states[0]);reports["0"]=checkpoint_report(agent,repaired_actor,anchors,demo_obs,scale,{})
  for u in range(1,101):
   oi=rng.integers(0,len(online),32);di=rng.integers(0,len(demo),32);batch=concat_rlpd_batches(fixed_batch(online,oi),fixed_batch(demo,di));batches[u]=batch
   bo=np.asarray(jax.device_get(batch["observations"]));dc=nearest(bo,np.concatenate([co,lo]),scale);dp=nearest(bo,demo_obs,scale);near=(np.linalg.norm(bo[:,14:16],axis=1)<=.01)&(np.abs(bo[:,13])<=.02)
   coverage.append({"update":u,"production_demo_fraction":.5,"online_fraction":.5,"near_hole_fraction":float(near.mean()),"local_corrective_support_fraction":float((dc<=THRESHOLD).mean()),"post_divergence_fraction":float((dp>THRESHOLD).mean()),"nearest_local_corrective_mean":float(dc.mean()),"nearest_local_corrective_median":float(np.median(dc))})
   agent,info=agent.update_high_utd(batch,utd_ratio=1);block((agent,info));losslog[u]=finite_scalars(info)
   if u in MICRO:
    states[u]=flax.serialization.to_bytes(agent.state);(OUT/f"checkpoints/checkpoint_{u}.msgpack").write_bytes(states[u]);reports[str(u)]=checkpoint_report(agent,repaired_actor,anchors,demo_obs,scale,losslog[u])
  for u in MICRO:
   st=flax.serialization.from_bytes(agent.state,states[u]);a=agent.replace(state=st);reports[str(u)]["c0"]=evaluate_micro(a,demo_obs,scale)
  first=next((u for u in MICRO if u>0 and reports[str(u)]["c0"]["success_count"]<2),None);req(first is not None,"no skill loss through 100")
  start_agent=agent.replace(state=flax.serialization.from_bytes(agent.state,states[0]));counter=counterfactual(start_agent,batches[1],anchors)
  pre_loss=max(u for u in MICRO if u<first);around=sorted(set([0,pre_loss,first]));decomp={}
  for u in around:
   aa=agent.replace(state=flax.serialization.from_bytes(agent.state,states[u]));decomp[str(u)]=components(aa,batches[max(1,u)],anchors)
  geom=critic_geometry(start_agent,co[:64],ca[:64])
  before_u=pre_loss;ba=agent.replace(state=flax.serialization.from_bytes(agent.state,states[before_u]));aa=agent.replace(state=flax.serialization.from_bytes(agent.state,states[first]))
  interference={}
  for k in ("local","approach","global_demo"):
   o,a,_=anchors[k];bp=modes(ba,o);ap=modes(aa,o);interference[k]={"action_change":float(np.mean(np.linalg.norm(ap[:,:6]-bp[:,:6],axis=1))),"expert_mae_before":float(np.mean(np.abs(bp[:,:6]-a[:,:6]))),"expert_mae_after":float(np.mean(np.abs(ap[:,:6]-a[:,:6])))}
  dump(OUT/"microscopic_reports.json",reports);dump(OUT/"coverage.json",coverage);dump(OUT/"counterfactual.json",counter);dump(OUT/"loss_decomposition.json",decomp);dump(OUT/"critic_geometry.json",geom);dump(OUT/"parameter_interference.json",interference)
  cavg={k:float(np.mean([x[k] for x in coverage])) for k in coverage[0] if k!="update"}
  actual_damage=counter["actual"]["corrective"]["ACTION_MODE_DELTA"];fresh_damage=counter["fresh"]["corrective"]["ACTION_MODE_DELTA"];shock=actual_damage>2*fresh_damage+1e-8
  qalign=decomp[str(pre_loss)]["Q"]["alignment"];ealign=decomp[str(pre_loss)]["ENTROPY"]["alignment"];oalign=decomp[str(pre_loss)]["OTHER"]["alignment"]
  local_delta=interference["local"]["expert_mae_after"]-interference["local"]["expert_mae_before"];gen_improve=min(interference["approach"]["expert_mae_after"]-interference["approach"]["expert_mae_before"],interference["global_demo"]["expert_mae_after"]-interference["global_demo"]["expert_mae_before"])<0
  pi=local_delta>0 and gen_improve
  if shock:root="ACTOR_OPTIMIZER_STATE_SHOCK"
  elif qalign=="AWAY":root="RLPD_Q_GRADIENT_LOCAL_MISALIGNMENT"
  elif ealign=="AWAY" and qalign!="AWAY":root="ENTROPY_POLICY_DISTRIBUTION_DRIFT"
  elif cavg["local_corrective_support_fraction"]<.25:root="CORRECTIVE_STATE_ABSENT_FROM_ACTOR_UPDATE_DISTRIBUTION"
  elif pi:root="SHARED_ACTOR_PARAMETER_INTERFERENCE"
  else:root="MIXED_RLPD_ACTOR_RETENTION_FAILURE"
  result={"RLPD_ACTOR_RETENTION_REVIEW":"PASS","REPAIRED_CHECKPOINT":str(ROUND1),"REPAIRED_CHECKPOINT_SHA256":sha(ROUND1),**hashes,"ACTOR_OPTIMIZER_SOURCE":"FRESH_SAME_HYPERPARAMETERS","OPTIMIZER_STATE_SHOCK":"YES" if shock else "NO","ACTUAL_OPTIMIZER_ACTION_DAMAGE":actual_damage,"FRESH_OPTIMIZER_ACTION_DAMAGE":fresh_damage,"FIRST_SKILL_LOSS_UPDATE":first,"GRAD_NORM_Q":decomp[str(pre_loss)]["Q"]["gradient_norm"],"GRAD_NORM_ENTROPY":decomp[str(pre_loss)]["ENTROPY"]["gradient_norm"],"GRAD_NORM_OTHER":decomp[str(pre_loss)]["OTHER"]["gradient_norm"],"Q_GRADIENT_CORRECTIVE_ALIGNMENT":qalign,"ENTROPY_GRADIENT_CORRECTIVE_ALIGNMENT":ealign,"OTHER_GRADIENT_CORRECTIVE_ALIGNMENT":oalign,"Q_ACTION_GRAD_TO_EXPERT_COSINE":geom,"ACTOR_UPDATE_NEAR_HOLE_FRACTION":cavg["near_hole_fraction"],"ACTOR_UPDATE_CORRECTIVE_SUPPORT_FRACTION":cavg["local_corrective_support_fraction"],"PARAMETER_INTERFERENCE":"YES" if pi else "NO","LOCAL_ACTION_CHANGE":interference["local"]["action_change"],"APPROACH_ACTION_CHANGE":interference["approach"]["action_change"],"GLOBAL_DEMO_ACTION_CHANGE":interference["global_demo"]["action_change"],"PRIMARY_ROOT_CAUSE":root,"RETENTION_CONSTRAINT_AUTHORIZED":root in ("RLPD_Q_GRADIENT_LOCAL_MISALIGNMENT","CORRECTIVE_STATE_ABSENT_FROM_ACTOR_UPDATE_DISTRIBUTION","SHARED_ACTOR_PARAMETER_INTERFERENCE") or (root=="MIXED_RLPD_ACTOR_RETENTION_FAILURE" and not shock and root!="ENTROPY_POLICY_DISTRIBUTION_DRIFT"),"collection_attempts":attempt,"collection_admitted":admit,"protected":{"PRODUCTION_DEMO_SHA_UNCHANGED":"YES","ONLINE_ADMISSION_THRESHOLD":THRESHOLD,"ONLINE_ADMISSION_THRESHOLD_CHANGED":"NO","CORRECTIVE_DATASET_SHA256":sha(CORR),"LOCAL_DATASET_SHA256":sha(LOCAL)},"elapsed_seconds":time.time()-started}
  dump(OUT/"result.json",result);print(json.dumps(jv(result),indent=2,sort_keys=True))
if __name__=="__main__":main()
