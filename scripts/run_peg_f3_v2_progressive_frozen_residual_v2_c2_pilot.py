#!/usr/bin/env python3
"""Stage L: progressive frozen C1 + trainable C2 residual, fixed 250 pilot."""
from __future__ import annotations
import json,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false')
ROOT=Path('/home/makoto/flexible_line_project');sys.path[:0]=[str(ROOT),str(ROOT/'external/serl/serl_launcher'),str(ROOT/'external/serl/franka_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim'),str(ROOT/'external/packages/peg_insertion/peg_insertion/code/env')]
import chex,flax,jax,jax.numpy as jnp,numpy as np,optax
from async_sac_state_sim import concat_rlpd_batches
import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_observation_gated_residual_c1_pilot as old
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import block,distance_model,evaluate,fixed_batch,load_config,make_train_env
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.observation_gated_residual_v1 import RESIDUAL_DEF,combined_mode,gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_distribution,progressive_gate,progressive_mode,support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay,snapshot_replay
MATCH=ROOT/'results/training/peg_f3_v2_observation_gated_residual_c1_matched_20260908_001';CP=MATCH/'checkpoints/c1_matched_learner_1000_residual_250.msgpack';RP=MATCH/'online_replay.pkl';PRE=ROOT/'results/preflight/peg/f3_v2_progressive_frozen_residual_v2_20260908_002';OUT=ROOT/'results/training/peg_f3_v2_progressive_frozen_residual_v2_c2_pilot_20260908_002';C2_TX=make_optimizer(learning_rate=3e-4,warmup_steps=2000);EVALS=(0,50,100,250)
def req(x,m):
 if not x:raise RuntimeError(m)
def init_c1(common):
 r=RESIDUAL_DEF.init(jax.random.PRNGKey(260910),np.zeros((1,20),np.float32))['params'];return {'agent_state':common.state,'residual_params':r,'residual_opt_state':old.RESIDUAL_TX.init(r)}
class Policy:
    def __init__(self,agent,c1p,c2p,support,scale,core,outer,record=False):
        self.agent=agent;self.c1p=c1p;self.c2p=c2p;self.support=support;self.scale=scale;self.core=core;self.outer=outer;self.record=record;self.classes=[];self.gates=[];self.norms=[]
    def sample_actions(self,observations,seed=None,deterministic=False,argmax=False,**_):
        obs=jnp.asarray(observations);dist=progressive_distribution(self.agent,self.c1p,self.c2p,obs,self.support,self.scale,self.core,self.outer,rng=seed,train=False);value=dist.mode() if argmax or deterministic else dist.sample(seed=seed)
        if self.record:
            d=np.asarray(jax.device_get(support_distance(obs,self.support,self.scale))).reshape(-1);g=np.asarray(jax.device_get(progressive_gate(obs,self.support,self.scale,self.core,self.outer))).reshape(-1);g0=np.asarray(jax.device_get(gate_value(obs))).reshape(-1);r=np.asarray(jax.device_get(RESIDUAL_DEF.apply({'params':self.c2p},obs))).reshape((-1,6));self.gates+=g.tolist();self.norms+=np.linalg.norm(r,axis=1).tolist();self.classes += [('C0_CORE' if z==0 else 'C1_CORE' if x<=self.core else 'C1_RECOVERY_SHELL' if x<=self.outer else 'C2_OUTSIDE') for x,z in zip(d,g0)]
        return value
    def stats(self):
        transitions=sum(self.classes[i]!=self.classes[i-1] for i in range(1,len(self.classes)))
        shell=sum(x=='C1_RECOVERY_SHELL' for x in self.classes);core=sum(x=='C1_CORE' for x in self.classes);c0=sum(x=='C0_CORE' for x in self.classes)
        shell_entries=sum((self.classes[i] in ('C1_RECOVERY_SHELL','C1_CORE')) and (i==0 or self.classes[i-1]=='C2_OUTSIDE') for i in range(len(self.classes)))
        core_entries=sum((self.classes[i]=='C1_CORE') and (i==0 or self.classes[i-1] not in ('C1_CORE','C0_CORE')) for i in range(len(self.classes)))
        handoff=0;seen=False
        for x in self.classes:
            if x=='C1_CORE':seen=True
            if seen and x=='C0_CORE':handoff+=1;seen=False
        a=np.asarray(self.gates);n=np.asarray(self.norms);return {'visited_states':len(self.classes),'class_state_counts':{'C2_OUTSIDE':self.classes.count('C2_OUTSIDE'),'C1_RECOVERY_SHELL':shell,'C1_CORE':core,'C0_CORE':c0},'C1_SHELL_ENTRY_COUNT':shell_entries,'C1_CORE_ENTRY_COUNT':core_entries,'C1_HANDOFF_SUCCESS':handoff,'MEAN_GATE':float(a.mean()),'P95_GATE':float(np.quantile(a,.95)),'MEAN_C2_RESIDUAL_NORM':float(n.mean()),'P95_C2_RESIDUAL_NORM':float(np.quantile(n,.95)),'class_transitions':transitions}
def eval_record(agent,c1p,c2p,support,scale,core,outer,level,demo_obs,demo_scale):
 p=Policy(agent,c1p,c2p,support,scale,core,outer,True);r=evaluate(p,level,demo_obs,demo_scale);r['interface']=p.stats();return r
def collect(env,agent,c1p,c2p,support,scale,core,outer,online,cfg,demo_obs,demo_scale,obs,rng,count=60):
 p=Policy(agent,c1p,c2p,support,scale,core,outer,True);admitted=attempts=0;mf=0.
 if obs is None:obs,_=env.reset(seed=cfg['seed'])
 while admitted<count:
  rng,key=jax.random.split(rng);a=np.asarray(jax.device_get(p.sample_actions(jax.device_put(obs),seed=key)),np.float32);n,r,t,tr,info=env.step(a);d=p4.demo_distance(n,demo_obs,demo_scale);attempts+=1;mf=max(mf,float(info.get('force_norm',0.)))
  if d<=p4.TH:online.insert({'observations':np.asarray(obs,np.float32),'actions':a,'next_observations':np.asarray(n,np.float32),'rewards':np.float32(r),'masks':np.float32(0 if t else 1),'dones':bool(t or tr)});admitted+=1;obs=n
  if t or tr or d>p4.TH:obs,_=env.reset(seed=cfg['seed'])
 return obs,rng,{'attempts':attempts,'admitted':admitted,'max_force':mf,'interface':p.stats()}
def critic_loss(agent,c1p,c2p,support,scale,core,outer,batch,params,rng):
 bs=batch['rewards'].shape[0];rng,key=jax.random.split(rng);dist=progressive_distribution(agent,c1p,c2p,batch['next_observations'],support,scale,core,outer,rng=key,train=True);na,nlp=dist.sample_and_log_prob(seed=key);tq=agent.forward_target_critic(batch['next_observations'],na,rng=rng)
 if agent.config['critic_subsample_size'] is not None:rng,sk=jax.random.split(rng);ids=jax.random.randint(sk,(agent.config['critic_subsample_size'],),0,agent.config['critic_ensemble_size']);tq=tq[ids]
 target=batch['rewards']+agent.config['discount']*batch['masks']*tq.min(0)
 if agent.config['backup_entropy']:target=target-agent.forward_temperature()*nlp
 pred=agent.forward_critic(batch['observations'],batch['actions'],rng=rng,grad_params=params);loss=jnp.mean((pred-target[None].repeat(agent.config['critic_ensemble_size'],0))**2);return loss,{'critic_loss':loss,'predicted_qs':pred.mean(),'target_qs':target.mean()}
@jax.jit
def critic_update(agent,c1p,c2p,support,scale,core,outer,batch):
 keys=jax.random.split(agent.state.rng,4);nr=jax.random.split(agent.state.rng)[0];(_,info),grads=jax.value_and_grad(lambda p:critic_loss(agent,c1p,c2p,support,scale,core,outer,batch,p,keys[2]),has_aux=True)(agent.state.params);up,no=agent.state.txs['critic'].update(grads,agent.state.opt_states['critic'],agent.state.params);params=optax.apply_updates(agent.state.params,up);opts=dict(agent.state.opt_states);opts['critic']=no;tau=agent.config['soft_target_update_rate'];target=jax.tree_util.tree_map(lambda p,t:tau*p+(1-tau)*t,params,agent.state.target_params);target=dict(target);target['modules_actor']=agent.state.target_params['modules_actor'];return agent.replace(state=agent.state.replace(step=agent.state.step+1,params=params,opt_states=opts,target_params=target,rng=nr)),info
def actor_loss(agent,c1p,c2p,support,scale,core,outer,batch,rng):
 _,pk,sk,ck=jax.random.split(rng,4);dist=progressive_distribution(agent,c1p,c2p,batch['observations'],support,scale,core,outer,rng=pk,train=True);a,lp=dist.sample_and_log_prob(seed=sk);q=agent.forward_critic(batch['observations'],a,rng=ck).mean(0);loss=-jnp.mean(q-agent.forward_temperature()*lp);return loss,{'c2_residual_actor_loss':loss,'entropy':-lp.mean(),'temperature':agent.forward_temperature()}
def temp_loss(agent,c1p,c2p,support,scale,core,outer,batch,params,rng):
 _,sk=jax.random.split(rng);dist=progressive_distribution(agent,c1p,c2p,batch['next_observations'],support,scale,core,outer,rng=sk,train=True);_,lp=dist.sample_and_log_prob(seed=sk);loss=agent.temperature_lagrange_penalty(-lp.mean(),grad_params=params);return loss,{'temperature_loss':loss}
@jax.jit
def c2_update(agent,c1p,c2p,c2opt,support,scale,core,outer,batch):
 keys=jax.random.split(agent.state.rng,4);nr=jax.random.split(agent.state.rng)[0];(_,ai),g=jax.value_and_grad(lambda p:actor_loss(agent,c1p,p,support,scale,core,outer,batch,keys[1]),has_aux=True)(c2p);up,no=C2_TX.update(g,c2opt,c2p);new=optax.apply_updates(c2p,up);(_,ti),tg=jax.value_and_grad(lambda p:temp_loss(agent,c1p,c2p,support,scale,core,outer,batch,p,keys[3]),has_aux=True)(agent.state.params);tu,to=agent.state.txs['temperature'].update(tg,agent.state.opt_states['temperature'],agent.state.params);params=optax.apply_updates(agent.state.params,tu);opts=dict(agent.state.opt_states);opts['temperature']=to;info=dict(ai);info.update(ti);info['c2_gradient_norm']=optax.global_norm(g);info['c2_parameter_delta_norm']=optax.global_norm(up);return agent.replace(state=agent.state.replace(step=agent.state.step+1,params=params,opt_states=opts,rng=nr)),new,no,info
def save(agent,c1p,c1opt,c2p,c2opt,u):
 p=OUT/'checkpoints'/f'c2_learner_{u:04d}_residual_{u//4:03d}.msgpack';v={'agent_state':agent.state,'c1_params':c1p,'c1_opt_state':c1opt,'c2_params':c2p,'c2_opt_state':c2opt};p.write_bytes(flax.serialization.to_bytes(v));return {'path':str(p),'sha256':p4.sha(p),'learner':u,'c2_updates':u//4}
def progress(a,b):
 checks={'success':b['success_count']>a['success_count'],'near_hole':b['near_hole_rate']>a['near_hole_rate'],'first_divergence':b['median_first_divergence']>a['median_first_divergence'],'corridor':b['median_corridor_steps']>a['median_corridor_steps'],'min_position':b['median_min_position']<a['median_min_position']-1e-6,'approach':b['median_approach_reduction']>a['median_approach_reduction']+1e-6};safe=b['contact_health'] and b['median_min_position']<=a['median_min_position']+.010;return bool(any(checks.values()) and safe),checks
def main():
 req(Path.cwd()==ROOT and not OUT.exists() and jax.default_backend()=='gpu','fresh/cwd/GPU');pre=json.loads((PRE/'result.json').read_text());matched=json.loads((MATCH/'result.json').read_text());req(pre['SCIENTIFIC_GATE']=='PASS' and matched['C1_MATCHED_CONFIRMATION']=='PASS' and p4.sha(CP)==matched['FINAL_CHECKPOINT']['sha256'] and p4.sha(RP)==matched['FINAL_REPLAY_SHA256'],'preflight/handoff');OUT.mkdir(parents=True);(OUT/'checkpoints').mkdir();started=time.time();before={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};p4.dump(OUT/'protected_before.json',before);cfg=load_config(2);demos=pickle.loads(p4.DEMO.read_bytes());demo_obs,demo_scale=distance_model(demos);sup=pickle.loads((PRE/'C1_CONFIRMED_SUPPORT_V1.pkl').read_bytes());support=np.asarray(sup['observations'],np.float32);scale=np.asarray(sup['normalization_scale']);core=pre['C1_CORE_THRESHOLD'];outer=pre['C1_SHELL_OUTER_THRESHOLD'];reports={'C2':{},'C1':{},'C0':{}};slots=[];collections=[];checkpoints={}
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  common,_,demo=geo.build_common_start(cfg,demos);loaded=flax.serialization.from_bytes(init_c1(common),CP.read_bytes());agent=common.replace(state=loaded['agent_state']);c1p=loaded['residual_params'];c1opt=loaded['residual_opt_state'];base=geo.thash(agent.state.params['modules_actor']);baseopt=geo.thash(agent.state.opt_states['actor']);basetarget=geo.thash(agent.state.target_params['modules_actor']);c1sha=geo.thash(c1p);c1optsha=geo.thash(c1opt);req(base==pre['C0_BASE_SHA'] and c1sha==pre['C1_RESIDUAL_SHA'],'frozen stack');c2p=RESIDUAL_DEF.init(jax.random.PRNGKey(260920),np.zeros((1,20),np.float32))['params'];c2opt=C2_TX.init(c2p);geo.OUT=OUT;rows,snaps,_,manifest=geo.build_critical(common);sm=json.loads(old.SMANIFEST.read_text());sel=sm['probe_indices'];probe=pickle.loads((PRE/'C1_BEHAVIOR_CRITICAL_PROBE_V1.pkl').read_bytes());pobs=np.asarray([x['observation'] for x in probe],np.float32);pref=np.asarray(jax.device_get(combined_mode(agent,c1p,pobs)),np.float32)
  env=make_train_env('C2');online=make_replay_buffer(env,capacity=cfg['training']['replay_capacity'],type='replay_buffer');ri=restore_replay(online,RP);req(ri['sha256']==matched['FINAL_REPLAY_SHA256'],'replay')
  reports['C2']['0']=eval_record(agent,c1p,c2p,support,scale,core,outer,'C2',demo_obs,demo_scale);reports['C1']['0']=eval_record(agent,c1p,c2p,support,scale,core,outer,'C1',demo_obs,demo_scale);reports['C0']['0']=eval_record(agent,c1p,c2p,support,scale,core,outer,'C0',demo_obs,demo_scale);checkpoints['0']=save(agent,c1p,c1opt,c2p,c2opt,0);arng=jax.random.PRNGKey(19800);rng=np.random.default_rng(20261308);obs=None;bound=len(online)
  for u in range(1,251):
   if (u-1)%100==0:obs,arng,ci=collect(env,agent,c1p,c2p,support,scale,core,outer,online,cfg,demo_obs,demo_scale,obs,arng);ci.update(block=(u-1)//100+1,replay_size=len(online));collections.append(ci);bound=len(online)
   batch=concat_rlpd_batches(fixed_batch(online,rng.integers(0,bound,32)),fixed_batch(demo,rng.integers(0,len(demo),32)));agent,ci=critic_update(agent,c1p,c2p,support,scale,core,outer,batch);block((agent,ci));req(geo.thash(agent.state.params['modules_actor'])==base and geo.thash(agent.state.opt_states['actor'])==baseopt and geo.thash(agent.state.target_params['modules_actor'])==basetarget and geo.thash(c1p)==c1sha and geo.thash(c1opt)==c1optsha,'frozen stack mutation')
   if u%4==0:
    prehash=geo.thash(c2p);agent,c2p,c2opt,info=c2_update(agent,c1p,c2p,c2opt,support,scale,core,outer,batch);block((agent,c2p,c2opt,info));cur=np.asarray(jax.device_get(progressive_mode(agent,c1p,c2p,pobs,support,scale,core,outer)));delta=float(np.max(np.abs(cur-pref)));req(delta<=1e-7,'C1 core invariance');slots.append({'learner':u,'slot':u//4,'params_changed':geo.thash(c2p)!=prehash,'C1_CORE_ACTION_INVARIANCE_MAX_DELTA':delta,'update':{k:float(np.asarray(jax.device_get(v))) for k,v in info.items()}})
   if u in (50,100,250):reports['C2'][str(u)]=eval_record(agent,c1p,c2p,support,scale,core,outer,'C2',demo_obs,demo_scale);p4.dump(OUT/f'c2_eval_{u:04d}.json',reports['C2'][str(u)])
   if u in (100,250):
    reports['C1'][str(u)]=eval_record(agent,c1p,c2p,support,scale,core,outer,'C1',demo_obs,demo_scale);reports['C0'][str(u)]=eval_record(agent,c1p,c2p,support,scale,core,outer,'C0',demo_obs,demo_scale);checkpoints[str(u)]=save(agent,c1p,c1opt,c2p,c2opt,u)
  env.close();p4.dump(OUT/'monitoring.json',reports);p4.dump(OUT/'residual_update_slots.json',slots);p4.dump(OUT/'collections.json',collections);p4.dump(OUT/'checkpoints.json',checkpoints);replay=snapshot_replay(online,OUT/'online_replay.pkl');initial=reports['C2']['0'];final=reports['C2']['250'];prog,checks=progress(initial,final);a=reports['C1']['0'];b=reports['C1']['250'];c1ok=b['contact_health'] and b['median_first_divergence']+1e-8>=a['median_first_divergence'] and b['median_corridor_steps']+1e-8>=a['median_corridor_steps'] and b['median_min_position']<=a['median_min_position']+1e-8;c0ok=reports['C0']['100']['success_count']>=2 and reports['C0']['250']['success_count']>=2 and reports['C0']['250']['contact_health'];nonzero=sum(x['params_changed'] and x['update']['c2_parameter_delta_norm']>0 for x in slots);invariant=max(x['C1_CORE_ACTION_INVARIANCE_MAX_DELTA'] for x in slots);passed=c0ok and c1ok and final['contact_health'] and prog and nonzero>0 and invariant<=1e-7;interface=final['interface']
  if passed:root='NONE';nxt='C2_CONTINUE_TO_1K'
  elif not c1ok:root='C1_INTERFACE_GATE_OR_STATE_DISTRIBUTION_FAILURE';nxt='C1_RECOVERY_SHELL_CAUSAL_REVIEW'
  elif not prog:root='C2_RESIDUAL_DOES_NOT_ACQUIRE_EARLY_APPROACH_POLICY';nxt='C2_DATA_OR_POLICY_CAPACITY_REVIEW'
  else:root='C2_CONTACT_OR_C0_GATE_FAILURE';nxt='STOP'
  result={'PROGRESSIVE_FROZEN_RESIDUAL_V2':'PASS' if passed else 'FAIL','C0_BASE_SHA':base,'C1_RESIDUAL_SHA':c1sha,'C1_RESIDUAL_SHA_UNCHANGED':'YES' if geo.thash(c1p)==c1sha else 'NO','C1_PROBE_SHA':pre['C1_PROBE_SHA'],'C1_CORE_THRESHOLD':core,'C1_SHELL_OUTER_THRESHOLD':outer,'C1_CORE_ACTION_INVARIANCE_MAX_DELTA':invariant,'C2_RESIDUAL_ARCHITECTURE':pre['C2_RESIDUAL_ARCHITECTURE'],'C2_PILOT':'PASS' if passed else 'FAIL','TRAINABLE_RESIDUAL_UPDATE_COUNT':len(slots),'NONZERO_C2_RESIDUAL_UPDATES':nonzero,'C2_SUCCESS_250':f"{final['success_count']}/3",'C2_NEAR_HOLE_250':final['near_hole_rate'],'C2_FIRST_DIVERGENCE_250':final['median_first_divergence'],'C2_CORRIDOR_250':final['median_corridor_steps'],'C2_MIN_POSITION_250':final['median_min_position'],'C2_CONTACT_250':'PASS' if final['contact_health'] else 'FAIL','C2_PROGRESS_CHECKS':checks,'C1_REGRESSION_FIRST_DIVERGENCE_250':b['median_first_divergence'],'C1_REGRESSION_CORRIDOR_250':b['median_corridor_steps'],'C1_REGRESSION_MIN_POSITION_250':b['median_min_position'],'C1_REGRESSION_CONTACT_250':'PASS' if b['contact_health'] else 'FAIL','C1_WITHIN_CONFIRMED_HEALTHY_ENVELOPE':'YES' if c1ok else 'NO','C0_SUCCESS_250':f"{reports['C0']['250']['success_count']}/3",'C1_SHELL_ENTRY_COUNT':interface['C1_SHELL_ENTRY_COUNT'],'C1_CORE_ENTRY_COUNT':interface['C1_CORE_ENTRY_COUNT'],'C1_HANDOFF_SUCCESS':interface['C1_HANDOFF_SUCCESS'],'NEXT_STAGE_SUPPORT_ENTRY_RATE':float((interface['C1_SHELL_ENTRY_COUNT']+interface['C1_CORE_ENTRY_COUNT'])/3),'CONTACT_IS_DOWNSTREAM':'YES' if any(checks.values()) and not final['contact_health'] and interface['C1_CORE_ENTRY_COUNT']==0 else 'NO','C2_1K':'NOT_RUN','C2_MATCHED_CONFIRMATION':'NOT_RUN','FINAL_CHECKPOINT':checkpoints['250'],'FINAL_REPLAY':str(OUT/'online_replay.pkl'),'FINAL_REPLAY_SHA256':replay['sha256'],'FIRST_BAD_GATE':None if passed else 'C2_250_PILOT','PRIMARY_ROOT_CAUSE':root,'NEXT':nxt,'PRODUCTION_DEMO_SHA_UNCHANGED':'YES','ONLINE_ADMISSION_THRESHOLD':p4.TH,'ONLINE_ADMISSION_THRESHOLD_CHANGED':'NO','EVAL_RESET_CANONICAL':'NOT_RUN','EVAL_GOAL_CANONICAL':'NOT_RUN','EVAL_CURRICULUM_ACTIVE':'TRAIN_C2_ONLY','PROTECTED_ARTIFACTS_CHANGED':'NO','TRAINING_RUNNING':False,'RESIDUAL_PROCESSES':'NONE','elapsed_seconds':time.time()-started};after={'demo':p4.sha(p4.DEMO),'corrective':p4.sha(p4.CORR),'local':p4.sha(p4.LOCAL)};req(before==after,'protected');p4.dump(OUT/'result.json',result);p4.dump(OUT/'protected_after.json',after);print(json.dumps(p4.jv(result),indent=2,sort_keys=True))
if __name__=='__main__':main()
