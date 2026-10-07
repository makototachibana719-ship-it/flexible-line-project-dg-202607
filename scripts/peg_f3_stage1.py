"""One checkpoint, ten inference-only episodes. Original evaluator remains untouched."""
import os, sys, json, time, hashlib, subprocess, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Set approved inference runtime before importing JAX; no file is sourced as code.
RUNTIME={'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_FLAGS':'--xla_gpu_enable_triton_gemm=false','MUJOCO_GL':'egl'}
os.environ.update(RUNTIME)
import numpy as np
import jax
from flax import serialization
from evaluation.peg_insertion import evaluate_policy as original
from training.peg_insertion import formal_policy as policy

OUT=ROOT/'results/validation/peg/f3_stage1_20260904'
CP=ROOT/'results/checkpoints/peg_rlpd_baseline_seed0/checkpoint_9000'
def sha(path):return policy.sha(path)
def check(ok,reason):
    if not ok:raise RuntimeError(reason)
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def signature(state):
    h=hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        value=np.asarray(leaf);check(np.isfinite(value).all(),'nonfinite model state')
        h.update(str((value.shape,str(value.dtype))).encode());h.update(value.tobytes())
    return h.hexdigest()
def equal(left,right):
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    return np.array_equal(np.asarray(left),np.asarray(right))
def forbidden(*args,**kwargs):raise RuntimeError('training/update forbidden in evaluator')

def main():
    OUT.mkdir(exist_ok=False)
    result=dict(PEG_F3_EVAL_PREFLIGHT='FAIL',PEG_F3_CHECKPOINT_9000_LOAD='FAIL',PEG_F3_CHECKPOINT_9000_SMOKE='FAIL',
                PEG_F3_SCREENING='BLOCKED',evaluation_started=False,training_started=False,checkpoint_modified=False,cloud_modified=False)
    env=None;rows=[];protected={}
    try:
        allowpath=ROOT/'results/validation/peg/f2_artifact_adjudication_20260904/adjudication.json'
        allow=json.loads(allowpath.read_text());check(allow['PEG_FORMAL_EVALUATION']=='READY','artifact release')
        item=next(x for x in allow['bundles'] if x['step']==9000)
        check(CP.resolve()==(ROOT/'results/checkpoints/peg_rlpd_baseline_seed0/checkpoint_9000').resolve(),'exact path')
        for name,digest in item['hashes'].items():check(sha(CP.parent/name)==digest,'checkpoint hash '+name)
        for pattern in ('[r]un_peg_training','[a]sync_sac_state_sim'):
            check(subprocess.run(['pgrep','-af',pattern],capture_output=True).returncode==1,'learner/actor running')
        gate=json.loads((ROOT/'results/validation/peg/formal_readiness_v3.json').read_text())
        check(policy.bindings()==gate['bindings'],'runtime/config/Demo binding')
        protected_paths=list(CP.parent.glob('checkpoint_*'))+[original.CFG,ROOT/'configs/peg_insertion/runtime_gpu_formal.env',Path(original.__file__),allowpath]
        protected={str(p):sha(p) for p in protected_paths};write('protected_before.json',protected)
        cfg=original.yaml.safe_load(original.CFG.read_text())
        check(cfg['environment']=='AuboPegInsert-State-v0','environment id')
        check(jax.default_backend()=='gpu','GPU backend')
        agent=original.make_sac_agent(cfg['seed'],np.zeros(20,np.float32),np.zeros(7,np.float32))
        agent=agent.replace(state=original.restore_fresh_state(CP.parent,agent.state,9000))
        check(equal(serialization.to_state_dict(agent.state),serialization.msgpack_restore(CP.read_bytes())),'model exact load')
        initial_state=agent.state;initial=signature(initial_state)
        result['PEG_F3_CHECKPOINT_9000_LOAD']='PASS'
        for name in ('update','update_high_utd'):
            if hasattr(type(agent),name):setattr(type(agent),name,forbidden)
        with original.portable_peg_runtime(ROOT):env=original.make_peg_state_env(render_mode='rgb_array',max_episode_steps=500)
        obs,_=env.reset(seed=0)
        check(obs.shape==(20,) and np.isfinite(obs).all(),'observation contract')
        check(env.action_space.shape==(7,) and np.all(env.action_space.low==-1) and np.all(env.action_space.high==1),'action contract')
        task=env.env.task
        check(task.position_tolerance==.002 and task.orientation_tolerance==.05,'success thresholds')
        check(env.env.control_dt==.02,'control dt')
        write('preflight.json',dict(status='PASS',checkpoint=str(CP),hashes=item['hashes'],backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],
            environment=cfg['environment'],runtime=RUNTIME,obs_shape=[20],action_shape=[7],seeds=list(range(10)),max_steps=500,argmax=True,
            reward=cfg['contracts']['reward'],success=cfg['contracts']['success'],original_evaluator=str(original.__file__),
            metrics=dict(cycle_time='steps * 0.02 simulated seconds',approach_reduction='reset position error minus minimum including reset',action_saturation='fraction of all 7 action coordinates with abs >= 0.99'),
            replay='no buffer constructed, no replay writer imported',optimizer='no update call; agent update methods guarded',state_sha256=initial))
        result['PEG_F3_EVAL_PREFLIGHT']='PASS';print('PREFLIGHT PASS / MODEL LOAD PASS',flush=True)
        for episode in range(10):
            obs,_=env.reset(seed=episode);check(np.isfinite(obs).all(),'reset nonfinite')
            first=float(np.linalg.norm(obs[13:16]));minimum=first;lat=[];axial=[];forces=[];saturated=0;ret=0.;t0=time.monotonic()
            result['evaluation_started']=True
            for index in range(500):
                check(agent.state is initial_state,'policy state identity changed')
                action=np.asarray(jax.device_get(agent.sample_actions(observations=obs,argmax=True)))
                check(action.shape==(7,) and np.isfinite(action).all(),'action/model output nonfinite')
                check(np.all(np.abs(action)<=1.000001),'action bounds')
                saturated+=int((np.abs(action)>=.99).sum())
                obs,reward,terminated,truncated,info=env.step(action)
                check(obs.shape==(20,) and np.isfinite(obs).all() and np.isfinite(reward),'observation/reward nonfinite')
                keys=('position_error','lateral_error','axial_error','orientation_error','force_norm','is_success')
                check(all(k in info for k in keys),'metric schema missing')
                check(all(np.isfinite(info[k]) for k in keys),'metric schema nonfinite')
                success=bool(info['is_success'])
                expected=info['lateral_error']<=.002 and info['axial_error']<=.003 and info['orientation_error']<=.05
                check(success==expected,'success definition mismatch')
                expected_reward=-80*info['lateral_error']-20*info['axial_error']-2*info['orientation_error']+2*(info['lateral_error']<.01)+25*success
                check(np.isclose(reward,expected_reward),'reward definition mismatch')
                minimum=min(minimum,float(info['position_error']));lat.append(float(info['lateral_error']));axial.append(float(info['axial_error']));forces.append(float(info['force_norm']));ret+=float(reward)
                if terminated or truncated:break
            check(signature(agent.state)==initial,'evaluator changed policy state')
            row=dict(episode=episode,seed=episode,success=success,steps=index+1,cycle_time=(index+1)*.02,wall_seconds=time.monotonic()-t0,
                     return_=ret,first_position_err=first,min_position_err=minimum,approach_reduction=first-minimum,min_lateral_err=min(lat),min_axial_err=min(axial),
                     max_force=max(forces),action_saturation=saturated/(7*(index+1)),terminated=bool(terminated),truncated=bool(truncated),
                     termination_reason='SUCCESS' if success else ('TRUNCATED' if truncated else ('TERMINATED' if terminated else 'EVALUATOR_HORIZON')))
            row['return']=row.pop('return_');rows.append(row)
            with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
        result.update(PEG_F3_CHECKPOINT_9000_SMOKE='PASS',PEG_F3_SCREENING='READY',episodes=len(rows),successes=sum(r['success'] for r in rows),success_rate=np.mean([r['success'] for r in rows]))
        for output,key in [('mean_steps','steps'),('mean_cycle_time','cycle_time'),('mean_return','return'),('mean_min_position_err','min_position_err'),('mean_max_force','max_force'),('mean_action_saturation','action_saturation')]:result[output]=float(np.mean([r[key] for r in rows]))
    except Exception as exc:
        result['error']=repr(exc);traceback.print_exc()
    finally:
        if env is not None:env.close()
        after={path:sha(Path(path)) for path in protected};write('protected_after.json',after)
        if protected!=after:result.update(checkpoint_modified=True,PEG_F3_CHECKPOINT_9000_SMOKE='FAIL',PEG_F3_SCREENING='BLOCKED',error='protected file hash changed')
        result.setdefault('episodes',len(rows));write('result.json',result);print(json.dumps(result,indent=2),flush=True)
    return 0 if result['PEG_F3_CHECKPOINT_9000_SMOKE']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
