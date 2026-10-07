"""Read-only contracts and one checkpoint9000 inference episode; no production edits."""
import sys, json, csv, pickle, hashlib
from pathlib import Path
sys.argv = [sys.argv[0], '9000']
import peg_f3_diagnostic_worker as worker
import peg_f3_trace
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = worker.ROOT
OUT = ROOT/'results/validation/peg/f3_contract_audit_20260904'
PKG = ROOT/'external/packages/peg_insertion/peg_insertion'

def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def rot(q):
    return R.from_quat(np.asarray(q)[[1,2,3,0]])

def angle(a,b):
    return float((rot(a).inv()*rot(b)).magnitude())

def append(name,row):
    with (OUT/(name+'.jsonl')).open('a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n')
    p=OUT/(name+'.csv'); fresh=not p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if fresh:w.writeheader()
        w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def demo_audit():
    path=PKG/'demos/peg_insert_20_demos_state.pkl'
    assert worker.sha(path)=='44ee651768d9f231caccf81d5a58460c8f44bc17463de7e08ea157b3c2a53ce7'
    data=pickle.loads(path.read_bytes())
    index=[json.loads(s) for s in (PKG/'demos/peg_demos_v1/index.jsonl').read_text().splitlines()]
    offset=0; summaries=[]
    for ep in range(5):
        meta=index[ep]; src=np.load(PKG/'demos/peg_demos_v1'/meta['file'],allow_pickle=False)
        n=len(src['actions']); dp=[]; dr=[]; ap=[]; ar=[]; discontinuity=0
        for i in range(n):
            item=data[offset+i]
            for key in ('observations','actions','next_observations'):
                assert np.array_equal(item[key],src[key][i]), (ep,i,key)
            o=np.asarray(item['observations']); a=np.asarray(item['actions']); no=np.asarray(item['next_observations'])
            dpos=no[:3]-o[:3]; drot=(rot(o[3:7]).inv()*rot(no[3:7])).as_rotvec()
            request=a[:3]*.002; requestrot=R.from_euler('xyz',a[3:6]*np.pi/180).as_rotvec()
            if i and not np.array_equal(src['next_observations'][i-1],o):discontinuity+=1
            dp.append(dpos);dr.append(drot);ap.append(request);ar.append(requestrot)
            append('demo_transitions',dict(episode=ep,transition=i,obs_t=o.tolist(),action_t=a.tolist(),obs_t1=no.tolist(),
                realized_delta_position=dpos.tolist(),realized_local_rotvec=drot.tolist(),requested_delta_position=request.tolist(),requested_local_rotvec=requestrot.tolist()))
        def compare(req,real):
            req=np.asarray(req);real=np.asarray(real);rn=np.linalg.norm(req,axis=1);dn=np.linalg.norm(real,axis=1);mask=(rn>1e-8)&(dn>1e-8)
            cos=np.sum(req[mask]*real[mask],axis=1)/(rn[mask]*dn[mask])
            return dict(active_pairs=int(mask.sum()),positive_direction_fraction=float(np.mean(cos>0)) if len(cos) else None,
                median_cosine=float(np.median(cos)) if len(cos) else None,median_realized_requested_norm_ratio=float(np.median(dn[mask]/rn[mask])) if len(cos) else None,
                mean_residual=float(np.linalg.norm(real-req,axis=1).mean()))
        summaries.append(dict(episode=ep,transitions=n,metadata=meta,exact_pkl_npz_match=True,observation_discontinuities=discontinuity,
            translation=compare(ap,dp),rotation=compare(ar,dr)))
        offset+=n
    dump('demo_audit.json',dict(episodes=summaries,transitions=offset,note='Actions change accumulated target, not instantaneous realized pose. Tracking ratios alone do not establish contract mismatch.'))

factory=worker.original.make_peg_state_env
trace=[]
def observed_factory(*args,**kwargs):
    env=factory(*args,**kwargs); raw=env.env; ctrl=raw.controller
    sensors=[]
    for i in range(raw.model.nsensor):
        sensors.append(dict(id=i,name=raw.model.sensor(i).name,adr=int(raw.model.sensor_adr[i]),dim=int(raw.model.sensor_dim[i]),type=int(raw.model.sensor_type[i])))
    dump('runtime_contract.json',dict(sensors=sensors,controller=type(ctrl).__name__,ee_site=ctrl.ee_site_name,
        action_low=raw.action_space.low.tolist(),action_high=raw.action_space.high.tolist(),
        control_dt=raw.control_dt,physics_dt=raw.physics_dt,translation_scale=env.translation_scale,rotation_scale=env.rotation_scale))
    original_control=ctrl.compute_control; captured={}
    def control(*args,**kwargs):
        target=kwargs.get('target',args[0] if args else None)
        captured['target']=np.asarray(target).copy()
        result=original_control(*args,**kwargs)
        captured['ctrl']=np.asarray(result).copy()
        return result
    ctrl.compute_control=control
    original_step=env.step
    def step(action):
        before=env._target_pose.copy()
        result=original_step(action)
        obs,reward,terminated,truncated,info=result
        target=captured['target']; applied=env._target_pose.copy()
        correct={}
        for name in ('end_force_sensor','end_torque_sensor'):
            sid=raw.model.sensor(name).id; adr=int(raw.model.sensor_adr[sid]);dim=int(raw.model.sensor_dim[sid])
            correct[name]=raw.data.sensordata[adr:adr+dim].tolist()
        row=dict(step=len(trace)+1,sim_time=float(raw.data.time),policy_action=np.asarray(action).tolist(),previous_target=before.tolist(),
            requested_translation_delta=(np.clip(action[:3],-1,1)*env.translation_scale).tolist(),
            requested_rotation_delta=(np.clip(action[3:6],-1,1)*env.rotation_scale).tolist(),
            applied_target=applied.tolist(),controller_target_position=target[:3].tolist(),controller_target_orientation=target[3:7].tolist(),
            actual_ee_position=obs[:3].tolist(),actual_ee_orientation=obs[3:7].tolist(),
            target_realized_translation_error=float(np.linalg.norm(target[:3]-obs[:3])),
            target_realized_rotation_error=angle(target[3:7],obs[3:7]),
            adapter_controller_target_max_difference=float(np.max(np.abs(applied-target))),
            joint_control=captured['ctrl'].tolist(),observed_wrench=obs[7:13].tolist(),correct_sensor_slices=correct,
            contact_count=int(raw.data.ncon),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
        trace.append(row);append('target_realized_trace',row)
        return result
    env.step=step
    return env

if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    sources=list((PKG/'code').rglob('*.py'))+[ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml',ROOT/'configs/peg_insertion/runtime_gpu_formal.env',PKG/'demos/peg_insert_20_demos_state.pkl']
    before={str(p):worker.sha(p) for p in sources};dump('source_before.json',before)
    demo_audit()
    worker.OUT=OUT/'checkpoint_9000'
    worker.original.make_peg_state_env=observed_factory
    peg_f3_trace.record=lambda *args,**kwargs:None
    code=worker.main()
    after={str(p):worker.sha(p) for p in sources};dump('source_after.json',after)
    assert before==after
    dump('tracking_summary.json',dict(episode_exit=code,steps=len(trace),source_unchanged=True,
        translation_mean=float(np.mean([r['target_realized_translation_error'] for r in trace])) if trace else None,
        rotation_mean=float(np.mean([r['target_realized_rotation_error'] for r in trace])) if trace else None,
        key_steps=[r for r in trace if r['step'] in (1,47,65,67,71,120,500)]))
    raise SystemExit(code)
