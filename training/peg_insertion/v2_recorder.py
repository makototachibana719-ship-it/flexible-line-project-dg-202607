"""Durable V2-only recorder; CPU validation, no learning/physics imports."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from training.peg_insertion.observation_contract import V2

PHASES=('setup','approach','lateral_alignment','orientation_correction','insertion','success')
TYPES=('standard','lateral_plus_y','lateral_minus_y','orientation_roll','orientation_pitch')
HASHES=('source_hash','config_hash','runtime_hash','env_hash','recorder_hash')
ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'external/packages/peg_insertion/peg_insertion'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def vector(x,n):
    a=np.asarray(x,dtype=float);require(a.shape==(n,) and np.isfinite(a).all(),'shape/finite violation');return a
def contract(version):require(version==V2,'V2 observation contract required')

def bindings(root=ROOT):
    root=Path(root);pkg=root/'external/packages/peg_insertion/peg_insertion'
    recorder=[root/'training/peg_insertion/v2_recorder.py',root/'scripts/record_peg_v2_demo.py',root/'training/peg_insertion/observation_contract.py',root/'training/peg_insertion/peg_runtime_adapter.py']
    env=sorted((pkg/'code/env/mujoco_env/mujoco_env').rglob('*.py'))+[p for p in (pkg/'code/env/mujoco_env/mujoco_env/assets').rglob('*') if p.is_file()]
    env+=[pkg/'code/example/async_rlpd_peg_sim/peg_env_adapter.py']
    config=root/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml';runtime=root/'configs/peg_insertion/runtime_gpu_formal.env'
    frozen=root/'results/validation/peg/f3_data_review_20260904/frozen_spec_manifest.json'
    for relative,h in json.loads(frozen.read_text()).items():
        require(sha(root/relative.replace('\\','/'))==h,'frozen protocol/source mismatch: '+relative)
    files={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in sorted(set(env+recorder+[config,runtime,frozen]))}
    return dict(source_hash=digest(files),config_hash=sha(config),runtime_hash=sha(runtime),
        env_hash=digest({str(p.relative_to(root)).replace('\\','/'):sha(p) for p in env}),recorder_hash=digest({p.name:sha(p) for p in recorder}),files=files)

def validate_binding(value):
    for k in HASHES:
        v=value.get(k);require(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v),'hash required: '+k)
    require(bool(value.get('files')),'source file manifest required')

def validate_snapshot(s):
    contract(s.get('observation_contract_version'))
    obs=vector(s['obs'],20);vector(s['target_pose'],7);actual=vector(s['actual_ee_pose'],7)
    require(np.allclose(obs[:7],actual,rtol=0,atol=1e-6),'actual pose/obs mismatch')
    raw=np.asarray(s['raw_sensordata'],dtype=float);require(raw.ndim==1 and np.isfinite(raw).all(),'raw sensors finite/vector required')
    table=s['sensors'];require(bool(table),'sensor table required');names=set();ids=set()
    for item in table:
        i,a,d=item['id'],item['adr'],item['dim']
        require(all(type(x) is int for x in (i,a,d)) and i>=0 and a>=0 and d>0 and a+d<=len(raw),'sensor adr/dim invalid')
        require(i not in ids and item['name'] not in names,'duplicate sensor');ids.add(i);names.add(item['name'])
    require(ids==set(range(len(table))) and sum(x['dim'] for x in table)==len(raw),'full sensor table required')
    occupied=[j for x in table for j in range(x['adr'],x['adr']+x['dim'])]
    require(sorted(occupied)==list(range(len(raw))),'sensor address overlap/gap')
    for name,sl in [('end_force_sensor',slice(7,10)),('end_torque_sensor',slice(10,13))]:
        item=next((x for x in table if x['name']==name),None);require(item is not None and item['dim']==3,'force/torque dim3 required')
        require(np.allclose(obs[sl],raw[item['adr']:item['adr']+3],rtol=1e-6,atol=1e-6),'raw sensor reconstruction mismatch')
    m=s['metrics'];require(all(k in m and np.isfinite(m[k]) and m[k]>=0 for k in ('lateral_error','axial_error','orientation_error')),'metrics required/finite')
    require(np.isfinite(s['sim_time']),'sim time finite required')

def validate_transition(row,previous,index,episode_type,binding):
    contract(row.get('observation_contract_version'));require(row.get('episode_type')==episode_type,'episode type mismatch')
    require(row.get('phase') in PHASES,'phase required');require(row.get('step')==index,'step discontinuity')
    require(row.get('hashes')==binding,'hash binding changed')
    validate_snapshot(row['before']);validate_snapshot(row['after'])
    require(row['before']==previous,'snapshot discontinuity')
    require(row['before']['sensors']==row['after']['sensors'],'sensor mapping drift')
    require(np.isclose(row['after']['sim_time']-row['before']['sim_time'],.02,atol=1e-8,rtol=0),'time discontinuity')
    require(row['obs']==row['before']['obs'] and row['next_obs']==row['after']['obs'],'transition obs mismatch')
    a=vector(row['action'],7);require(np.max(np.abs(a))<=1,'action bounds')
    require(all(type(row[k]) is bool for k in ('terminated','truncated','success')),'termination boolean required')
    m=row['after']['metrics'];success=m['lateral_error']<=.002 and m['axial_error']<=.003 and m['orientation_error']<=.05
    require(row['success']==success and row['terminated']==success,'success definition mismatch')
    require(not(row['truncated'] and row['success']),'truncation cannot be successful demo')
    require((row['phase']=='success')==success,'success phase mismatch')
    if index==1 and not success:require(row['phase']=='setup','first transition must preserve setup')
    reward=-80*m['lateral_error']-20*m['axial_error']-2*m['orientation_error']+2*(m['lateral_error']<.01)+25*success
    require(np.isfinite(row['reward']) and np.isclose(row['reward'],reward,atol=1e-5),'reward mismatch')

class EpisodeWriter:
    """Exclusive directory + durable append; interruption remains quarantined."""
    def __init__(self,path,episode_type,initial,binding,verify_binding):
        require(episode_type in TYPES,'episode_type required');validate_snapshot(initial);validate_binding(binding)
        require(verify_binding()==binding,'startup hashes changed')
        self.path=Path(path);require(not self.path.exists(),'dirty/existing output rejected')
        self.path.mkdir(parents=True,exist_ok=False)
        self.type=episode_type;self.initial=initial;self.previous=initial;self.binding=binding;self.verify=verify_binding;self.rows=0;self.ended=False;self.closed=False
        self._json('episode.json',dict(observation_contract_version=V2,episode_type=episode_type,hashes=binding,initial=initial,training_approved=False))
        self._json('status.json',dict(status='IN_PROGRESS',eligible=False,reason='missing complete status means interrupted/quarantined'))
        self.stream=(self.path/'transitions.jsonl').open('x')
    def _json(self,name,value):
        tmp=self.path/(name+'.tmp')
        with tmp.open('w') as f:json.dump(value,f,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.path/name)
    def append(self,row):
        require(not self.closed and not self.ended,'episode already ended')
        try:validate_transition(row,self.previous,self.rows+1,self.type,self.binding)
        except Exception as exc:
            self._json('rejected_transition.json',dict(reason=str(exc),raw_repr=repr(row),eligible=False))
            raise
        self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.stream.flush();os.fsync(self.stream.fileno())
        self.rows+=1;self.previous=row['after'];self.ended=row['terminated'] or row['truncated'];self.last=row
    def finish(self):
        require(not self.closed and self.ended,'cannot finalize interrupted episode')
        require(self.verify()==self.binding,'end hashes changed')
        self.stream.close();self.closed=True
        status='COMPLETED_SUCCESS' if self.last['success'] else 'COMPLETED_REJECTED'
        self._json('status.json',dict(status=status,eligible=False,needs_pilot_coverage_review=True,transitions=self.rows,
            transitions_sha256=sha(self.path/'transitions.jsonl'),training_approved=False))
    def abort(self,reason):
        if not self.closed:self.stream.close();self.closed=True
        self._json('status.json',dict(status='INTERRUPTED',eligible=False,transitions=self.rows,reason=str(reason),training_approved=False))

def snapshot(env):
    """Passive copy only; does not step, reset, or call controller."""
    raw=env.env;ob=raw._get_obs();obs=env._state(ob);metrics=raw.task._goal_metrics(ob)
    return dict(observation_contract_version=env.observation_contract_version,obs=obs.tolist(),target_pose=env._target_pose.tolist(),
        actual_ee_pose=obs[:7].tolist(),raw_sensordata=raw.data.sensordata.copy().tolist(),sim_time=float(raw.data.time),
        sensors=[dict(name=raw.model.sensor(i).name,id=i,adr=int(raw.model.sensor_adr[i]),dim=int(raw.model.sensor_dim[i])) for i in range(raw.model.nsensor)],
        metrics={k:float(metrics[k]) for k in ('lateral_error','axial_error','orientation_error')})

def preflight(output,root=ROOT):
    require(not Path(output).exists(),'dirty/existing output rejected')
    import yaml
    cfg=yaml.safe_load((root/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml').read_text())
    contract(cfg['observation']['contract_version'])
    value=bindings(root);validate_binding(value)
    return dict(status='PASS',observation_contract_version=V2,hashes=value,output=str(output),
        scope='CPU/static preflight only; runtime sensor/env validation required before first recorded step',demo_recording_started=False)
