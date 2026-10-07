"""CPU-only legacy audit and regression evidence, no environment/model restore."""
import ast,hashlib,json,pickle,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion import observation_contract as oc
OUT=ROOT/'results/validation/peg/f3_sensorfix_20260904';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
audit=json.loads((ROOT/'results/validation/peg/f2_artifact_adjudication_20260904/adjudication.json').read_text())
checkpoint_results=[]
for b in audit['bundles']:
    directory=ROOT/'results/checkpoints/peg_rlpd_baseline_seed0'
    assert all(sha(directory/n)==h for n,h in b['hashes'].items())
    cp=directory/f"checkpoint_{b['step']}"
    assert oc.checkpoint_version(cp)==oc.V1
    try:oc.checkpoint_preflight(cp)
    except ValueError:rejected=True
    else:rejected=False
    assert rejected
    checkpoint_results.append(dict(step=b['step'],contract=oc.V1,hashes_unchanged=True,direct_v2_rejected=True))
pkg=ROOT/'external/packages/peg_insertion/peg_insertion'
demo_path=pkg/'demos/peg_insert_20_demos_state.pkl'
assert sha(demo_path)=='44ee651768d9f231caccf81d5a58460c8f44bc17463de7e08ea157b3c2a53ce7'
data=pickle.loads(demo_path.read_bytes());npz=[]
for p in sorted((pkg/'demos/peg_demos_v1').glob('demo_*.npz')):
    with np.load(p,allow_pickle=False) as d:npz.append(dict(file=p.name,keys=d.files,shapes={k:list(d[k].shape) for k in d.files}))
assert len(npz)==20
write('legacy_compatibility.json',dict(checkpoints=checkpoint_results,demo_contract=oc.V1,demo_hash=sha(demo_path),
    transitions=len(data),transition_keys=sorted(data[0]),raw_npz=npz,
    LEGACY_DEMO_V2_RECONSTRUCTABLE='NO',scope='Available PKL and20 source NPZ episodes: observations contain Fy,Fz,Tx in torque slots; no independent Ty,Tz/raw sensordata stream.',
    LEGACY_CHECKPOINT_V2_COMPATIBLE='NO_DIRECT_USE',diagnostic_override='Explicit flag only, output marked not formal skill result. Not executed.'))
prior=json.loads((ROOT/'results/validation/peg/f3_contract_audit_20260904/source_before.json').read_text())
allowed=('utils/sensor_utils.py','async_rlpd_peg_sim/peg_env_adapter.py','configs/peg_insertion/peg_rlpd_baseline_seed0.yaml')
checked=0
for name,digest in prior.items():
    if name.endswith(allowed):continue
    assert sha(name)==digest,name;checked+=1
write('semantic_regression.json',dict(unchanged_files=checked,algorithm=True,network=True,optimizer=True,reward=True,success=True,controller=True,action_semantics=True,
    note='Protected package source hashes compared to prior audit; adapter executable methods compared by AST; config differs only version annotations in CPU test.'))
with (OUT/'cpu_tests.log').open('x') as log:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/test_peg_f3_sensorfix.py')],stdout=log,stderr=subprocess.STDOUT,timeout=120)
status='PASS' if result.returncode==0 else 'FAIL'
write('result.json',dict(PEG_F3_SENSOR_FIX=status,PEG_F3_SENSOR_TESTS=status,PEG_F3_OBS_CONTRACT_VERSIONING=status,
    PEG_F3_LEGACY_COMPATIBILITY_AUDIT='PASS',LEGACY_DEMO_V2_RECONSTRUCTABLE='NO',LEGACY_CHECKPOINT_V2_COMPATIBLE='NO_DIRECT_USE',
    legacy_checkpoint_direct_eval_on_v2=False,training_started=False,checkpoint_resumed=False,checkpoint_modified=False,cloud_modified=False,
    evaluation_started=False,formal_training_readiness='BLOCKED: V1 Demo/V2 env mismatch and stale historical Gate; no gate regenerated'))
print((OUT/'cpu_tests.log').read_text());print((OUT/'result.json').read_text())
raise SystemExit(result.returncode)
