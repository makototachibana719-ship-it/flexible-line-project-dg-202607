"""CPU/static monitoring acceptance. Never starts training or contacts W&B."""
import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.peg_insertion import formal_policy as p, formal_preflight as f

OUT=ROOT/'results/validation/peg/f2_monitor_20260904'
OLD=ROOT/'results/validation/peg/formal_readiness_v2.json'

def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2)

def main():
    if p.GATE.exists():raise FileExistsError('v3 exists; never overwrite Gate')
    OUT.mkdir(parents=True,exist_ok=False)
    old=json.loads(OLD.read_text())
    assert old['formal_ready'] is True and old['schema_version']==2
    assert p.sha(OLD)=='1e6bf0c8ce3e80b233984984ca2ca51231b1568549a2b7e4dc88b000a350f916'
    protected={str(OLD.relative_to(ROOT)):p.sha(OLD)}
    for root in ('f0_sync_20260904','f0_r_branch_20260904'):
        for path in (ROOT/'results/validation/peg'/root/'checkpoints').glob('checkpoint_*'):
            protected[str(path.relative_to(ROOT))]=p.sha(path)
    protected['results/validation/peg/formal_readiness.json']=p.sha(ROOT/'results/validation/peg/formal_readiness.json')
    write(OUT/'protected_before.json',protected)
    allowed={'training/peg_insertion/run_peg_training.py','training/peg_insertion/formal_policy.py','scripts/test_peg_f1.py'}
    for name,digest in old['source_hashes'].items():
        if name not in allowed:assert p.sha(ROOT/name)==digest,name
    assert p.bindings()==old['bindings']
    for name,digest in old['evidence_hashes'].items():assert p.sha(ROOT/name)==digest,name
    syntax=[]
    for name in p.SOURCES:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text());syntax.append(name)
        if name.endswith('.sh'):subprocess.run(['bash','-n',str(ROOT/name)],check=True);syntax.append(name)
    write(OUT/'syntax.json',syntax)
    tests=[]
    for name in ('test_peg_f2_monitor.py','test_peg_f1.py','test_peg_f0_schema.py','test_peg_f0_policy.py'):
        result=subprocess.run([sys.executable,'scripts/'+name],cwd=ROOT,capture_output=True,text=True,timeout=60)
        tests.append(dict(script=name,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    write(OUT/'tests.json',tests)
    if any(t['exit_code'] for t in tests):
        print(json.dumps(tests,indent=2));raise RuntimeError('CPU regression failed; no v3 Gate')
    p.validate_runtime(True)
    gate=copy.deepcopy(old)
    gate.update(schema_version=3,formal_step_watchdog=True,formal_finite_loss_guard=True,
        formal_rng_guard=True,formal_per_update_local_monitor=True,formal_monitoring_gate='PASS',
        monitor_implementation_hash=p.sha(ROOT/p.MONITOR_SOURCE),
        monitor_policy=dict(timeout_seconds=120,snapshot_seconds=[30,60,110],wandb_log_period=10),
        source_hashes={name:p.sha(ROOT/name) for name in p.SOURCES},
        validation_scope='Historical GPU exact-path evidence plus CPU/static F2 monitoring regression; no new GPU run',
        monitor_validation_scope='Synthetic CPU failures and subprocess hard-exit; native GPU timeout not re-exercised',
        formal_training_started=False,formal_training='READY_TO_START',wandb_run_created=False)
    for path in (OLD,OUT/'tests.json',OUT/'syntax.json',OUT/'protected_before.json'):
        gate['evidence_hashes'][str(path.relative_to(ROOT))]=p.sha(path)
    candidate=OUT/'gate_candidate.json';write(candidate,gate)
    with patch.object(p,'GATE',candidate):
        p.require_gate()
        preflight=f.preflight('fresh_learner')
    write(OUT/'actual_fresh_preflight.json',preflight)
    gate['evidence_hashes'][str((OUT/'actual_fresh_preflight.json').relative_to(ROOT))]=p.sha(OUT/'actual_fresh_preflight.json')
    assert all(p.sha(ROOT/name)==digest for name,digest in protected.items())
    write(p.GATE,gate)
    p.require_gate()
    result=dict(status='PASS',formal_ready=True,formal_training_started=False,wandb_run_created=False,
        formal_watchdog='PASS',finite_loss_guard='PASS',rng_continuity_guard='PASS',local_step_monitor='PASS',
        wandb_cadence_unchanged=True,checkpoint_cadence_unchanged=True,sync_policy_unchanged=True,
        algorithm_changed=False,runtime_config_demo_unchanged=True,protected_artifacts_unchanged=True,
        gate_sha256=p.sha(p.GATE),execution='READY_TO_START')
    write(OUT/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
