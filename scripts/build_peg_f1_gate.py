"""CPU/static F1 hardening acceptance and source-bound Gate publication only."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.peg_insertion import formal_policy as p,formal_preflight as f

OUT=p.ROOT/'results/validation/peg/f1_hardening_20260904'
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2)
def main():
    if p.GATE.exists():raise FileExistsError('never overwrite readiness v2')
    OUT.mkdir(parents=True,exist_ok=False)
    evidence=['results/diagnostics/peg_f0_verifier_fix_20260904/reaudit.json',
              'results/validation/peg/f0_r_branch_20260904/audit.json',
              'results/validation/peg/f0_r_branch_20260904/offline_final.json']
    checks=[json.loads((p.ROOT/name).read_text()) for name in evidence]
    assert all(c['status']=='PASS' for c in checks)
    assert checks[0]['wandb_run_id']=='hr88a6u1' and checks[0]['acceptance']=='PASS_WITH_PER_LEAF_SYNC'
    assert checks[1]['source_run_id']=='hr88a6u1' and checks[1]['updates']==list(range(26,31))
    # Preserve every existing B/branch checkpoint, old Gate, and old raw audits.
    protected={}
    for root in ('f0_sync_20260904','f0_r_branch_20260904'):
        for path in (p.ROOT/'results/validation/peg'/root/'checkpoints').glob('checkpoint_*'):protected[str(path)]=p.sha(path)
    old=p.ROOT/'results/validation/peg/formal_readiness.json';protected[str(old)]=p.sha(old)
    write(OUT/'protected_before.json',protected)
    for name in p.SOURCES:
        if name.endswith('.py'):ast.parse((p.ROOT/name).read_text())
        if name.endswith('.sh'):subprocess.run(['bash','-n',str(p.ROOT/name)],check=True)
    tests=[]
    for name in ('test_peg_f1.py','test_peg_f0_schema.py','test_peg_f0_policy.py'):
        result=subprocess.run([sys.executable,'scripts/'+name],cwd=p.ROOT,capture_output=True,text=True,timeout=60)
        tests.append(dict(script=name,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    write(OUT/'tests.json',tests)
    if any(t['exit_code'] for t in tests):
        print(json.dumps(tests,indent=2));raise RuntimeError('CPU tests failed; no Gate published')
    p.validate_runtime(True)
    assert p.bindings()['demo_hash']==f.config()['demo']['sha256']
    evidence.append(str((OUT/'tests.json').relative_to(p.ROOT)))
    gate=dict(schema_version=2,formal_ready=True,formal_training_started=False,
        exact_path_acceptance_50='PASS_WITH_PER_LEAF_SYNC',exact_path_checkpoint_resume_5='PASS',
        checkpoint_resume_mode='NEW_BRANCH_RUN',same_run_continuation_in_f0_r=False,
        sync_policy='PER_LEAF_BLOCK_UNTIL_READY',sync_exact_root_cause='UNRESOLVED',
        runtime=p.RUNTIME,bindings=p.bindings(),source_hashes={name:p.sha(p.ROOT/name) for name in p.SOURCES},
        evidence_hashes={name:p.sha(p.ROOT/name) for name in evidence},
        role_aware_preflight={role:'PASS' for role in f.ROLES},
        role_preflight_scope='CPU fixtures; fresh learner additionally checks actual clean output paths. No actor or learner launched.',
        run_binding=dict(project=f.PROJECT,fresh_name=f.RUN_NAME,resume='must_same_run_id'),
        checkpoint_cadence=dict(basis='global_learner_update',period=1000,max_steps=10000),
        validation_scope='Historical bounded GPU evidence plus current CPU/static hardening; no new GPU run',
        live_launch_checks='Gate, runtime, clean/bound outputs, endpoint/session/process locks; resume cloud history must not exceed checkpoint',
        formal_training='READY_TO_START')
    candidate=OUT/'gate_candidate.json';write(candidate,gate)
    with patch.object(p,'GATE',candidate):
        result=f.preflight('fresh_learner')
    write(OUT/'actual_fresh_preflight.json',result)
    rel=str((OUT/'actual_fresh_preflight.json').relative_to(p.ROOT));gate['evidence_hashes'][rel]=p.sha(p.ROOT/rel)
    assert all(p.sha(Path(name))==digest for name,digest in protected.items())
    write(p.GATE,gate)
    p.require_gate()
    result=dict(status='PASS',formal_ready=True,formal_training='READY_TO_START',formal_training_started=False,
        sync_policy='PASS',role_aware_preflight='PASS',run_binding='PASS',global_checkpoint_cadence='PASS',
        formal_runtime_freeze='PASS',gate_enforcement='PASS',gpu_training_run=False,protected_artifacts_unchanged=True)
    write(OUT/'result.json',result)
    print(json.dumps(result,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
