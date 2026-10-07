"""CPU-only regression and read-only protection audit; no training or cloud API."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/validation/peg/f2_retention_20260904'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def protected():
    cfg=yaml.safe_load((ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml').read_text())
    files=[ROOT/cfg['demo']['path']]
    for directory in ('configs/peg_insertion','results/checkpoints/peg_rlpd_baseline_seed0',
                      'results/validation/peg/f2_run_20260904'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files.extend((ROOT/'results/validation/peg').glob('formal_readiness_v*.json'))
    for directory in ('training/peg_insertion','external/serl/serl_launcher/serl_launcher/agents'):
        files.extend(p for p in (ROOT/directory).rglob('*.py') if p.name not in ('resume.py','bundle_retention.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def main():
    OUT.mkdir(exist_ok=False)
    before=protected()
    (OUT/'protected_before.json').write_text(json.dumps(before,indent=2))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'external/serl/serl_launcher')+os.pathsep+str(ROOT))
    commands=[['scripts/test_peg_bundle_retention.py'],['scripts/test_peg_snapshot_consistency.py'],
              ['scripts/test_peg_f1.py'],['scripts/test_peg_f2_exitfix.py']]
    results=[]
    for args in commands:
        run=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (OUT/(Path(args[0]).stem+'.log')).write_text(run.stdout)
        results.append(dict(command=args,exit_code=run.returncode))
        print(args,run.returncode,flush=True)
    for name in ('training/peg_insertion/resume.py','training/peg_insertion/bundle_retention.py','scripts/test_peg_bundle_retention.py'):
        compile((ROOT/name).read_text(),name,'exec')
    after=protected();unchanged=before==after
    (OUT/'protected_after.json').write_text(json.dumps(after,indent=2))
    sys.path.insert(0,str(ROOT))
    from training.peg_insertion.bundle_retention import inventory
    formal=ROOT/'results/checkpoints/peg_rlpd_baseline_seed0'
    try:inventory(formal);partial='NOT_DETECTED'
    except ValueError as exc:partial=str(exc)
    import flax.training.checkpoints as flaxcp
    import inspect
    (OUT/'installed_flax_cleanup.txt').write_text(inspect.getsource(flaxcp._remove_invalid_ckpts))
    passed=all(r['exit_code']==0 for r in results) and unchanged and 'incomplete' in partial
    report=dict(result='PASS' if passed else 'FAIL',tests=results,protected_files_unchanged=unchanged,
                protected_file_count=len(before),current_formal_incomplete_detected=partial,
                formal_training_rerun=False,checkpoint_resumed=False,evaluation_started=False,cloud_modified=False,
                changed_source_hashes={n:sha(ROOT/n) for n in ('training/peg_insertion/resume.py','training/peg_insertion/bundle_retention.py')})
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
