"""Re-audit only. Never calls run_once, training entry, resume or wandb.init."""
import json
from pathlib import Path
import verify_peg_f0 as v

if __name__=='__main__':
    v.OUT=v.ROOT/'results/validation/peg/f0_sync_20260904'
    dest=v.ROOT/'results/diagnostics/peg_f0_verifier_fix_20260904'
    dest.mkdir(parents=True,exist_ok=False)
    protected=list((v.OUT/'checkpoints').glob('checkpoint_*'))+[v.OUT/'audit.json',v.OUT/'cloud.json',v.OUT.parent/'formal_readiness.json']
    before={str(p):v.sha(p) for p in protected}
    summary=json.loads((v.OUT/'learner/result.json').read_text())
    if summary['wandb_run_id']!='hr88a6u1':raise ValueError('wrong B run')
    try:
        result=v.audit(evidence_dir=dest)
        proc=json.loads((v.OUT/'processes.json').read_text())
        assert proc['exit_codes'][0]==0 and proc['error'] is None
        assert all(v.sha(Path(p))==h for p,h in before.items())
        result.update(acceptance='PASS_WITH_PER_LEAF_SYNC',training_rerun=False,cloud_history_modified=False,
                      sync_exact_root_cause='UNRESOLVED',formal_training_started=False,
                      protected_hashes=before,source_hashes_scope='current audit-time source, not historical execution attestation')
    except Exception as exc:result=dict(status='FAIL',error=repr(exc),formal_ready=False)
    v.write(dest/'reaudit.json',result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
