"""CPU-only exit repair Gate; re-adjudicate existing evidence without cloud writes."""
import ast,copy,json,math,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion import formal_policy as p,formal_preflight as f
OUT=ROOT/'results/validation/peg/f2_exitfix_20260904_review'
OLD=ROOT/'results/validation/peg/formal_readiness_v3.json'
RUN=ROOT/'results/validation/peg/f2_run_20260904'
def write(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2)
def main():
 if p.GATE.exists():raise FileExistsError('never overwrite v4')
 OUT.mkdir(parents=True,exist_ok=False)
 old=json.loads(OLD.read_text());assert p.sha(OLD)=='9967d06cb4e35715b1ed7fdb262ce1ae6f3462f10e5a5e3955b5a94871a9039c'
 allowed={'training/peg_insertion/run_peg_training.py','training/peg_insertion/formal_policy.py','scripts/test_peg_f1.py'}
 for name,digest in old['source_hashes'].items():
  if name not in allowed:assert p.sha(ROOT/name)==digest,name
 assert p.bindings()==old['bindings']
 protected={str(OLD.relative_to(ROOT)):p.sha(OLD)}
 for path in RUN.glob('*'):
  if path.is_file():protected[str(path.relative_to(ROOT))]=p.sha(path)
 for path in (ROOT/'results/checkpoints/peg_rlpd_baseline_seed0').glob('checkpoint_*'):protected[str(path.relative_to(ROOT))]=p.sha(path)
 write(OUT/'protected.json',protected)
 tests=[]
 for name in ('test_peg_f2_exitfix.py','test_peg_f2_monitor.py','test_peg_f1.py','test_peg_f0_schema.py','test_peg_f0_policy.py'):
  result=subprocess.run([sys.executable,'scripts/'+name],cwd=ROOT,capture_output=True,text=True,timeout=60)
  tests.append(dict(script=name,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
 write(OUT/'tests.json',tests)
 if any(x['exit_code'] for x in tests):print(json.dumps(tests,indent=2));raise RuntimeError('CPU tests failed')
 for name in p.SOURCES:
  if name.endswith('.py'):ast.parse((ROOT/name).read_text())
  if name.endswith('.sh'):subprocess.run(['bash','-n',str(ROOT/name)],check=True)
 raw=json.loads((RUN/'result.json').read_text());assert raw['PEG_FORMAL_TRAINING']=='FAIL' and raw['wandb_run_id']=='e6fea100' and raw['wandb_state']=='failed'
 rows=[json.loads(x) for x in (Path(raw['monitor_directory'])/'formal_monitor_updates.jsonl').read_text().splitlines()]
 assert [r['global_update'] for r in rows]==list(range(10000))
 assert all(r['true_update_seconds']<120 and r['rng_before']!=r['rng_after'] and all(math.isfinite(r[k]) for k in ('critic_loss','actor_loss','temperature_loss')) for r in rows)
 assert all(a['rng_after']==b['rng_before'] for a,b in zip(rows,rows[1:]))
 cloud=json.loads((RUN/'cloud_history.json').read_text());assert len(cloud)==1000 and sorted(r['_step'] for r in cloud)==list(range(0,10000,10))
 for row in cloud:
  for key,local in [('critic.critic_loss','critic_loss'),('actor.actor_loss','actor_loss'),('temperature.temperature_loss','temperature_loss')]:
   assert math.isfinite(row[key]) and abs(row[key]-rows[row['_step']][local])<=1e-6*max(1,abs(rows[row['_step']][local]))
 missing=[]
 for step in range(0,10000,1000):
  audit=json.loads((RUN/f'checkpoint_{step}_audit.json').read_text());assert audit['status']=='PASS'
  for name,digest in audit['hashes'].items():
   path=ROOT/'results/checkpoints/peg_rlpd_baseline_seed0'/name
   if not path.exists():missing.append(name)
   else:assert p.sha(path)==digest
 adjudication=dict(PEG_FORMAL_TRAINING_RAW_GATE='FAIL',PEG_FORMAL_TRAINING_COMPUTATION='PASS',PEG_FORMAL_CHECKPOINT_AUDIT='PASS',PEG_FORMAL_WANDB_HISTORY='PASS',PEG_FORMAL_WANDB_FINALIZATION='FAIL_KNOWN_BUG',PEG_FORMAL_TRAINING_ADJUDICATION='PASS_WITH_WANDB_FINALIZATION_BUG',run_id='e6fea100',recorded_cloud_state='failed',cloud_scope='Preserved F2 query evidence; no cloud mutation or new run',PEG_FORMAL_EVALUATION='READY',evaluation_scope='Ready for separately authorized evaluation; not evaluated',training_rerun=False,checkpoint_resumed=False,cloud_modified=False,evaluation_started=False)
 if missing:
  adjudication.update(PEG_FORMAL_CHECKPOINT_AUDIT='FAIL_CURRENT_RETENTION',historical_save_time_checkpoint_audits='PASS',missing_checkpoint_files=missing,PEG_FORMAL_TRAINING_ADJUDICATION='BLOCKED_ADDITIONAL_CHECKPOINT_RETENTION_BUG',PEG_FORMAL_EVALUATION='BLOCKED')
  removal=[line for line in (RUN/'learner.log').read_text(errors='replace').splitlines() if 'Removing checkpoint at' in line]
  write(OUT/'retention_discovery.json',dict(missing=missing,training_time_removal_log=removal,repair_performed=False))
 write(OUT/'adjudication.json',adjudication)
 gate=copy.deepcopy(old);gate.update(schema_version=4,exit_classification_fix='PASS',source_hashes={name:p.sha(ROOT/name) for name in p.SOURCES},validation_scope='F2 existing formal evidence and CPU exit-classification regression; no new training',formal_training='COMPLETED_WITH_WANDB_FINALIZATION_BUG',formal_training_started=True,training_started_this_task=False,evaluation_ready=True)
 if missing:gate.update(formal_ready=False,evaluation_ready=False,blocking_reason='checkpoint retention removed early bundle files',missing_checkpoint_files=missing)
 for path in (OLD,OUT/'tests.json',OUT/'protected.json',OUT/'adjudication.json',RUN/'post_audit.json',RUN/'result.json',RUN/'cloud_history.json',Path(raw['monitor_directory'])/'formal_monitor_updates.jsonl'):
  gate['evidence_hashes'][str(path.relative_to(ROOT))]=p.sha(path)
 candidate=OUT/'gate_candidate.json';write(candidate,gate)
 p.validate_runtime(True)
 with patch.object(p,'GATE',candidate):
  if missing:
   try:p.require_gate()
   except RuntimeError:pass
   else:raise AssertionError('blocked v4 must reject launch')
  else:
   p.require_gate()
   try:f.preflight('fresh_learner')
   except ValueError as exc:assert 'fresh output not clean' in str(exc)
   else:raise AssertionError('completed run outputs must reject fresh restart')
 assert all(p.sha(ROOT/name)==digest for name,digest in protected.items())
 write(p.GATE,gate)
 result=dict(PEG_F2_EXIT_CLASSIFICATION_FIX='PASS',PEG_F2_EXIT_TESTS='PASS',PEG_FORMAL_READINESS_V4='FAIL' if missing else 'PASS',**adjudication,gate_sha256=p.sha(p.GATE))
 write(OUT/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
