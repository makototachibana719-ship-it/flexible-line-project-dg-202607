"""CPU-only verification and immutable evidence report; never calls runner.main."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import bindings,sha

def main():
    out=ROOT/'results/validation/peg/f3_init_evidence_fix_20260905'
    out.mkdir(parents=True,exist_ok=False)
    before=bindings()
    reference=json.loads((ROOT/'results/demos/peg_v2_pilot/standard_20260905_001/episode.json').read_text())['hashes']
    assert before==reference
    gate=json.loads((ROOT/'results/validation/peg/f3_axial_telemetry_20260905/binding.json').read_text())
    assert all(sha(ROOT/f)==h for f,h in gate['telemetry_files'].items())
    assert sha(gate['installed_solver_file'])==gate['installed_solver_sha256']
    p=subprocess.run([sys.executable,str(ROOT/'scripts/test_peg_initial_evidence.py')],capture_output=True,text=True)
    (out/'cpu_tests.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0
    assert bindings()==before
    result=dict(PEG_F3_INITIAL_EVIDENCE_PERSISTENCE='PASS',PEG_F3_INITIAL_GUARD_RECORDING='PASS',
                PEG_F3_DIAGNOSTIC_RUNNER_NONINVASIVE='PASS',BASELINE_CONTACT_REVIEW='READY',
                simulator_diagnostic_started=False,demo_recording_started=False,training_started=False,
                production_hashes_unchanged=True,telemetry_hashes_unchanged=True,cpu_tests=7,
                baseline_contacts_available=False,root_cause='UNRESOLVED')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'binding.json').write_text(json.dumps(dict(production=before,telemetry=gate,
        runner_sha256=sha(ROOT/'scripts/run_peg_axial_bounded_diagnostic.py'),
        test_sha256=sha(ROOT/'scripts/test_peg_initial_evidence.py')),indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
