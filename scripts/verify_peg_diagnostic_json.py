"""No simulator imports or calls; CPU tests and protected hash verification."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import bindings,sha

def main():
    out=ROOT/'results/validation/peg/f3_diagnostic_json_20260905';out.mkdir(parents=True,exist_ok=False)
    b=bindings();reference=json.loads((ROOT/'results/demos/peg_v2_pilot/standard_20260905_001/episode.json').read_text())['hashes']
    assert b==reference
    gate=json.loads((ROOT/'results/validation/peg/f3_axial_telemetry_20260905/binding.json').read_text())
    assert all(sha(ROOT/f)==h for f,h in gate['telemetry_files'].items())
    assert sha(gate['installed_solver_file'])==gate['installed_solver_sha256']
    p=subprocess.run([sys.executable,str(ROOT/'scripts/test_peg_diagnostic_json.py')],capture_output=True,text=True)
    (out/'cpu_tests.log').write_text(p.stdout+p.stderr)
    print(p.stdout+p.stderr)
    assert p.returncode==0 and bindings()==b
    result=dict(PEG_F3_DIAGNOSTIC_JSON_NORMALIZATION='PASS',PEG_F3_NUMPY_SCALAR_SERIALIZATION='PASS',
        PEG_F3_ATOMIC_SNAPSHOT_WRITE='PASS',PEG_F3_SERIALIZATION_FAIL_CLOSED='PASS',BASELINE_CONTACT_REVIEW='READY',
        simulator_started=False,demo_recording_started=False,training_started=False,production_hashes_unchanged=True,
        telemetry_capture_unchanged=True,guard_and_action_ast_unchanged=True)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'binding.json').write_text(json.dumps(dict(production=b,telemetry=gate,runner_sha256=sha(ROOT/'scripts/run_peg_axial_bounded_diagnostic.py')),indent=2))
if __name__=='__main__':main()
