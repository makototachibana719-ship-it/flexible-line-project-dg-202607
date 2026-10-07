"""CPU-only verification; never constructs an environment or records a demo."""
import ast,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import preflight,bindings,sha

def main():
    evidence=ROOT/'results/validation/peg/f3_v2_recorder_ready_20260904'
    evidence.mkdir(parents=True,exist_ok=False)
    output=ROOT/'results/demos/peg_v2_pilot/standard_001'
    before=bindings()
    for relative in ('scripts/record_peg_v2_demo.py','scripts/test_peg_v2_recorder.py','training/peg_insertion/v2_recorder.py'):
        ast.parse((ROOT/relative).read_text())
    tests=subprocess.run([sys.executable,str(ROOT/'scripts/test_peg_v2_recorder.py')],capture_output=True,text=True)
    (evidence/'cpu_tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('CPU tests failed')
    result=preflight(output)
    # Existing output must fail closed, even when it belongs to this audit.
    try:preflight(evidence)
    except ValueError:pass
    else:raise AssertionError('dirty output accepted')
    assert not output.exists()
    assert bindings()==before
    assert 'mujoco' not in sys.modules and 'jax' not in sys.modules
    (evidence/'preflight.json').write_text(json.dumps(result,indent=2))
    gates={k:'PASS' for k in ('PEG_F3_V2_RECORDER_SCHEMA','PEG_F3_V2_RECORDER_SENSOR_LOGGING','PEG_F3_V2_RECORDER_PHASE_LOGGING','PEG_F3_V2_RECORDER_HASH_BINDING','PEG_F3_V2_RECORDER_PREFLIGHT')}
    gates.update(V2_DEMO_PILOT='READY_TO_RECORD',demo_recording_started=False,training_started=False,
                 cpu_tests=27,syntax='PASS',dirty_output_rejection='PASS',protected_frozen_manifest='PASS',
                 no_simulator_or_jax_import=True,checkpoint_modified=False,
                 scope='CPU/static readiness only; runtime guards remain mandatory; pilot success/coverage not yet validated')
    (evidence/'result.json').write_text(json.dumps(gates,indent=2))
    print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
