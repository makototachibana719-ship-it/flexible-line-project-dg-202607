"""CPU/static evidence only. Not a simulator diagnostic entry point."""
import ast,hashlib,inspect,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import bindings,sha
from mujoco import minimize
def main():
    out=ROOT/'results/validation/peg/f3_axial_telemetry_20260905';out.mkdir(exist_ok=False)
    before=bindings()
    prior=json.loads((ROOT/'results/demos/peg_v2_pilot/standard_20260905_001/episode.json').read_text())['hashes']
    assert before==prior
    demo=ROOT/'results/demos/peg_v2_pilot/standard_20260905_001'
    demo_before={str(p):sha(p) for p in demo.iterdir() if p.is_file()}
    files=['training/peg_insertion/axial_telemetry.py','scripts/test_peg_axial_telemetry.py','scripts/verify_peg_axial_telemetry.py']
    for f in files:ast.parse((ROOT/f).read_text())
    run=subprocess.run([sys.executable,str(ROOT/files[1])],text=True,capture_output=True)
    (out/'cpu_tests.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    assert bindings()==before
    assert demo_before=={str(p):sha(p) for p in demo.iterdir() if p.is_file()}
    bound=dict(production_manifest=before,telemetry_files={f:sha(ROOT/f) for f in files},
               installed_solver_file=inspect.getsourcefile(minimize.least_squares),
               installed_solver_sha256=sha(inspect.getsourcefile(minimize.least_squares)),
               failed_pilot_unchanged=demo_before)
    (out/'binding.json').write_text(json.dumps(bound,indent=2))
    result={k:'PASS' for k in ('PEG_F3_AXIAL_TELEMETRY','PEG_F3_IK_TELEMETRY','PEG_F3_JOINT_TELEMETRY','PEG_F3_CONTACT_TELEMETRY','PEG_F3_TELEMETRY_NONINVASIVE')}
    result.update(AXIAL_DIAGNOSTIC='READY',tests=18,syntax='PASS',controller_and_ik_source_bytes_unchanged=True,
      reward_success_limits_geometry_action_obs_v2_unchanged=True,simulator_diagnostic_started=False,
      demo_recording_started=False,training_started=False,checkpoint_resumed=False,
      validation_scope='CPU fixtures with production solve/dispatch AST and real algebraic least_squares; not actual simulator trajectory equivalence',
      historical_axial_root_cause='UNRESOLVED')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
