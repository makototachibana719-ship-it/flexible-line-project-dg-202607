"""CPU/static verification only. Never imports or constructs the simulator environment."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import bindings

OUT=ROOT/'results/validation/peg/f3_axial_telemetry_failure_evidence_20260905'
EXPECTED_BINDINGS={
    'source_hash':'38837053932d3c8049ecf8a4ac4e0f8501e4afb401a00cff635f1997ce97eb94',
    'config_hash':'419fd4c795f481563ab68f1d3f1fed2fd4251ea050c8f2c400ec8a72592da67b',
    'runtime_hash':'ed29e50af02dd02b5583850ad19ec4572908c4758f2d4b8b80c3ef0d4fcb23d1',
    'env_hash':'3599b698902c336871779a9d265f86c42dcc40ef9b3a70e7d5f7b2676ef2c081',
    'recorder_hash':'3807b6a61885501a0540c61237032284fe51624614ede583618e89b4c1e254b6'}
PROTECTED={
    'external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/controllers/cartesian_ik.py':'048974d7e6166ab15c2aa2857f281873cc15ea30c7c12185ae479610b560025e',
    'external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/controllers/joint_position.py':'0bc3619c6f7562bbfa13d833847dbfcf390751389aa03facc10e60a7c8a705be',
    'external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/assets/models/manipulators/Aubo_i5/Aubo_i5.xml':'b6ce6ec7683ed68951fa6ac603d8975ebf85e3f66edf06516ad2538def4c76c5',
    'runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml':'9e3662138e40d5bae75e1dd29607c3a87a9dc693cfa7add7c82237527c146477'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def function(tree,name):return next(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    tests=[('legacy','scripts/test_peg_axial_telemetry.py',18),
           ('failure','scripts/test_peg_axial_telemetry_failure.py',14),
           ('contract','scripts/test_peg_axial_diagnostic_contract.py',12)]
    test_results={}
    for name,file,count in tests:
        run=subprocess.run([sys.executable,str(ROOT/file)],text=True,capture_output=True)
        (OUT/(name+'_tests.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==0,run.stdout+run.stderr
        assert ('Ran %d test' % count) in run.stderr
        test_results[name]=count
    current={key:bindings()[key] for key in EXPECTED_BINDINGS}
    assert current==EXPECTED_BINDINGS
    protected={path:sha(ROOT/path) for path in PROTECTED}
    assert protected==PROTECTED
    telemetry=ROOT/'training/peg_insertion/axial_telemetry.py'
    runner=ROOT/'scripts/run_peg_axial_bounded_diagnostic_retry.py'
    old_runner=Path(str(runner)+'.before_failure_evidence_20260905')
    telemetry_tree=ast.parse(telemetry.read_text());runner_tree=ast.parse(runner.read_text());old_tree=ast.parse(old_runner.read_text())
    cls=function(telemetry_tree,'AxialTelemetry');step=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='step')
    raw_step_calls=[node for node in ast.walk(step) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='step']
    assert len(raw_step_calls)==2  # enabled original env.step plus disabled pass-through branch; either path calls once
    main_node=function(runner_tree,'main')
    observer_steps=[node for node in ast.walk(main_node) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='step']
    assert len(observer_steps)==1
    for name in ('validate_post_exclusion_invariants','run_pre_action_boundary'):
        assert ast.dump(function(runner_tree,name))==ast.dump(function(old_tree,name))
    assert "observer.set_context(phase=historical[\"phase\"])" in runner.read_text()
    result={
      'PEG_F3_TELEMETRY_FAILURE_EVIDENCE':'PASS',
      'PEG_F3_TELEMETRY_FAILURE_STAGE_CLASSIFICATION':'PASS',
      'PEG_F3_TELEMETRY_FAILURE_PERSISTENCE':'PASS',
      'PEG_F3_TELEMETRY_NONINVASIVE_REGRESSION':'PASS',
      'AXIAL_DIAGNOSTIC':'READY',
      'tests':sum(test_results.values()),'test_groups':test_results,'syntax':'PASS',
      'failure_stages':['RAW_STEP','IK_CAPTURE','JOINT_CAPTURE','ACTUATOR_CAPTURE','CONTACT_CAPTURE','POSTPROCESS','JSON_NORMALIZE','SERIALIZE','WRITE','FSYNC','PUBLISH'],
      'normal_steps_schema_preserved':True,'single_original_step_per_path':True,
      'production_bindings_unchanged':True,'protected_hashes_unchanged':True,
      'simulator_started':False,'demo_recording_started':False,'training_started':False,
      'diagnostic_rerun_started':False}
    binding={'production_bindings':current,'protected':protected,
      'telemetry_before_sha256':sha(str(telemetry)+'.before_failure_evidence_20260905'),
      'telemetry_after_sha256':sha(telemetry),'runner_before_sha256':sha(old_runner),
      'runner_after_sha256':sha(runner),'failure_test_sha256':sha(ROOT/'scripts/test_peg_axial_telemetry_failure.py')}
    (OUT/'binding.json').write_text(json.dumps(binding,indent=2))
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
