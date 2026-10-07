"""CPU tests: no MuJoCo instance, JAX, policy evaluation, or training."""
import ast,copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion import observation_contract as oc
PKG=ROOT/'external/packages/peg_insertion/peg_insertion'
SENSOR=PKG/'code/env/mujoco_env/mujoco_env/utils/sensor_utils.py'
ADAPTER=PKG/'code/example/async_rlpd_peg_sim/peg_env_adapter.py'
spec=importlib.util.spec_from_file_location('sensor_under_test',SENSOR);sensor=importlib.util.module_from_spec(spec);spec.loader.exec_module(sensor)

def model(adr=(0,3),dim=(3,3),ids=(0,1)):
    return SimpleNamespace(sensor_adr=np.array(adr),sensor_dim=np.array(dim),sensor=lambda name:SimpleNamespace(id=ids[0] if name=='end_force_sensor' else ids[1]))
def monitor(m=None):return sensor.ForceSensorMonitor(m or model(),SimpleNamespace(sensordata=np.arange(20,dtype=float)))
def function(path,name,namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

class SensorTests(unittest.TestCase):
    def test_id_equals_address(self):np.testing.assert_array_equal(monitor().get_force_data(),[0,1,2])
    def test_id_not_address(self):np.testing.assert_array_equal(monitor().get_torque_data(),[3,4,5])
    def test_noncontiguous(self):
        m=monitor(model((2,11)));np.testing.assert_array_equal(m.get_force_data(),[2,3,4]);np.testing.assert_array_equal(m.get_torque_data(),[11,12,13])
    def test_reordered(self):np.testing.assert_array_equal(monitor(model((10,1))).get_wrench_data(),[10,11,12,1,2,3])
    def test_invalid_id(self):
        for i in (-1,2,True,0.5):
            with self.subTest(i=i),self.assertRaises(ValueError):monitor(model(ids=(0,i)))
    def test_wrong_dimension(self):
        for dims in ((2,3),(3,4),(3,0)):
            with self.subTest(dims=dims),self.assertRaises(ValueError):monitor(model(dim=dims))
    def test_bad_address(self):
        for adr in ((-1,3),(0,19)):
            with self.subTest(adr=adr),self.assertRaises(ValueError):monitor(model(adr=adr))
    def test_missing_sensor(self):
        def missing(n):raise KeyError(n)
        with self.assertRaises(ValueError):monitor(SimpleNamespace(sensor=missing))
    def test_copy(self):
        m=monitor();r=m.get_torque_data();r[:]=999;np.testing.assert_array_equal(m.get_torque_data(),[3,4,5])
    def test_shape20(self):
        tree=ast.parse(ADAPTER.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_state');method.decorator_list=[]
        ns={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),'adapter','exec'),ns)
        result=ns['_state'](dict(tcp_pos=np.zeros(3),tcp_quat=[1,0,0,0],wrench=monitor().get_wrench_data(),desired_goal=[1,2,3,1,0,0,0]))
        self.assertEqual(result.shape,(20,));np.testing.assert_array_equal(result[10:13],[3,4,5])

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.cp=Path(self.tmp.name)/'checkpoint_1';self.cp.write_bytes(b'synthetic model')
        self.digest=hashlib.sha256(self.cp.read_bytes()).hexdigest();self.reg=Path(self.tmp.name)/'registry.json';self.reg.write_text(json.dumps({'checkpoint_sha256':{self.digest:oc.V1}}))
        p=patch.object(oc,'REGISTRY',self.reg);p.start();self.addCleanup(p.stop)
    def test_legacy_label(self):self.assertEqual(oc.checkpoint_version(self.cp),oc.V1)
    def test_mismatch_rejected(self):
        with self.assertRaises(ValueError):oc.checkpoint_preflight(self.cp)
    def test_explicit_diagnostic(self):
        result=oc.checkpoint_preflight(self.cp,compatibility_diagnostic=True);self.assertFalse(result['formal_skill_result'])
    def test_unknown_rejected(self):
        self.cp.write_bytes(b'unknown')
        with self.assertRaises(ValueError):oc.checkpoint_preflight(self.cp,compatibility_diagnostic=True)
    def test_unknown_version(self):
        with self.assertRaises(ValueError):oc.require_match('V3')
    def test_no_relabel(self):
        self.cp.with_name(self.cp.name+'_resume_metadata.json').write_text(json.dumps({'observation_contract_version':oc.V2}))
        with self.assertRaises(ValueError):oc.checkpoint_version(self.cp)
    def test_v2_hash_bound(self):
        self.reg.write_text('{"checkpoint_sha256":{}}');self.cp.with_name(self.cp.name+'_resume_metadata.json').write_text(json.dumps(dict(observation_contract_version=oc.V2,observation={'contract_version':oc.V2},model_sha256=self.digest)))
        self.assertTrue(oc.checkpoint_preflight(self.cp)['formal_skill_result'])
    def test_hash_corruption(self):
        self.reg.write_text('{"checkpoint_sha256":{}}');self.cp.with_name(self.cp.name+'_resume_metadata.json').write_text(json.dumps(dict(observation_contract_version=oc.V2,observation={'contract_version':oc.V2},model_sha256='bad')))
        with self.assertRaises(ValueError):oc.checkpoint_version(self.cp)
    def test_training_demo_mismatch(self):
        cfg=yaml.safe_load((ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml').read_text());self.assertEqual(oc.validate_config(cfg),oc.V2)
        with self.assertRaises(ValueError):oc.validate_config(cfg,for_training=True)
    def test_demo_no_relabel(self):
        self.reg.write_text(json.dumps({'demo_sha256':{'original':oc.V1}}))
        cfg={'observation':{'contract_version':oc.V2},'demo':{'sha256':'original','observation_contract_version':oc.V2}}
        with self.assertRaisesRegex(ValueError,'cannot be relabeled'):oc.validate_config(cfg,for_training=True)
    def test_restore_rejects_before_flax(self):
        fn=function(ROOT/'training/peg_insertion/checkpoint.py','restore_fresh_state',{'Path':Path,'Any':object})
        with self.assertRaises(ValueError):fn(self.cp.parent,None,1)
    def test_load_bundle_rejects_before_replay(self):
        fn=function(ROOT/'training/peg_insertion/resume.py','load_bundle',{'Path':Path})
        cfg={'observation':{'contract_version':oc.V2},'demo':{'observation_contract_version':oc.V2}}
        with self.assertRaises(ValueError):fn(self.cp,cfg,Path('unused'),None)
    def test_preflight_rejects_before_gate(self):
        from training.peg_insertion.formal_preflight import preflight
        for role in ('fresh_learner','actor_join','resume_learner','resume_actor'):
            with self.subTest(role=role),self.assertRaisesRegex(ValueError,'observation contract mismatch'):preflight(role)
    def test_evaluator_rejects_before_environment(self):
        import argparse
        cfg=ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml'
        fn=function(ROOT/'evaluation/peg_insertion/evaluate_policy.py','main',{'argparse':argparse,'Path':Path,'CFG':cfg,'ROOT':ROOT,'yaml':yaml})
        with patch.object(sys,'argv',['eval','--checkpoint',str(self.cp)]),self.assertRaisesRegex(ValueError,'observation contract mismatch'):fn()

class RegressionTests(unittest.TestCase):
    def test_config_semantics(self):
        p=ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml';new=yaml.safe_load(p.read_text());old=yaml.safe_load(Path(str(p)+'.before_f3_sensorfix_20260904').read_text())
        del new['observation']['contract_version'];del new['demo']['observation_contract_version'];self.assertEqual(new,old)
        self.assertEqual(new['training']['max_steps'],10000);self.assertEqual(new['training']['checkpoint_period'],1000)
    def test_adapter_functions_unchanged(self):
        def methods(p):
            cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef))
            return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(methods(ADAPTER),methods(Path(str(ADAPTER)+'.before_f3_sensorfix_20260904')))
    def test_syntax(self):
        for p in ROOT.rglob('*.before_f3_sensorfix_20260904'):
            original=Path(str(p).removesuffix('.before_f3_sensorfix_20260904'))
            if original.suffix=='.py':ast.parse(original.read_text())
    def test_new_metadata_version(self):
        source=(ROOT/'training/peg_insertion/resume.py').read_text()
        self.assertIn("payload['observation_contract_version']=cfg['observation']['contract_version']",source)

if __name__=='__main__':unittest.main(verbosity=2)
