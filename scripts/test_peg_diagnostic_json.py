"""CPU-only NumPy serialization and no-clobber persistence regressions."""
import ast,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_peg_axial_bounded_diagnostic as runner
from test_peg_initial_evidence import InitialEvidenceTests

class NumpyPersistenceTests(InitialEvidenceTests):
    def test_real_shaped_roundtrip(self):
        self.state['qpos']=np.array([.1,.2]);self.state['qvel']=np.array([0.,1.])
        self.state['ctrl']=np.array([.3])
        self.state['joints'][0].update(controller_arm_member=np.bool_(True),range=np.array([-1.,1.]),id=np.int32(0))
        self.contact.update(geom1_id=np.int64(3),distance=np.float64(-.006123),
                            position=np.array([1,2,3],dtype=np.float32),extra=[np.bool_(False),{'x':(np.float32(.25),np.int32(7))}])
        with self.assertRaisesRegex(RuntimeError,'dangerous'):self.dispatch()
        r=json.loads((self.out/'initial_snapshot.json').read_text())
        self.assertIs(r['joints'][0]['controller_arm_member'],True)
        self.assertIs(type(r['joints'][0]['id']),int)
        self.assertEqual(r['qpos'],[.1,.2]);self.assertEqual(r['ctrl'],[.3])
        self.assertEqual(r['contacts'][0]['extra'],[False,{'x':[.25,7]}])
        self.assertEqual(r['contacts'][0]['distance'],float(self.contact['distance']))
        self.observer.step.assert_not_called()
    def test_invalid_values_block_guard(self):
        for value in (np.nan,np.inf,-np.inf,np.float32('nan'),np.array([np.inf]),object()):
            with self.subTest(value=repr(value)):
                self.initial['invalid']=value
                with patch.object(runner,'initial_guard') as guard:
                    with self.assertRaises((TypeError,ValueError)):self.dispatch()
                    guard.assert_not_called();self.observer.step.assert_not_called()
                self.assertFalse((self.out/'initial_snapshot.json').exists())
        self.assertEqual(json.loads((self.out/'initial_snapshot.json.failure.json').read_text())['status'],'PERSISTENCE_FAILED')
    def test_atomic_publish_sees_complete_json(self):
        real=runner.os.link
        def link(src,dst):
            self.assertFalse(Path(dst).exists())
            self.assertEqual(json.loads(Path(src).read_text()),{'x':[1,True]})
            real(src,dst)
        with patch.object(runner.os,'link',side_effect=link):
            runner.durable_save(self.out/'data.json',{'x':[np.int64(1),np.bool_(True)]})
        self.assertEqual(list(self.out.glob('*.tmp')),[])
    def test_publish_race_never_overwrites(self):
        def race(src,dst):Path(dst).write_text('existing');raise FileExistsError(str(dst))
        with patch.object(runner.os,'link',side_effect=race):
            with self.assertRaises(FileExistsError):runner.durable_save(self.out/'data.json',{})
        self.assertEqual((self.out/'data.json').read_text(),'existing')
    def test_guard_and_action_ast_unchanged_since_fix(self):
        p=Path(runner.__file__)
        old=ast.parse(Path(str(p)+'.before_serialization_20260905').read_text());new=ast.parse(p.read_text())
        for name in ('initial_guard','persist_initial_evidence_and_guard','main'):
            def fn(t):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(fn(old)),ast.dump(fn(new)))

if __name__=='__main__':unittest.main(verbosity=2)
