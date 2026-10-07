"""CPU-only persistence/guard tests; no simulator initialization."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import run_peg_axial_bounded_diagnostic as runner

class InitialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)
        self.contact=dict(geom1_id=3,geom1_name='robot',geom2_id=8,geom2_name='fixture',
                          distance=-.006123,position=[1,2,3],normal=[1,0,0],
                          contact_force_torque_local=[1,2,3,4,5,6],force_status='available')
        self.state=dict(qpos=[.1],qvel=[.2],ctrl=[.3],
                        joints=[dict(name='joint',qpos_address=0,qvel_address=0,range=[-1,1],distance_lower=1.1,distance_upper=.9)],
                        actuators=[dict(ctrl=.3,ctrlrange=[-1,1],clip_flag=False)],contacts=[self.contact])
        self.observer=Mock();self.observer._snapshot.side_effect=lambda:copy.deepcopy(self.state)
        self.initial=dict(target_pose=[0,0,0,1,0,0,0])
    def dispatch(self):
        runner.persist_initial_evidence_and_guard(self.observer,self.initial,self.out)
        self.observer.step([0]*7)
    def test_exact_event_and_zero_actions(self):
        with self.assertRaisesRegex(RuntimeError,'dangerous initial penetration'):self.dispatch()
        self.observer.step.assert_not_called()
        event=json.loads((self.out/'initial_guard_event.json').read_text())
        self.assertEqual(event['triggering_pair'],self.contact)
        self.assertEqual(event['exact_distance'],self.contact['distance'])
        self.assertEqual(event['threshold'],-.005);self.assertTrue(event['pre_action'])
        self.assertEqual(event['snapshot_sha256'],runner.sha(self.out/'initial_snapshot.json'))
    def test_fsync_before_guard_and_event_fsync(self):
        real_guard=runner.initial_guard;real_fsync=runner.os.fsync;events=[]
        def sync(fd):real_fsync(fd);events.append('fsync')
        def guard(contacts):
            self.assertIn('fsync',events)
            saved=json.loads((self.out/'initial_snapshot.json').read_text())
            for key in self.state:self.assertEqual(saved[key],self.state[key])
            events.append('guard');return real_guard(contacts)
        with patch.object(runner.os,'fsync',side_effect=sync),patch.object(runner,'initial_guard',side_effect=guard):
            with self.assertRaises(RuntimeError):self.dispatch()
        self.assertEqual(events[-1],'fsync')
    def test_boundary_pass_dispatches_once(self):
        self.contact['distance']=-.005
        self.dispatch();self.observer.step.assert_called_once_with([0]*7)
        self.assertFalse((self.out/'initial_guard_event.json').exists())
    def test_failed_persistence_prevents_guard_and_action(self):
        with patch.object(runner.os,'fsync',side_effect=OSError('disk')),patch.object(runner,'initial_guard') as guard:
            with self.assertRaises(OSError):self.dispatch()
            guard.assert_not_called();self.observer.step.assert_not_called()
    def test_existing_snapshot_not_overwritten(self):
        (self.out/'initial_snapshot.json').write_text('original')
        with self.assertRaises(FileExistsError):self.dispatch()
        self.assertEqual((self.out/'initial_snapshot.json').read_text(),'original')
        self.observer.step.assert_not_called()
    def test_action_loop_ast_unchanged(self):
        path=Path(runner.__file__)
        old=ast.parse(Path(str(path)+'.before_init_evidence_20260905').read_text())
        new=ast.parse(path.read_text())
        def loop(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='historical')
        self.assertEqual(ast.dump(loop(old)),ast.dump(loop(new)))
    def test_snapshot_does_not_mutate_source(self):
        before=copy.deepcopy(self.state)
        with self.assertRaises(RuntimeError):self.dispatch()
        self.assertEqual(self.state,before)

if __name__=='__main__':unittest.main(verbosity=2)
