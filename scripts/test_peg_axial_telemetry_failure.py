"""CPU-only durable failure-evidence and noninvasiveness regressions."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_peg_axial_telemetry import Env, contact_force, minimize
from training.peg_insertion.axial_telemetry import AxialTelemetry, FAILURE_STAGES


class FailureEvidenceTests(unittest.TestCase):
    action=np.array([.1,.2,.3,1.,0.,0.,0.])

    def observer(self,root,env=None):
        env=env or Env();out=Path(root)/'telemetry'
        observer=AxialTelemetry(env,out,solver=minimize.least_squares,contact_force_reader=contact_force)
        observer.set_context(phase='setup')
        self.addCleanup(observer.close)
        return env,observer,out

    def artifact(self,out):
        files=list(out.glob('failure_step_*.json'))
        self.assertEqual(len(files),1)
        return json.loads(files[0].read_text())

    def test_success_row_schema_and_numpy_serialization(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            original=o._snapshot
            def snapshot():
                value=original();value['numpy_fixture']={'b':np.bool_(True),'i':np.int64(7),'f':np.float32(.25),'a':np.array([[1,2],[3,4]])}
                return value
            o._snapshot=snapshot
            returned=o.step(self.action);o.close()
            row=json.loads((out/'steps.jsonl').read_text())
            self.assertTrue(row['complete']);self.assertEqual(row['numpy_fixture'] if 'numpy_fixture' in row else row['pre']['numpy_fixture'],{'b':True,'i':7,'f':.25,'a':[[1,2],[3,4]]})
            self.assertEqual(env.calls,1);self.assertEqual(returned[1],1.)
            self.assertEqual(list(out.glob('failure_step_*.json')),[])

    def test_failure_stage_vocabulary_is_frozen(self):
        self.assertEqual(FAILURE_STAGES,('RAW_STEP','IK_CAPTURE','JOINT_CAPTURE','ACTUATOR_CAPTURE','CONTACT_CAPTURE',
            'POSTPROCESS','JSON_NORMALIZE','SERIALIZE','WRITE','FSYNC','PUBLISH'))

    def test_postprocess_exception_is_exact_and_durable(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            with patch.object(o,'_singular_values',side_effect=ArithmeticError('svd fixture')):
                returned=o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'POSTPROCESS')
            self.assertEqual(failure['last_successful_stage'],'CONTACT_CAPTURE')
            self.assertEqual(failure['exception_type'],'ArithmeticError')
            self.assertEqual(failure['exception_message'],'svd fixture')
            self.assertIn('svd fixture',failure['traceback_summary'])
            self.assertFalse(failure['complete']);self.assertEqual(failure['phase'],'setup')
            self.assertTrue(failure['availability']['raw_step']);self.assertEqual(env.calls,1);self.assertEqual(returned[1],1.)

    def test_json_normalize_exception_classified(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            with patch.object(o,'_normalize_row',side_effect=TypeError('unsupported fixture')):
                o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'JSON_NORMALIZE')
            self.assertEqual(failure['last_successful_stage'],'POSTPROCESS')
            self.assertEqual(failure['exception_type'],'TypeError');self.assertEqual(env.calls,1)

    def test_serialize_exception_classified(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            with patch.object(o,'_serialize_row',side_effect=ValueError('serialize fixture')):
                o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'SERIALIZE')
            self.assertEqual(failure['last_successful_stage'],'JSON_NORMALIZE')
            self.assertEqual(failure['exception_message'],'serialize fixture');self.assertEqual(env.calls,1)

    def test_write_exception_classified(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            with patch.object(o,'_write_step_payload',side_effect=OSError('write fixture')):
                o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'WRITE');self.assertEqual(failure['last_successful_stage'],'SERIALIZE');self.assertEqual(env.calls,1)

    def test_fsync_exception_classified_and_durable(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            with patch.object(o,'_step_fsync',side_effect=OSError('fsync fixture')):
                o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'FSYNC')
            self.assertEqual(failure['last_successful_stage'],'WRITE')
            self.assertEqual(failure['exception_message'],'fsync fixture');self.assertEqual(env.calls,1)

    def test_publish_failure_is_explicitly_not_durable(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            with patch.object(o,'_singular_values',side_effect=RuntimeError('primary fixture')),patch.object(o,'_failure_publish',side_effect=OSError('publish fixture')):
                o.step(self.action)
            self.assertFalse(o.last['failure_evidence_durable'])
            self.assertEqual(o.last['failure_artifact_error']['failure_stage'],'PUBLISH')
            self.assertEqual(list(out.glob('failure_step_*.json')),[]);self.assertEqual(env.calls,1)

    def test_failure_publish_is_atomic_exclusive_and_directory_synced(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root);events=[]
            def publish(temp,final):
                self.assertFalse(Path(final).exists())
                staged=json.loads(Path(temp).read_text())
                self.assertEqual(staged['exception_message'],'atomic fixture')
                events.append('publish');os.link(temp,final)
            real_sync=o._sync_directory
            def sync(path):events.append('dir_fsync');return real_sync(path)
            with patch.object(o,'_singular_values',side_effect=RuntimeError('atomic fixture')),patch.object(o,'_failure_publish',side_effect=publish),patch.object(o,'_sync_directory',side_effect=sync):
                o.step(self.action)
            self.assertTrue(o.last['failure_evidence_durable']);self.assertEqual(events[0],'publish')
            self.assertIn('dir_fsync',events);self.assertEqual(env.calls,1)

    def test_existing_failure_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root);target=out/'failure_step_000001.json';target.write_text('sentinel')
            with patch.object(o,'_singular_values',side_effect=RuntimeError('new fixture')):
                o.step(self.action)
            self.assertEqual(target.read_text(),'sentinel')
            self.assertFalse(o.last['failure_evidence_durable'])
            self.assertEqual(o.last['failure_artifact_error']['failure_stage'],'PUBLISH');self.assertEqual(env.calls,1)

    def test_missing_ik_field_classification(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            o.codes['solve_ik']=None;o.codes['least_squares']=None
            o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'IK_CAPTURE')
            self.assertIn('ik',failure['missing_fields']);self.assertFalse(failure['availability']['ik'])
            self.assertEqual(env.calls,1)

    def test_joint_contact_actuator_capture_stages_block_raw_step(self):
        for method,stage in (('_joints','JOINT_CAPTURE'),('_actuators','ACTUATOR_CAPTURE'),('_contacts','CONTACT_CAPTURE')):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as root:
                env,o,out=self.observer(root)
                with patch.object(o,method,side_effect=RuntimeError(stage+' fixture')):
                    with self.assertRaisesRegex(RuntimeError,stage+' fixture'):o.step(self.action)
                failure=self.artifact(out)
                self.assertEqual(failure['failure_stage'],stage);self.assertFalse(failure['availability']['raw_step'])
                self.assertEqual(env.calls,0)

    def test_raw_step_exception_preserved_and_durable(self):
        with tempfile.TemporaryDirectory() as root:
            env,o,out=self.observer(root)
            def fail(action):env.calls+=1;raise LookupError('raw fixture')
            env.step=fail
            with self.assertRaisesRegex(LookupError,'raw fixture'):o.step(self.action)
            failure=self.artifact(out)
            self.assertEqual(failure['failure_stage'],'RAW_STEP')
            self.assertEqual(failure['exception_type'],'LookupError');self.assertEqual(env.calls,1)

    def test_on_off_outputs_and_solver_call_counts_bit_exact(self):
        off,on=Env(),Env();action=self.action.copy()
        off_result=AxialTelemetry(off,enabled=False).step(action)
        with tempfile.TemporaryDirectory() as root:
            observer=AxialTelemetry(on,Path(root)/'telemetry',solver=minimize.least_squares,contact_force_reader=contact_force)
            observer.set_context(phase='setup');on_result=observer.step(action);observer.close()
        np.testing.assert_array_equal(off_result[0],on_result[0]);self.assertEqual(off_result[1:],on_result[1:])
        np.testing.assert_array_equal(off.env.data.ctrl,on.env.data.ctrl)
        self.assertEqual(off.calls,on.calls);self.assertEqual(off.env.controller.res_calls,on.env.controller.res_calls)
        self.assertEqual(off.env.controller.jac_calls,on.env.controller.jac_calls)


if __name__=='__main__':unittest.main(verbosity=2)
