"""Synthetic CPU-only recorder tests; files stay in TemporaryDirectory."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import *

def binding():return {**{k:'a'*64 for k in HASHES},'files':{'synthetic':'b'*64}}
def state(t=0,success=False):
    return dict(observation_contract_version=V2,obs=[0,0,0,1,0,0,0,0,1,2,3,4,5,0,0,0,1,0,0,0],target_pose=[0,0,0,1,0,0,0],actual_ee_pose=[0,0,0,1,0,0,0],raw_sensordata=[0,1,2,3,4,5],
        sensors=[dict(name='end_force_sensor',id=0,adr=0,dim=3),dict(name='end_torque_sensor',id=1,adr=3,dim=3)],sim_time=t,
        metrics=dict(lateral_error=0 if success else .02,axial_error=0,orientation_error=0))
def row(success=False):
    before=state();after=state(.02,success)
    return dict(observation_contract_version=V2,episode_type='standard',phase='success' if success else 'setup',step=1,obs=before['obs'],action=[0]*7,next_obs=after['obs'],reward=27 if success else -1.6,terminated=success,truncated=False,success=success,before=before,after=after,hashes=binding())
class Validation(unittest.TestCase):
    def check_row(self,r):validate_transition(r,state(),1,'standard',binding())
    def test_valid(self):self.check_row(row())
    def test_v1(self):
        r=row();r['observation_contract_version']='PEG_OBS_CONTRACT_V1_LEGACY'
        with self.assertRaises(ValueError):self.check_row(r)
    def test_shape(self):
        for key,n in [('obs',19),('action',6)]:
            r=row();r[key]=r[key][:n]
            with self.subTest(key=key),self.assertRaises(ValueError):self.check_row(r)
    def test_nan(self):
        r=row();r['action'][0]=float('nan')
        with self.assertRaises(ValueError):self.check_row(r)
    def test_sensor_address(self):
        r=row();r['after']['sensors'][1]['adr']=1
        with self.assertRaises(ValueError):self.check_row(r)
    def test_sensor_dim(self):
        r=row();r['after']['sensors'][1]['dim']=2
        with self.assertRaises(ValueError):self.check_row(r)
    def test_raw_reconstruction(self):
        r=row();r['after']['raw_sensordata'][5]=123
        with self.assertRaises(ValueError):self.check_row(r)
    def test_full_raw(self):
        s=state();s['raw_sensordata'].append(99)
        with self.assertRaises(ValueError):validate_snapshot(s)
    def test_phase_required(self):
        r=row();del r['phase']
        with self.assertRaises(ValueError):self.check_row(r)
    def test_setup_not_skipped(self):
        r=row();r['phase']='insertion'
        with self.assertRaises(ValueError):self.check_row(r)
    def test_episode_type(self):
        r=row();r['episode_type']='other'
        with self.assertRaises(ValueError):self.check_row(r)
    def test_hash_required(self):
        b=binding();del b['env_hash']
        with self.assertRaises(ValueError):validate_binding(b)
    def test_hash_mismatch(self):
        r=row();r['hashes']['config_hash']='b'*64
        with self.assertRaises(ValueError):self.check_row(r)
    def test_continuity(self):
        r=row();r['before']['sim_time']=.001
        with self.assertRaises(ValueError):self.check_row(r)
    def test_step_gap(self):
        r=row();r['step']=2
        with self.assertRaises(ValueError):self.check_row(r)
    def test_time_gap(self):
        r=row();r['after']['sim_time']=.04
        with self.assertRaises(ValueError):self.check_row(r)
    def test_truncated_success(self):
        r=row(True);r['truncated']=True
        with self.assertRaises(ValueError):self.check_row(r)
    def test_false_success(self):
        r=row();r['success']=True
        with self.assertRaises(ValueError):self.check_row(r)
    def test_reward(self):
        r=row();r['reward']=0
        with self.assertRaises(ValueError):self.check_row(r)
class Persistence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'episode'
    def writer(self):
        w=EpisodeWriter(self.path,'standard',state(),binding(),binding);self.addCleanup(lambda: w.stream.close() if not w.stream.closed else None);return w
    def test_success_unapproved(self):
        w=self.writer();w.append(row(True));w.finish();s=json.loads((self.path/'status.json').read_text());self.assertEqual(s['status'],'COMPLETED_SUCCESS');self.assertFalse(s['eligible'])
    def test_dirty(self):
        self.path.mkdir()
        with self.assertRaises(ValueError):self.writer()
    def test_interruption(self):
        w=self.writer();w.append(row());w.abort('test interruption');s=json.loads((self.path/'status.json').read_text());self.assertEqual(s['status'],'INTERRUPTED');self.assertEqual(s['transitions'],1);self.assertFalse(s['eligible'])
    def test_hard_interruption_marker(self):
        w=self.writer();w.append(row());self.assertEqual(json.loads((self.path/'status.json').read_text())['status'],'IN_PROGRESS')
    def test_no_incomplete_finalize(self):
        w=self.writer();w.append(row())
        with self.assertRaises(ValueError):w.finish()
    def test_rejected_row_preserved(self):
        w=self.writer();r=row();r['phase']='invalid'
        with self.assertRaises(ValueError):w.append(r)
        self.assertTrue((self.path/'rejected_transition.json').exists())
    def test_end_hash_change(self):
        w=self.writer();w.append(row(True));w.verify=lambda:{}
        with self.assertRaises(ValueError):w.finish()
    def test_no_postterminal_append(self):
        w=self.writer();w.append(row(True))
        with self.assertRaises(ValueError):w.append(row())
if __name__=='__main__':unittest.main(verbosity=2)
