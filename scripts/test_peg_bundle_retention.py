"""CPU serialization/retention regression. All writes use temporary fixtures."""
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.peg_insertion import bundle_retention as b, resume as r, checkpoint as c

ROOT=Path(__file__).resolve().parents[1]

class Replay:
    def __init__(self):
        self._lock=threading.RLock()
        self.project_original_insert=True
        self.project_actor_steps=3
        self._capacity=8
        self._insert_index=3
        self.dataset_dict={'observations':np.arange(8,dtype=np.float32)}
        self.np_random=np.random.default_rng(7)
    def __len__(self):return 3

class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.directory=self.root/'bundles'
        self.manifest=self.root/'manifest.json';self.manifest.write_text('{}')
        self.cfg=dict(seed=0,environment='CPU_fixture',algorithm='fixture',observation={},action={},
                      demo=dict(sha256='fixture',transitions=3),wandb=dict(project='fixture',run_name='fixture'))
        self.source={'binding':'unchanged fixture'}
        self.replay=Replay();self.state={'params':np.array([1.,2.],dtype=np.float32)}
    def save(self,step,keep=20):
        return r.save_bundle(self.directory,self.state,self.replay,step,3,self.cfg,self.manifest,'fixture',self.source,keep_bundles=keep)
    def check(self,steps):
        self.assertEqual(b.inventory(self.directory),steps)
        self.assertEqual(len(list(self.directory.glob('checkpoint_*'))),3*len(steps))
        for step in steps:
            cp,rp,meta=b.paths(self.directory,step)
            m=json.loads(meta.read_text())
            self.assertEqual(m['model_sha256'],r.sha256_file(cp))
            self.assertEqual(m['online_replay_sha256'],r.sha256_file(rp))
            self.assertEqual(m['source_snapshot'],self.source)
            self.assertEqual(m['config_hash'],r.canonical_hash(self.cfg))
            self.assertEqual(m['demo_sha256'],self.cfg['demo']['sha256'])
            self.assertEqual(m['learner_update_steps'],step)
    def test_ten_actual_flax_bundles_keep_twenty(self):
        original=c._legacy_save_checkpoint
        def isolated(directory,*args,**kwargs):
            self.assertTrue(Path(directory).name.startswith('.bundle_stage_'))
            self.assertEqual(list(Path(directory).glob('checkpoint_*')),[])
            self.assertEqual(kwargs['keep'],20)
            return original(directory,*args,**kwargs)
        with patch.object(c,'_legacy_save_checkpoint',side_effect=isolated) as spy:
            for step in range(0,10000,1000):self.save(step)
            self.assertEqual(spy.call_count,10)
        self.check(list(range(0,10000,1000)))
    def test_twentyfive_actual_flax_bundles_keep_twenty(self):
        for step in range(0,25000,1000):self.save(step)
        self.check(list(range(5000,25000,1000)))
        for step in range(0,5000,1000):
            self.assertTrue(all(not p.exists() for p in b.paths(self.directory,step)))
        self.assertFalse(list(self.directory.glob('.bundle_retired_*')))
    def test_numeric_global_order(self):
        for step in (2,10,1000,9000):self.save(step,2)
        self.check([1000,9000])
        with self.assertRaises(ValueError):self.save(100)
        self.check([1000,9000])
    def test_retained_bytes_and_hashes_unchanged(self):
        for step in (0,1000,2000):self.save(step)
        before={p.name:p.read_bytes() for step in (1000,2000) for p in b.paths(self.directory,step)}
        self.assertEqual(b.retain_bundles(self.directory,2),[0])
        self.assertEqual(before,{name:(self.directory/name).read_bytes() for name in before})
        self.check([1000,2000])
    def test_incomplete_preexisting_rejected_without_mutation(self):
        self.save(0);b.paths(self.directory,0)[1].unlink()
        before={p.name:p.read_bytes() for p in self.directory.glob('checkpoint_*')}
        with self.assertRaisesRegex(ValueError,'incomplete'):self.save(1000)
        with self.assertRaisesRegex(ValueError,'incomplete'):b.retain_bundles(self.directory,1)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.directory.glob('checkpoint_*')})
    def test_failed_serialization_never_publishes_partial(self):
        self.save(0)
        with patch.object(r,'write_replay_snapshot',side_effect=OSError('fixture disk error')):
            with self.assertRaises(OSError):self.save(1000)
        self.check([0])
        self.assertTrue(list(self.directory.glob('.bundle_stage_*')))
    def test_publish_rename_exception_rolls_back(self):
        self.save(0);original=os.rename;calls=[]
        def fail_second(src,dst):
            calls.append((src,dst))
            if len(calls)==2:raise OSError('fixture rename')
            return original(src,dst)
        with patch.object(b.os,'rename',side_effect=fail_second):
            with self.assertRaises(OSError):self.save(1000)
        self.check([0]);self.assertFalse((self.directory/b.JOURNAL).exists())
    def test_retire_exception_rolls_back_all_members(self):
        for step in (0,1000):self.save(step)
        original=os.rename;calls=[]
        def fail_second(src,dst):
            calls.append((src,dst))
            if len(calls)==2:raise OSError('fixture retire')
            return original(src,dst)
        with patch.object(b.os,'rename',side_effect=fail_second):
            with self.assertRaises(OSError):b.retain_bundles(self.directory,1)
        self.check([0,1000])
    def test_hard_interruption_journal_fails_closed(self):
        self.save(0);(self.directory/b.JOURNAL).write_text('[]')
        with self.assertRaises(RuntimeError):b.inventory(self.directory)
        with self.assertRaises(RuntimeError):self.save(1000)
        with self.assertRaises(RuntimeError):r.load_bundle(b.paths(self.directory,0)[0],self.cfg,self.manifest,'fixture')
    def test_serialization_and_binding_body_unchanged(self):
        old=(ROOT/'training/peg_insertion/resume.py.before_f2_retention_20260904').read_text()
        new=(ROOT/'training/peg_insertion/resume.py').read_text()
        old_body=old.split('def save_bundle(',1)[1].split('\n',1)[1].split('\ndef load_bundle(',1)[0]
        new_body=new.split('def _save_bundle_staged(',1)[1].split('\n',1)[1].split('\ndef load_bundle(',1)[0]
        self.assertEqual(old_body,new_body)
        cfg=yaml.safe_load((ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml').read_text())
        self.assertEqual(cfg['training']['max_steps'],10000)
        self.assertEqual(cfg['training']['checkpoint_period'],1000)

if __name__=='__main__':unittest.main(verbosity=2)
