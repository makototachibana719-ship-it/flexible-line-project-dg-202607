"""Deterministic bounded concurrency regressions; no GPU training or W&B."""
import copy
import json
import pickle
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import gym
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serl_launcher.data.data_store import ReplayBufferDataStore
from training.peg_insertion import resume as r
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.adapter=repaired_replay_datastore_insert();self.adapter.__enter__()
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):
        self.adapter.__exit__(None,None,None);self.temp.cleanup()
    def buffer(self,capacity=128):
        b=ReplayBufferDataStore(gym.spaces.Box(-1,1,(20,),dtype=np.float32),
                               gym.spaces.Box(-1,1,(7,),dtype=np.float32),capacity)
        r.install_actor_step_tracking(b)
        return b
    def insert(self,b,value):
        b.insert(dict(observations=np.full(20,value,dtype=np.float32),
                      next_observations=np.full(20,value,dtype=np.float32),
                      actions=np.zeros(7,dtype=np.float32),rewards=np.float32(value),
                      masks=np.float32(1),dones=False))
    def start(self,fn):
        errors=[]
        def guarded():
            try:fn()
            except BaseException as exc:errors.append(exc)
        t=threading.Thread(target=guarded,daemon=True);t.start();return t,errors
    def joined(self,pair):
        t,errors=pair;t.join(5);self.assertFalse(t.is_alive(),'deadlock');self.assertEqual(errors,[])
    def test_insert_and_counter_are_atomic(self):
        b=self.buffer();entered=threading.Event();release=threading.Event();captured=threading.Event()
        original=b.project_original_insert
        def held(data):
            original(data);entered.set()
            if not release.wait(5):raise TimeoutError('test release')
        b.project_original_insert=held
        writer=self.start(lambda:self.insert(b,7));self.assertTrue(entered.wait(5))
        result=[]
        reader=self.start(lambda:(result.append(r.capture_replay_snapshot(b)),captured.set()))
        self.assertFalse(captured.wait(.1),'snapshot escaped insertion lock')
        release.set();self.joined(writer);self.joined(reader)
        self.assertEqual((result[0]['size'],result[0]['actor_steps']),(1,1))
        self.assertTrue(np.all(result[0]['dataset_dict']['observations']==7))
    def test_detached_arrays_rng_and_wrapped_index(self):
        b=self.buffer(4)
        for i in range(6):self.insert(b,i)
        p=r.capture_replay_snapshot(b);saved=pickle.dumps(p)
        self.assertEqual((p['size'],p['insert_index'],p['actor_steps']),(4,2,6))
        self.insert(b,9);b.sample(2)
        self.assertEqual(pickle.dumps(p),saved)
        self.assertNotEqual(pickle.dumps(p['rng_state']),pickle.dumps(r._rng(b)))
        self.assertEqual(b.project_actor_steps,7)
    def test_sample_rng_uses_same_snapshot_lock(self):
        b=self.buffer();self.insert(b,1)
        held=threading.Event();release=threading.Event();captured=threading.Event();result=[]
        def sampling():
            with b._lock:
                b.sample(1);held.set()
                if not release.wait(5):raise TimeoutError('sample release')
        sampler=self.start(sampling);self.assertTrue(held.wait(5))
        reader=self.start(lambda:(result.append(r.capture_replay_snapshot(b)),captured.set()))
        self.assertFalse(captured.wait(.1));release.set()
        self.joined(sampler);self.joined(reader)
        self.assertEqual(pickle.dumps(result[0]['rng_state']),pickle.dumps(r._rng(b)))
    def test_live_insert_during_model_save_no_rewind(self):
        b=self.buffer();self.insert(b,1)
        cfg=dict(seed=0,environment='test',algorithm='test',observation={},action={},
                 demo={'sha256':'test','transitions':1},wandb={'project':'test','run_name':'test'})
        manifest=self.root/'manifest.json';manifest.write_text('{}')
        def model_save(directory,state,step):
            writer=self.start(lambda:[self.insert(b,2) for _ in range(30)])
            self.joined(writer)  # fails if model I/O is still holding replay lock
            cp=directory/f'checkpoint_{step}';cp.write_bytes(b'test model');return cp
        with patch.object(r,'save_fresh_state',side_effect=model_save):
            for step,expected in ((0,1),(1,31)):
                cp=r.save_bundle(self.root,None,b,step,0,cfg,manifest,'test',{})
                meta,rp=r.sidecars(cp);m=json.loads(meta.read_text());p=pickle.loads(rp.read_bytes())
                self.assertEqual(m['actor_steps'],expected)
                self.assertEqual(p['actor_steps'],expected);self.assertEqual(p['size'],expected)
                self.assertEqual(m['online_replay_sha256'],r.sha256_file(rp))
                self.assertEqual(b.project_actor_steps,expected+30)
    def test_collision_does_not_mutate_counter(self):
        b=self.buffer();self.insert(b,1);path=self.root/'replay.pkl'
        r.snapshot_replay(b,path)
        with self.assertRaises(FileExistsError):r.snapshot_replay(b,path)
        self.assertEqual(b.project_actor_steps,1)


if __name__=='__main__':
    unittest.main(verbosity=2)
