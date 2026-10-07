import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.peg_insertion import formal_policy as p
from verify_peg_f0 import unique_history

class Safety(unittest.TestCase):
    def test_missing_gate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p,'GATE',Path(d)/'missing'):
            with self.assertRaisesRegex(RuntimeError,'FORMAL_LAUNCH_BLOCKED'):p.require_gate()
    def test_incomplete_gate(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'gate';f.write_text(json.dumps({'formal_ready':False,'exact_path_candidate':'PASS'}))
            with patch.object(p,'GATE',f),self.assertRaises(RuntimeError):p.require_gate()
    def test_unbound_gate(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'gate';f.write_text(json.dumps({'formal_ready':True,'exact_path_candidate':'PASS','exact_path_resume':'PASS','runtime':p.RUNTIME,'source_hashes':{}}))
            with patch.object(p,'GATE',f),self.assertRaisesRegex(RuntimeError,'stale'):p.require_gate()
    def test_history_exact_duplicate_warns(self):
        self.assertEqual(unique_history([{'_step':0,'loss':1.0}]*2,[0],{0:{'loss':1.0}}),[0])
    def test_history_conflict_fails(self):
        with self.assertRaises(ValueError):unique_history([{'_step':0,'loss':1},{'_step':0,'loss':2}],[0],{0:{'loss':1}})
    def test_history_missing_fails(self):
        with self.assertRaises(ValueError):unique_history([],range(1),{0:{'loss':1}})

if __name__=='__main__':unittest.main()
