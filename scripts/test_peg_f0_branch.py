"""CPU-only branch policy/restore assertions; never launches training."""
import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from training.peg_insertion import branch_acceptance as b

class BranchTests(unittest.TestCase):
    def test_source_syntax(self):
        import ast
        for name in ('training/peg_insertion/run_peg_training.py','training/peg_insertion/branch_acceptance.py','scripts/verify_peg_f0_branch.py'):
            ast.parse((b.ROOT/name).read_text())
    def test_exact_source_config(self):
        from training.peg_insertion.resume import canonical_hash
        cfg=yaml.safe_load((b.ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml').read_text())
        frozen=copy.deepcopy(cfg)
        src=b.source_config(cfg)
        meta=json.loads(b.SOURCE.with_name(b.SOURCE.name+'_resume_metadata.json').read_text())
        self.assertEqual(canonical_hash(src),meta['config_hash'])
        self.assertEqual(cfg,frozen)
    def test_source_hashes_and_approval(self):b.validate_source(b.SOURCE,b.SOURCE_RUN)
    def test_wrong_source_rejected(self):
        with self.assertRaises(ValueError):b.validate_source(b.SOURCE.with_name('checkpoint_0'),b.SOURCE_RUN)
    def test_wrong_run_rejected(self):
        with self.assertRaises(ValueError):b.validate_source(b.SOURCE,'different')
    def test_exact_tree(self):
        value={'a':np.array([1,2],dtype=np.uint32),'rng':{'s':12}}
        self.assertEqual(b.assert_equal_tree(copy.deepcopy(value),value),2)
    def test_wrong_values(self):
        with self.assertRaises(AssertionError):b.assert_equal_tree(np.array([2]),np.array([1]))
    def test_wrong_dtype(self):
        with self.assertRaises(AssertionError):b.assert_equal_tree(np.array([1],dtype=np.float32),np.array([1],dtype=np.float64))
    def test_wrong_structure(self):
        with self.assertRaises(AssertionError):b.assert_equal_tree({'a':1},{'b':1})
    def test_actor_progress(self):
        values=b.actor_range(1261,1500)
        self.assertEqual((values.start,values.stop,len(values)),(1261,2761,1500))
    def test_provenance(self):
        self.assertTrue(b.provenance()['resume_branch'])
        self.assertFalse(b.provenance()['same_run_history_continuation'])
        self.assertEqual(b.provenance()['source_checkpoint_step'],25)

if __name__=='__main__':unittest.main()
