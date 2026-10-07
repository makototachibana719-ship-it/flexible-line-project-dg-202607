import unittest
from verify_peg_f0 import unique_history,LOSS_KEYS

class Schema(unittest.TestCase):
    def setUp(self):
        self.local={i:{k:float(i+1) for k in LOSS_KEYS} for i in range(50)}
        self.rows=[dict(_step=i,**self.local[i]) for i in range(50)]
    def test_complete(self):self.assertEqual(unique_history(self.rows,range(50),self.local),[])
    def test_missing_step(self):
        with self.assertRaises(ValueError):unique_history(self.rows[:-1],range(50),self.local)
    def test_step_only(self):
        with self.assertRaises(ValueError):unique_history([{'_step':i} for i in range(50)],range(50),self.local)
    def test_missing_field(self):
        del self.rows[5][LOSS_KEYS[1]]
        with self.assertRaises(ValueError):unique_history(self.rows,range(50),self.local)
    def test_nonfinite(self):
        self.rows[5][LOSS_KEYS[0]]=float('nan')
        with self.assertRaises(ValueError):unique_history(self.rows,range(50),self.local)
    def test_mismatch(self):
        self.rows[5][LOSS_KEYS[0]]=99.
        with self.assertRaises(ValueError):unique_history(self.rows,range(50),self.local)
    def test_conflict(self):
        self.rows.append(dict(self.rows[0],**{LOSS_KEYS[0]:99.}))
        with self.assertRaises(ValueError):unique_history(self.rows,range(50),self.local)
    def test_identical_warning(self):
        self.rows.append(dict(self.rows[0]))
        self.assertEqual(unique_history(self.rows,range(50),self.local),[0])

if __name__=='__main__':unittest.main()
