"""Exercise actual entry classifier and finally statement without importing training."""
import ast
import sys
import unittest
import subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'training/peg_insertion/run_peg_training.py').read_text())
function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='wandb_exit_code')
finish=next(x for x in ast.walk(tree) if isinstance(x,ast.If) and "active.get('run')" in ast.unparse(x.test) and 'wandb_exit_code' in ast.unparse(x))
scope={'sys':sys}
exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual classifier>','exec'),scope)

class ExitTests(unittest.TestCase):
 def check(self,exception,expected):
  run=MagicMock();local=dict(scope,a=SimpleNamespace(profile='formal'),active={'run':run})
  caught=None
  try:
   try:
    if exception is not None:raise exception
   finally:exec(compile(ast.Module(body=[finish],type_ignores=[]),'<actual finally>','exec'),local)
  except BaseException as exc:caught=exc
  self.assertIs(caught,exception)
  run.finish.assert_called_once_with(exit_code=expected)
 def test_normal_return(self):self.check(None,0)
 def test_exit_none(self):self.check(SystemExit(None),0)
 def test_exit_zero(self):self.check(SystemExit(0),0)
 def test_exit_one(self):self.check(SystemExit(1),1)
 def test_exit_timeout(self):self.check(SystemExit(124),124)
 def test_runtime_error(self):self.check(RuntimeError('failure'),1)
 def test_exit_message(self):self.check(SystemExit('failure'),1)
 def test_interrupt(self):self.check(KeyboardInterrupt(),1)
 def test_watchdog_bypasses_success_finalization(self):
  with tempfile.TemporaryDirectory() as directory:
   code="""import sys,time
from pathlib import Path
from unittest.mock import MagicMock
from training.peg_insertion import formal_monitor as m
m.TIMEOUT_SECONDS=.05
root=Path(sys.argv[1])
monitor=m.FormalMonitor(root/'monitor',1,rng=lambda x:x,sync=lambda x:None,replay=lambda:(1,1),observer=lambda *_:MagicMock())
try:monitor.update(lambda:time.sleep(5),0)
finally:(root/'finish_called').write_text('incorrect if hard exit reaches this')
"""
   result=subprocess.run([sys.executable,'-c',code,directory],cwd=ROOT,timeout=10,capture_output=True)
   self.assertEqual(result.returncode,124)
   self.assertFalse((Path(directory)/'finish_called').exists())
   self.assertIn('update_timeout',(Path(directory)/'monitor/formal_status.json').read_text())
 def test_only_classifier_changed(self):
  path=ROOT/'training/peg_insertion/run_peg_training.py'
  old=path.with_name(path.name+'.before_f2_exitfix_20260904').read_text()
  new=path.read_text();block=new[new.index('def wandb_exit_code('):new.index('def load_config(')]
  restored=new.replace(block,'').replace('wandb_exit_code(sys.exc_info()[1])','1 if sys.exc_info()[0] else 0')
  self.assertEqual(restored,old)

if __name__=='__main__':unittest.main()
