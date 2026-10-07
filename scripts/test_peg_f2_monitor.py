"""CPU-only monitoring regression. Synthetic timeouts do not run a learner."""
import ast
import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion import formal_monitor as m, formal_policy as p
from training.peg_insertion.update_observer import UpdateObserver

LOSSES = {'critic': {'critic_loss': 1.0}, 'actor': {'actor_loss': 2.0},
          'temperature': {'temperature_loss': 3.0}}

class Timer:
    def __init__(self, delay, callback, args=()):
        self.delay, self.callback, self.args = delay, callback, args
        self.cancelled = False
    def start(self): pass
    def cancel(self): self.cancelled = True

class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.observer = MagicMock()
        self.sync = MagicMock()
        self.monitor = m.FormalMonitor(Path(self.temp.name)/'session', 0,
            rng=lambda a:a, sync=self.sync, replay=lambda:(100, 100),
            observer=lambda *_:self.observer, timer=Timer)
    def update(self, before=0, after=1, losses=None):
        return self.monitor.update(lambda:(after, LOSSES if losses is None else losses), before)
    def failure(self, reason):
        record=json.loads((self.monitor.directory/'formal_status.json').read_text())
        self.assertEqual(record['reason'], reason)
        self.assertEqual(record['formal_status'], 'FAIL')
    def test_normal_under_limit(self):
        after,info=self.update(); self.assertEqual(after,1); self.assertIs(info,LOSSES)
        self.assertEqual(self.sync.call_count,2)
        self.observer.arm.assert_called_once_with(delays=(30,60,110))
        self.observer.close.assert_called_once()
    def test_timeout_elapsed(self):
        times=iter([0,0,1,121,121]); self.monitor.clock=lambda:next(times)
        with self.assertRaises(m.MonitorFailure):self.update()
        self.failure('update_timeout')
    def test_critic_nan(self):
        losses=copy.deepcopy(LOSSES);losses['critic']['critic_loss']=float('nan')
        with self.assertRaises(m.MonitorFailure):self.update(losses=losses)
        self.failure('non_finite_loss')
    def test_actor_inf(self):
        losses=copy.deepcopy(LOSSES);losses['actor']['actor_loss']=float('inf')
        with self.assertRaises(m.MonitorFailure):self.update(losses=losses)
        self.failure('non_finite_loss')
    def test_temperature_missing(self):
        losses=copy.deepcopy(LOSSES);del losses['temperature']
        with self.assertRaises(m.MonitorFailure):self.update(losses=losses)
        self.failure('non_finite_loss')
    def test_rng_mismatch_before_call(self):
        self.update();call=MagicMock()
        with self.assertRaises(m.MonitorFailure):self.monitor.update(call,100)
        call.assert_not_called();self.failure('rng_continuity_failure')
    def test_rng_not_advancing(self):
        with self.assertRaises(m.MonitorFailure):self.update(1,1)
        self.failure('rng_continuity_failure')
    def test_every_update_and_startup(self):
        for index in range(12):self.update(index,index+1)
        rows=[json.loads(line) for line in (self.monitor.directory/'formal_monitor_updates.jsonl').read_text().splitlines()]
        self.assertEqual([r['global_update'] for r in rows],list(range(12)))
        self.assertEqual([r['startup_window'] for r in rows],[True]*10+[False]*2)
        fields={'global_update','timestamp','dispatch_seconds','sync_seconds','true_update_seconds','rng_before','rng_after','critic_loss','actor_loss','temperature_loss','finite','replay_size','actor_steps'}
        for row in rows:self.assertTrue(fields<=row.keys());self.assertTrue(row['finite'])
        self.assertEqual(self.monitor.previous_rng,12)
    def test_resume_global_number(self):
        self.monitor.index=1001;self.update()
        row=json.loads((self.monitor.directory/'formal_monitor_updates.jsonl').read_text())
        self.assertEqual(row['global_update'],1001);self.assertFalse(row['startup_window'])
    def test_no_retry_after_failure(self):
        with self.assertRaises(m.MonitorFailure):self.update(0,0)
        with self.assertRaises(m.MonitorFailure):self.update()
        self.failure('rng_continuity_failure')
    def test_exception_cancels_and_records(self):
        with self.assertRaises(ValueError):self.monitor.update(lambda:(_ for _ in ()).throw(ValueError('synthetic')),0)
        self.failure('update_exception');self.observer.close.assert_called_once()
    def test_watchdog_hard_exit_subprocess(self):
        # Same watchdog implementation, shortened only in this isolated CPU fixture.
        path=Path(self.temp.name)/'timeout'
        code="""import sys,time
from pathlib import Path
from unittest.mock import MagicMock
from training.peg_insertion import formal_monitor as m
m.TIMEOUT_SECONDS=0.05
monitor=m.FormalMonitor(Path(sys.argv[1]),77,rng=lambda x:x,sync=lambda x:None,replay=lambda:(1,1),observer=lambda *_:MagicMock())
monitor.update(lambda:time.sleep(5),0)
"""
        result=subprocess.run([sys.executable,'-c',code,str(path)],cwd=ROOT,timeout=10,capture_output=True,text=True)
        self.assertEqual(result.returncode,124,result.stderr)
        status=json.loads((path/'formal_status.json').read_text())
        self.assertEqual(status['failed_update'],77);self.assertEqual(status['reason'],'update_timeout')
    def test_snapshot_schedule_reuses_observer(self):
        timers=[]
        def factory(*args,**kwargs):
            timer=Timer(*args,**kwargs);timers.append(timer);return timer
        observer=UpdateObserver(Path(self.temp.name),1,None)
        with patch('training.peg_insertion.update_observer.threading.Timer',side_effect=factory):observer.arm(delays=m.SNAPSHOT_SECONDS)
        self.assertEqual([t.delay for t in timers],[30,60,110]);observer.close()
        self.assertTrue(all(t.cancelled for t in timers))
    def test_completed_update_cannot_timeout_later(self):
        timers=[]
        def factory(*args):
            timer=Timer(*args);timers.append(timer);return timer
        self.monitor.timer=factory;exit_fn=MagicMock();self.monitor.hard_exit=exit_fn
        self.update();timers[0].callback()
        self.assertEqual(timers[0].delay,120);self.assertTrue(timers[0].cancelled)
        exit_fn.assert_not_called();self.assertFalse(self.monitor.failed)
    def test_timeout_during_sync_subprocess(self):
        path=Path(self.temp.name)/'sync_timeout'
        code="""import sys,time
from pathlib import Path
from unittest.mock import MagicMock
from training.peg_insertion import formal_monitor as m
m.TIMEOUT_SECONDS=0.05
monitor=m.FormalMonitor(Path(sys.argv[1]),1,rng=lambda x:x,sync=lambda x:time.sleep(5),replay=lambda:(1,1),observer=lambda *_:MagicMock())
monitor.update(lambda:(1,{}),0)
"""
        result=subprocess.run([sys.executable,'-c',code,str(path)],cwd=ROOT,timeout=10,capture_output=True,text=True)
        self.assertEqual(result.returncode,124,result.stderr)
        self.assertEqual(json.loads((path/'formal_status.json').read_text())['reason'],'update_timeout')
    def test_snapshot_files_cpu_fixture(self):
        observer=UpdateObserver(Path(self.temp.name),1,None)
        with patch('training.peg_insertion.update_observer.subprocess.run') as run:
            for delay in m.SNAPSHOT_SECONDS:observer.snapshot(delay)
        self.assertEqual(run.call_count,6)
        for delay in m.SNAPSHOT_SECONDS:
            for ext in ('stack','cpu','gpu'):self.assertTrue((Path(self.temp.name)/f'update_1_{delay}s.{ext}.txt').is_file())

class SourceTests(unittest.TestCase):
    def test_frozen_files(self):
        old=json.loads((ROOT/'results/validation/peg/formal_readiness_v2.json').read_text())
        allowed={'training/peg_insertion/run_peg_training.py','training/peg_insertion/formal_policy.py','scripts/test_peg_f1.py'}
        for name,digest in old['source_hashes'].items():
            if name not in allowed:self.assertEqual(p.sha(ROOT/name),digest,name)
        self.assertEqual(p.bindings(),old['bindings'])
    def test_wandb_ten_and_candidate_only_override(self):
        source=(ROOT/'training/peg_insertion/run_peg_training.py').read_text()
        self.assertIn("if candidate:argv.append('--log_period=1')",source)
        delivered=(ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/async_sac_state_sim.py').read_text()
        tree=ast.parse(delivered)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='log_period']
        self.assertEqual(len(calls),1);self.assertEqual(calls[0].args[1].value,10)
    def test_original_call_signature_unchanged(self):
        source=(ROOT/'training/peg_insertion/run_peg_training.py').read_text()
        self.assertIn('original_formal_update(self,*args,**kwargs)',source)
        self.assertIn("if a.profile=='formal' and a.role=='learner':",source)
        self.assertIn('sync=per_leaf_sync',source)
    def test_actual_wrapper_preserves_omitted_pmap_axis(self):
        tree=ast.parse((ROOT/'training/peg_insertion/run_peg_training.py').read_text())
        wrapper=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='formal_update')
        original=MagicMock(return_value=('agent','info'))
        monitor=MagicMock();monitor.update.side_effect=lambda call,agent:call()
        scope=dict(original_formal_update=original,monitor=monitor)
        exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual wrapper>','exec'),scope)
        owner=object();batch=object()
        self.assertEqual(scope['formal_update'](owner,batch,utd_ratio=1),('agent','info'))
        original.assert_called_once_with(owner,batch,utd_ratio=1)
    def test_diagnostic_rejected_before_runtime(self):
        source=(ROOT/'training/peg_insertion/run_peg_training.py').read_text();tree=ast.parse(source)
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_main')
        checks=[n for n in ast.walk(func) if isinstance(n,ast.If) and 'diagnostic_values' in ast.unparse(n.test)]
        self.assertTrue(checks)
        self.assertTrue(any(isinstance(n,ast.Raise) for n in ast.walk(checks[0])))
    def test_historical_v2_rejected(self):
        with patch.object(p,'GATE',ROOT/'results/validation/peg/formal_readiness_v2.json'):
            with self.assertRaisesRegex(RuntimeError,'stale'):p.require_gate()

if __name__=='__main__':unittest.main()
