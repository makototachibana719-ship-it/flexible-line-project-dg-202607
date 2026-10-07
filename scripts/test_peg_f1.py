"""Synthetic CPU-only policy tests; never start training or contact W&B."""
import ast
import copy
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from training.peg_insertion import formal_policy as p,formal_preflight as f
REAL_ROOT=p.ROOT

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('configs/peg_insertion','deliverables/gpu_training_handoff/peg_insertion','results/checkpoints/formal'):
            (self.root/name).mkdir(parents=True)
        (self.root/'demo').write_bytes(b'synthetic fixture only')
        self.cfg=dict(seed=0,environment='fixture',algorithm='fixture',observation={},action={},
            demo=dict(path='demo',sha256=p.sha(self.root/'demo'),transitions=1),
            training=dict(max_steps=10000,checkpoint_period=1000,replay_capacity=10000),
            wandb=dict(project=f.PROJECT,run_name=f.RUN_NAME),
            outputs=dict(checkpoint_dir='results/checkpoints/formal',log_dir='results/logs',actor_log_dir='results/actor'))
        self.config=self.root/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml'
        self.config.write_text(yaml.safe_dump(self.cfg))
        (self.root/'configs/peg_insertion/runtime_gpu_formal.env').write_text(''.join(k+'='+v+'\n' for k,v in p.RUNTIME.items()))
        (self.root/'deliverables/gpu_training_handoff/peg_insertion/demo_manifest.json').write_text('{}')
        (self.root/'source.py').write_text('# fixture')
        (self.root/'evidence').write_text('PASS fixture')
        for obj,name,value in ((p,'ROOT',self.root),(p,'GATE',self.root/'gate.json'),(p,'SOURCES',['source.py']),(p,'MONITOR_SOURCE','source.py')):
            mock=patch.object(obj,name,value);mock.start();self.addCleanup(mock.stop)
        mock=patch.dict(os.environ,p.RUNTIME);mock.start();self.addCleanup(mock.stop)
        self.gate=dict(formal_ready=True,exact_path_acceptance_50='PASS_WITH_PER_LEAF_SYNC',exact_path_checkpoint_resume_5='PASS',
            runtime=p.RUNTIME,source_hashes={'source.py':p.sha(self.root/'source.py')},bindings=p.bindings(),
            sync_policy='PER_LEAF_BLOCK_UNTIL_READY',sync_exact_root_cause='UNRESOLVED',
            role_aware_preflight={r:'PASS' for r in f.ROLES},evidence_hashes={'evidence':p.sha(self.root/'evidence')},
            run_binding=dict(project=f.PROJECT,fresh_name=f.RUN_NAME,resume='must_same_run_id'),
            checkpoint_cadence=dict(basis='global_learner_update',period=1000,max_steps=10000))
        self.gate.update(schema_version=4,exit_classification_fix='PASS',formal_step_watchdog=True,formal_finite_loss_guard=True,
            formal_rng_guard=True,formal_per_update_local_monitor=True,formal_monitoring_gate='PASS',
            monitor_implementation_hash=p.sha(self.root/'source.py'),
            monitor_policy=dict(timeout_seconds=120,snapshot_seconds=[30,60,110],wandb_log_period=10))
        self.write_gate()
    def write_gate(self):p.GATE.write_text(json.dumps(self.gate))
    def checkpoint(self):
        cp=self.root/'results/checkpoints/formal/checkpoint_1000';cp.write_bytes(b'fixture model, not executable')
        rp=cp.with_name(cp.name+'_online_replay.pkl')
        rp.write_bytes(pickle.dumps(dict(format_version=1,capacity=10000,size=1234,insert_index=1234,actor_steps=1234,rng_state={'fixture':1})))
        m=dict(checkpoint_step=1000,learner_update_steps=1000,wandb_project=f.PROJECT,wandb_name=f.RUN_NAME,wandb_run_id='same-run',
            config_hash=f.canonical(self.cfg),seed=0,environment='fixture',algorithm='fixture',observation={},action={},
            demo_sha256=self.cfg['demo']['sha256'],demo_transition_count=1,demo_manifest_hash=self.gate['bindings']['demo_manifest_hash'],
            model_sha256=p.sha(cp),online_replay_sha256=p.sha(rp),actor_steps=1234,online_replay_size=1234,online_replay_capacity=10000,
            source_snapshot={'formal_binding':dict(source_hashes=self.gate['source_hashes'],bindings=self.gate['bindings'])})
        mp=cp.with_name(cp.name+'_resume_metadata.json');mp.write_text(json.dumps(m))
        return cp,mp,rp
    def session(self,checkpoint=None):
        f.control().mkdir(parents=True,exist_ok=True)
        session=dict(project=f.PROJECT,run_name=f.RUN_NAME,run_id='same-run',environment='fixture',
            config_hash=self.gate['bindings']['config_hash'],source_hashes=self.gate['source_hashes'],endpoint='127.0.0.1:5488',
            pid=os.getpid(),pid_start=f.pid_start(os.getpid()),checkpoint=str(checkpoint) if checkpoint else None)
        (f.control()/'learner.json').write_text(json.dumps(session))
    def test_valid_gate(self):self.assertTrue(p.require_gate()['formal_ready'])
    def test_v2_schema_rejected(self):
        self.gate['schema_version']=2;self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_missing_monitor_guard_rejected(self):
        self.gate['formal_step_watchdog']=False;self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_monitor_hash_rejected(self):
        self.gate['monitor_implementation_hash']='stale';self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_monitor_policy_rejected(self):
        self.gate['monitor_policy']['timeout_seconds']=999;self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_missing_gate(self):
        p.GATE.unlink()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_false_gate(self):
        self.gate['formal_ready']=False;self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_stale_source(self):
        (self.root/'source.py').write_text('changed')
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_runtime_hash(self):
        self.gate['bindings']['runtime_hash']='bad';self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_runtime_env(self):
        with patch.dict(os.environ,{'XLA_FLAGS':'bad'}),self.assertRaises(RuntimeError):p.require_gate()
    def test_inherited_jax_override(self):
        with patch.dict(os.environ,{'JAX_DISABLE_JIT':'1'}),self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_installed_version(self):
        with patch('importlib.metadata.version',return_value='wrong'),self.assertRaises(ValueError):f.preflight('fresh_learner')
    def test_wrong_config(self):
        self.config.write_text(self.config.read_text()+'\nchanged: true\n')
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_demo(self):
        (self.root/'demo').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_wrong_evidence(self):
        (self.root/'evidence').write_text('FAIL')
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_missing_role_approval(self):
        self.gate['role_aware_preflight']['resume_actor']='FAIL';self.write_gate()
        with self.assertRaises(RuntimeError):p.require_gate()
    def test_fresh_learner(self):self.assertEqual(f.preflight('fresh_learner')['status'],'PASS')
    def test_dirty_fresh(self):
        (self.root/'results/checkpoints/formal/old').write_text('old')
        with self.assertRaises(ValueError):f.preflight('fresh_learner')
    def test_fresh_run_override(self):
        with self.assertRaises(ValueError):f.preflight('fresh_learner',run_id='inherited')
    def test_duplicate_actor_lock(self):
        with f.role_lock('actor'):
            self.assertTrue(f.active('actor'))
            with self.assertRaises(RuntimeError):
                with f.role_lock('actor'):pass
            with self.assertRaises(ValueError):f.preflight('actor_join',run_id='same-run')
        self.assertFalse(f.active('actor'))
    def test_duplicate_learner(self):
        with f.role_lock('learner'),self.assertRaises(ValueError):f.preflight('fresh_learner')
    def test_actor_join_existing_outputs(self):
        self.session();(self.root/'results/checkpoints/formal/learner-file').write_text('allowed')
        with patch.object(f,'active',side_effect=lambda r:r=='learner'),patch.object(f.socket,'create_connection',return_value=MagicMock()):
            self.assertEqual(f.preflight('actor_join',run_id='same-run')['status'],'PASS')
    def test_actor_wrong_run(self):
        self.session()
        with patch.object(f,'active',side_effect=lambda r:r=='learner'),self.assertRaises(ValueError):f.preflight('actor_join',run_id='wrong')
    def test_actor_wrong_endpoint(self):
        with self.assertRaises(ValueError):f.preflight('actor_join',run_id='same-run',ip='remote')
    def test_actor_no_learner(self):
        with self.assertRaises(ValueError):f.preflight('actor_join',run_id='same-run')
    def test_resume_learner(self):
        cp,_,_=self.checkpoint()
        with patch.object(f,'cloud_resume_boundary') as cloud:
            result=f.preflight('resume_learner',cp,'same-run',check_cloud=True)
            cloud.assert_called_once_with('same-run',1000)
        self.assertEqual(result['checkpoint_step'],1000)
    def test_resume_actor_metadata_steps(self):
        cp,_,_=self.checkpoint();self.session(cp)
        with patch.object(f,'active',side_effect=lambda r:r=='learner'),patch.object(f.socket,'create_connection',return_value=MagicMock()):
            self.assertEqual(f.preflight('resume_actor',cp,'same-run')['actor_steps'],1234)
    def test_resume_wrong_run(self):
        cp,_,_=self.checkpoint()
        with self.assertRaises(ValueError):f.preflight('resume_learner',cp,'wrong')
    def test_missing_triplet(self):
        cp,_,rp=self.checkpoint();rp.unlink()
        with self.assertRaises(ValueError):f.preflight('resume_learner',cp,'same-run')
    def test_model_hash(self):
        cp,_,_=self.checkpoint();cp.write_bytes(b'tampered')
        with self.assertRaises(ValueError):f.preflight('resume_learner',cp,'same-run')
    def test_bundle_source_binding(self):
        cp,mp,_=self.checkpoint();m=json.loads(mp.read_text());m['source_snapshot']={};mp.write_text(json.dumps(m))
        with self.assertRaises(ValueError):f.preflight('resume_learner',cp,'same-run')
    def test_metadata_counter(self):
        cp,mp,_=self.checkpoint();m=json.loads(mp.read_text());m['actor_steps']=1233;mp.write_text(json.dumps(m))
        with self.assertRaises(ValueError):f.preflight('resume_learner',cp,'same-run')
    def test_cloud_ahead_rejected(self):
        cp,_,_=self.checkpoint()
        with patch.object(f,'cloud_resume_boundary',side_effect=ValueError('ahead')),self.assertRaises(ValueError):f.preflight('resume_learner',cp,'same-run',check_cloud=True)

class SourceTests(unittest.TestCase):
    def test_per_leaf_no_batched(self):
        from training.peg_insertion.sync_policy import block_until_ready,JaxView
        leaves=[MagicMock(),MagicMock()]
        fake=MagicMock();fake.tree_util.tree_leaves.return_value=leaves
        with patch.dict(sys.modules,{'jax':fake}):
            tree=object();self.assertIs(block_until_ready(tree),tree)
        for leaf in leaves:leaf.block_until_ready.assert_called_once_with()
        fake.block_until_ready.assert_not_called()
        self.assertIs(JaxView(fake).block_until_ready,block_until_ready)
    def test_global_cadence_actual_expression(self):
        path=REAL_ROOT/'external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/async_sac_state_sim.py'
        tree=ast.parse(path.read_text())
        learner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='learner')
        assignment=next(n for n in ast.walk(learner) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='update_steps' for t in n.targets))
        self.assertEqual(ast.unparse(assignment.value),'FLAGS.learner_start_step')
        condition=next(n.test for n in ast.walk(learner) if isinstance(n,ast.If) and 'FLAGS.checkpoint_period' in ast.unparse(n.test))
        from types import SimpleNamespace
        expr=compile(ast.Expression(condition),'<actual cadence>','eval')
        def saves(start,count):return [step for step in range(start,start+count) if eval(expr,{'FLAGS':SimpleNamespace(checkpoint_period=1000),'update_steps':step})]
        self.assertEqual(saves(0,10000),list(range(0,10000,1000)))
        self.assertEqual(saves(26,5),[])
        self.assertEqual(saves(999,3),[1000])
        self.assertEqual(saves(1001,2000),[2000,3000])
    def test_entry_global_mapping(self):
        source=(REAL_ROOT/'training/peg_insertion/run_peg_training.py').read_text()
        self.assertIn('global_step=int(step)',source)
        self.assertIn('global_step=step',source)
        self.assertIn('--learner_start_step=',source)
        self.assertNotIn('jax.block_until_ready(new_agent)',source)
    def test_formal_shell_source(self):
        for name in ('run_train_learner.sh','run_train_actor.sh','run_resume_learner.sh','run_resume_actor.sh'):
            text=(REAL_ROOT/'deliverables/gpu_training_handoff/peg_insertion'/name).read_text()
            self.assertIn('source "$ROOT/configs/peg_insertion/runtime_gpu_formal.env"',text)
            self.assertIn('scripts/preflight_peg_formal.py --role',text)

if __name__=='__main__':unittest.main()
