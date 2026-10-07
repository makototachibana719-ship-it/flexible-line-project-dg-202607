"""CPU fixtures plus unchanged production solve/dispatch AST and real CPU solver.
No MjModel/MjData, env reset or simulator step is constructed/called.
"""
import ast,copy,json,sys,tempfile,unittest,types,typing
from pathlib import Path
import numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mujoco import minimize
from training.peg_insertion.axial_telemetry import AxialTelemetry,V2
BASE=ROOT/'external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/controllers'

def production_method(file,cls,name):
    tree=ast.parse((BASE/file).read_text());node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==name)
    ns=dict(np=np,minimize=minimize,Optional=typing.Optional,Dict=typing.Dict)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(BASE/file),'exec'),ns)
    return ns[name]

class Model:
    njnt=nq=nv=nu=6
    jnt_qposadr=np.arange(6);jnt_dofadr=np.arange(6)
    jnt_range=np.tile([-2.,2.],(6,1));jnt_limited=np.ones(6)
    actuator_ctrlrange=np.tile([-.5,.5],(6,1));actuator_ctrllimited=np.ones(6)
    actuator_trnid=np.array([[i,-1] for i in range(6)])
    def joint(self,i):return types.SimpleNamespace(name='joint'+str(i))
    def actuator(self,i):return types.SimpleNamespace(name='actuator'+str(i))
    def geom(self,i):return types.SimpleNamespace(name=['peg','hole'][i])

class JointController:
    use_target_as_ctrl=True
    compute_control=production_method('joint_position.py','JointPositionController','compute_control')

class Controller:
    dof=6;arm_qpos_adr=arm_dof_adr=0
    solve_ik=production_method('cartesian_ik.py','CartesianIKController','solve_ik')
    compute_control=production_method('cartesian_ik.py','CartesianIKController','compute_control')
    def __init__(self,data):
        self.data=data;self.joint_limits=(np.full(6,-2.),np.full(6,2.));self.joint_controller=JointController()
        self.res_calls=0;self.jac_calls=0
    def get_arm_qpos(self):return self.data.qpos.copy()
    def _ik_residual(self,q,target_pos,target_quat,reg_target,position_only=False):
        self.res_calls+=1
        return (q.reshape(6)-np.r_[target_pos,target_quat[:3]]).reshape(6,1)
    def compute_jacobian(self,q):return np.eye(6)[:3],np.eye(6)[3:]
    def _ik_jacobian(self,q,r,target_pos,target_quat,position_only=False):
        self.jac_calls+=1
        return np.vstack(self.compute_jacobian(q))

class Raw:
    def __init__(self):
        self.model=Model();self.data=types.SimpleNamespace(qpos=np.zeros(6),qvel=np.zeros(6),ctrl=np.zeros(6),ncon=1,
            contact=[types.SimpleNamespace(geom1=0,geom2=1,dist=-.001,frame=np.eye(3).ravel(),pos=np.array([1.,2.,3.]))])
        self.controller=Controller(self.data)
    def _set_action(self,action):
        output=self.controller.compute_control(action[:7]);self.data.ctrl[:]=output

class Env:
    observation_contract_version=V2
    def __init__(self):self.env=Raw();self.calls=0
    def step(self,a):
        self.calls+=1;self.env._set_action(a)
        # Deterministic fixture state update, NOT physics.
        self.env.data.qpos[:]=self.env.data.ctrl
        return self.env.data.qpos.copy(),1.,False,False,{'fixture':True}

def contact_force(m,d,i,out):out[:]=[1,2,3,4,5,6]

class Tests(unittest.TestCase):
    def run_pair(self):
        off,on=Env(),Env();a=np.array([.8,.1,.2,1.,0.,0.,0.]);obs=AxialTelemetry(on,solver=minimize.least_squares,contact_force_reader=contact_force)
        roff=AxialTelemetry(off,enabled=False).step(a);ron=obs.step(a)
        np.testing.assert_array_equal(roff[0],ron[0]);self.assertEqual(roff[1:],ron[1:])
        np.testing.assert_array_equal(off.env.data.ctrl,on.env.data.ctrl)
        self.assertEqual(off.env.controller.res_calls,on.env.controller.res_calls)
        self.assertEqual(off.env.controller.jac_calls,on.env.controller.jac_calls)
        self.assertEqual(off.calls,on.calls);self.assertTrue(obs.last['complete'])
        return on,obs,a
    def test_off_on_identical_production_dispatch(self):self.run_pair()
    def test_solver(self):
        env,o,a=self.run_pair();s=o.last['solver'];self.assertTrue(s['success']);self.assertEqual(s['nfev'],env.env.controller.res_calls);self.assertLess(s['cost'],1e-12)
    def test_jacobian(self):
        _,o,_=self.run_pair();self.assertTrue(o.last['jacobian_evaluations'])
        for j in o.last['jacobian_evaluations']:np.testing.assert_array_equal(j['singular_values'],np.ones(6))
    def test_joint_addresses_limits(self):
        _,o,_=self.run_pair();j=o.last['pre']['joints'][4];self.assertEqual(j['name'],'joint4');self.assertEqual(j['qpos_address'],4);self.assertEqual(j['distance_upper'],2)
    def test_ctrl_write_clip(self):
        _,o,_=self.run_pair();self.assertEqual(o.last['ctrl_before_write'],[0]*6);self.assertEqual(o.last['ctrl_after_write'],o.last['controller_output']);self.assertTrue(o.last['post']['actuators'][0]['clip_flag'])
    def test_contact_pairs(self):
        _,o,_=self.run_pair();c=o.last['post']['contacts'][0];self.assertEqual(c['geom1_name'],'peg');self.assertEqual(c['geom2_name'],'hole');self.assertEqual(c['distance'],-.001);self.assertEqual(c['contact_force_torque_local'],[1,2,3,4,5,6]);self.assertIsNone(c['impulse'])
    def test_no_state_mutation_by_snapshot(self):
        e=Env();o=AxialTelemetry(e,solver=minimize.least_squares,contact_force_reader=contact_force);before=copy.deepcopy(e.env.data.__dict__);o._snapshot()
        for k in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(before[k],getattr(e.env.data,k))
    def test_force_unavailable(self):
        def fail(*a):raise RuntimeError('unavailable')
        e=Env();o=AxialTelemetry(e,solver=minimize.least_squares,contact_force_reader=fail);self.assertIn('UNAVAILABLE',o._snapshot()['contacts'][0]['force_status'])
    def test_profile_restored(self):self.run_pair();self.assertIsNone(sys.getprofile())
    def test_existing_profile_rejected(self):
        sys.setprofile(lambda *a:None)
        try:
            with self.assertRaises(RuntimeError):AxialTelemetry(Env())
        finally:sys.setprofile(None)
    def test_v1_rejected(self):
        e=Env();e.observation_contract_version='V1'
        with self.assertRaises(ValueError):AxialTelemetry(e)
    def test_exception_not_swallowed(self):
        e=Env();o=AxialTelemetry(e,solver=minimize.least_squares,contact_force_reader=contact_force)
        def fail(a):raise RuntimeError('original failure')
        e.step=fail
        with self.assertRaisesRegex(RuntimeError,'original failure'):o.step(np.zeros(7))
        self.assertIsNone(sys.getprofile());self.assertIn('original failure',o.last['step_exception'])
    def test_output_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):AxialTelemetry(Env(),d,solver=minimize.least_squares,contact_force_reader=contact_force)
    def test_json_log(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'new';e=Env();o=AxialTelemetry(e,p,solver=minimize.least_squares,contact_force_reader=contact_force)
            o.step(np.array([.1,.2,.3,1.,0.,0.,0.]));o.close();r=json.loads((p/'steps.jsonl').read_text());self.assertTrue(r['complete'])
    def test_copied_values(self):
        e,o,_=self.run_pair();saved=copy.deepcopy(o.last);e.env.data.qpos[:]=99;self.assertEqual(saved,o.last)
    def test_functions_not_replaced(self):
        methods=(Controller.solve_ik,Controller.compute_control,minimize.least_squares,Raw._set_action)
        self.run_pair()
        self.assertEqual(methods,(Controller.solve_ik,Controller.compute_control,minimize.least_squares,Raw._set_action))
    def test_solver_exception_fallback_unchanged(self):
        def failing_solver(*a,**kw):raise ValueError('fixture solver failure')
        with patch.object(minimize,'least_squares',failing_solver):
            off,on=Env(),Env();o=AxialTelemetry(on,solver=failing_solver,contact_force_reader=contact_force)
            a=np.ones(7);r0=off.step(a);r1=o.step(a)
            np.testing.assert_array_equal(r0[0],r1[0]);np.testing.assert_array_equal(on.env.data.ctrl,np.zeros(6));self.assertIsNone(sys.getprofile())
            self.assertIsNone(o.last['solver']['success']);self.assertFalse(o.last['solver']['return_available'])
    def test_multiple_cpu_inputs(self):
        off,on=Env(),Env();o=AxialTelemetry(on,solver=minimize.least_squares,contact_force_reader=contact_force)
        for k in range(8):
            a=np.array([.01*k,-.02*k,.03*k,1.,.01*k,0.,0.])
            r0=off.step(a);r1=o.step(a);np.testing.assert_array_equal(r0[0],r1[0]);self.assertTrue(o.last['complete'])
        self.assertEqual(off.env.controller.res_calls,on.env.controller.res_calls);self.assertEqual(off.env.controller.jac_calls,on.env.controller.jac_calls)
if __name__=='__main__':unittest.main(verbosity=2)
