"""Offline evidence-only analysis. Never constructs an environment or agent."""
import json,pickle,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/validation/peg/f3_contract_audit_20260904'
PKG=ROOT/'external/packages/peg_insertion/peg_insertion'
def dump(n,x): (OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rot(q):return R.from_quat(np.asarray(q)[[1,2,3,0]])
rows=[json.loads(s) for s in (OUT/'target_realized_trace.jsonl').read_text().splitlines()]
demo=json.loads((OUT/'demo_audit.json').read_text())
runtime=json.loads((OUT/'runtime_contract.json').read_text())
all_demo=pickle.loads((PKG/'demos/peg_insert_20_demos_state.pkl').read_bytes())
actions=np.array([d['actions'] for d in all_demo])
obs=np.array([d['observations'] for d in all_demo]);nobs=np.array([d['next_observations'] for d in all_demo])
first=np.array(rows[0]['previous_target'])
dump('initial_state.json',dict(step=0,actual_ee_position=first[:3].tolist(),actual_ee_orientation=first[3:7].tolist(),controller_target=first.tolist(),policy_action=None,note='Reset state before any policy action; transition1 is action index0.'))
tree=ET.parse(ROOT/'runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml')
actuator_names=[x.attrib.get('name') for x in tree.getroot().find('actuator')]
sensor_xml=[x.attrib for x in tree.getroot().find('sensor')]
dump('generated_model_static.json',dict(actuators=actuator_names,sensors=sensor_xml,sha256=hashlib.sha256((ROOT/'runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml').read_bytes()).hexdigest()))
goal=obs[0,:3]+obs[0,13:16]
lo=np.minimum(first[:3],goal)-.05;hi=np.maximum(first[:3],goal)+.05
translation_res=[];rotation_res=[]
for r in rows:
    prev=np.array(r['previous_target']);target=np.array(r['applied_target']);a=np.array(r['policy_action'])
    translation_res.append(float(np.max(np.abs(np.clip(prev[:3]+a[:3]*.002,lo,hi)-target[:3]))))
    expected=rot(prev[3:7])*R.from_euler('xyz',a[3:6]*np.pi/180)
    rotation_res.append(float((expected.inv()*rot(target[3:7])).magnitude()))
assert max(translation_res)<2e-7 and max(rotation_res)<2e-7
assert max(r['adapter_controller_target_max_difference'] for r in rows)==0
assert len(rows)==500
prior=[json.loads(s) for s in (ROOT/'results/validation/peg/f3_diagnostic_20260904/trajectory_checkpoint9000.jsonl').read_text().splitlines()]
assert len(prior)==len(rows)
trajectory_equal=all(np.array_equal(a['eef_position'],b['actual_ee_position']) and np.array_equal(a['action'],b['policy_action']) and a['reward_total']==b['reward'] for a,b in zip(prior,rows))
assert trajectory_equal
dump('verification.json',dict(trajectory_exact_position_action_reward_match=trajectory_equal,steps=500,finite_trace=True,
    timestamp_note='raw MuJoCo data.time includes reset settling: first post-action0.04; episode elapsed time=step*0.02 (first0.02).',
    action_reconstruction='PASS',no_extra_episode=True))
active_rot=int(np.count_nonzero(np.any(np.abs(actions[:,3:6])>1e-8,axis=1)))
coverage=dict(total_transitions=len(actions),episodes=int(sum(bool(d['dones']) for d in all_demo)),
    nonzero_rotation_transitions=active_rot,rotation_max_abs=float(np.max(np.abs(actions[:,3:6]))),
    unique_gripper_actions=np.unique(actions[:,6]).tolist(),
    goal_position_range=np.ptp(obs[:,:3]+obs[:,13:16],axis=0).tolist(),
    quaternion_norm_max_error=float(np.max(np.abs(np.linalg.norm(obs[:,3:7],axis=1)-1))),
    note='Full-dataset coverage count is offline; detailed per-transition realized-motion comparison uses first five episodes (1328 transitions).')
dump('demo_coverage.json',coverage)
dump('action_contract.json',dict(policy_shape=[7],policy='bounded tanh action; deterministic argmax inference',
    translation=dict(indices='0:3',frame='world',unit='normalized command -> meters per target update',scale=.002,clip=[-1,1],
        integration='previous commanded target, NOT measured EE pose',workspace='componentwise min(reset EE,goal)-0.05 to max(reset EE,goal)+0.05'),
    rotation=dict(indices='3:6',representation='roll,pitch,yaw increment; Rz(yaw) Ry(pitch) Rx(roll)',frame='local commanded-target frame (right multiplication)',unit='radians after scale',scale=float(np.pi/180),clip=[-1,1],
        integration='R_target_next = R_target_prev @ R_delta',quaternion='wxyz throughout MuJoCo/adapter/IK'),
    gripper=dict(index=6,semantics='a>0 -> 1 close; a<=0 -> 0 open. Current assemble_axle is rigid and generated model has only q1..q6 actuators: no gripper actuator, so channel6 is inert.',clip=[-1,1]),
    controller=dict(type=runtime['controller'],target='absolute world position + wxyz orientation',ee_site=runtime['ee_site'],
        output='six joint-position targets -> joint_controller(use_target_as_ctrl) -> data.ctrl -> ten physics substeps',
        control_dt=.02,physics_dt=.002,ik_regularization=.001,ik_radius=.1),
    runtime_action_bounds=runtime['action_low'],runtime_action_high=runtime['action_high'],
    reconstruction_max_translation_residual=max(translation_res),reconstruction_max_rotation_residual=max(rotation_res),
    adapter_to_controller_max_difference=0,status='PASS'))
schema=[]
for start,names,unit,frame in [(0,['eef_x','eef_y','eef_z'],'m','world'),(3,['eef_qw','eef_qx','eef_qy','eef_qz'],'unit quaternion','world orientation'),
    (7,['force_x','force_y','force_z'],'N','sensor site'),(10,['claimed_torque_x','claimed_torque_y','claimed_torque_z'],'claimed Nm; actual mixed N/N/Nm','BUG: [Fy,Fz,Tx]'),
    (13,['goal_minus_eef_x','goal_minus_eef_y','goal_minus_eef_z'],'m','world'),(16,['goal_qw','goal_qx','goal_qy','goal_qz'],'unit quaternion','world orientation')]:
    schema.extend(dict(index=start+i,name=name,unit=unit,frame=frame) for i,name in enumerate(names))
dump('observation_contract.json',dict(shape=[20],dtype='float32',schema=schema,
    gripper_state='ABSENT; target-pose accumulator also absent',orientation_error='2*acos(abs(dot(normalized actual quaternion, normalized goal quaternion))) radians',
    reward='-80*lateral -20*abs(axial) -2*orientation +2*(lateral<0.01)+25*success',
    success='lateral<=0.002 AND abs(axial)<=0.003 AND orientation<=0.05',
    comparison='Training, Demo collector, evaluator all use same PegStateSERLEnv; same ordering/scales. Shared sensor bug is semantic failure, not a detected train/eval permutation.',
    status='FAIL',reason='indices10:13 are not the declared torque components'))
dump('force_sensor_audit.json',dict(status='FAIL',sensors=runtime['sensors'],
    implementation='get_force_data slices by force_sensor_id; get_torque_data slices by torque_sensor_id',
    force=dict(id=0,adr=0,dim=3,actual_slice='0:3',expected_slice='0:3',status='PASS for this model only'),
    torque=dict(id=1,adr=3,dim=3,actual_slice='1:4',expected_slice='3:6',actual_fields=['Fy','Fz','Tx'],status='FAIL'),
    observed_mismatch_steps=sum(not np.allclose(r['observed_wrench'][3:6],r['correct_sensor_slices']['end_torque_sensor']) for r in rows),
    contact_telemetry='raw.data.ncon counts all active MuJoCo contacts, not peg-hole contact forces. force_norm uses sensor force channels (correct slice here); no force criterion in success.',
    sensor_cutoff='Generated XML end_force_sensor cutoff=50, end_torque_sensor cutoff=10; norm ceiling 50*sqrt(3)=86.60254 is not unclipped physical force.',
    limitation='This audit does not infer failure causality from force norm or fix sensor source. Same shared bug existed in training and evaluation.'))
keys=[]
qgoal=obs[0,16:20]
for r in rows:
    if r['step'] in (1,47,65,67,71,120,500):
        keys.append(dict(step=r['step'],translation_tracking_error=r['target_realized_translation_error'],rotation_tracking_error=r['target_realized_rotation_error'],
            requested_target_goal_orientation_error=float((rot(qgoal).inv()*rot(r['controller_target_orientation'])).magnitude()),
            actual_goal_orientation_error=float((rot(qgoal).inv()*rot(r['actual_ee_orientation'])).magnitude())))
summary=dict(PEG_F3_ACTION_CONTRACT_AUDIT='PASS',PEG_F3_DEMO_CONTRACT_AUDIT='FAIL',PEG_F3_CONTROLLER_TRACKING_AUDIT='FAIL',
    PEG_F3_OBSERVATION_CONTRACT_AUDIT='FAIL',PEG_F3_FORCE_SENSOR_MAPPING='FAIL',
    DEMO_TRANSLATION_CONTRACT_MATCH='PASS',DEMO_ROTATION_CONTRACT_MATCH='FAIL',
    demo_rotation_failure_reason='UNVERIFIED_NO_ROTATIONAL_EXCITATION; FAIL means required empirical validation cannot pass, not evidence of wrong rotational units/frame.',
    root_cause_classification='CONTRACT_MISMATCH',
    primary_root_cause='Confirmed torque observation address bug. Its causal contribution to zero success is unresolved; cannot claim it alone caused policy failure.',
    performance_hypothesis='POLICY_DATA_QUALITY plus CONTROLLER_TRACKING: policy commands orientation away from goal; large tracking error coexists. Demo provides no nonzero rotation actions.',
    next_minimal_fix='Separately authorize sensor slicing by model.sensor_adr[id]:adr+dim with dim==3 CPU regression; do not silently feed corrected observations to existing checkpoint or mix corrected data with old-schema Demo.',
    key_tracking_steps=keys,episode_count=1,steps=500,training_started=False,checkpoint_resumed=False,checkpoint_modified=False,cloud_modified=False,
    tracking_note='FAIL describes failure to closely realize commanded pose on this trajectory; does not uniquely attribute it to an IK implementation bug versus actuator/contact/unreachable-target effects.',
    demo_coverage=coverage,
    source_unchanged=json.loads((OUT/'source_before.json').read_text())==json.loads((OUT/'source_after.json').read_text()))
dump('contract_audit_summary.json',summary)
print(json.dumps(summary,indent=2))
