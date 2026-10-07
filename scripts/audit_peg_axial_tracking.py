"""Offline existing-trace audit. No simulator, IK invocation or rollout."""
import json,sys,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import bindings,sha

def main():
    out=ROOT/'results/validation/peg/f3_axial_tracking_audit_20260905';out.mkdir(exist_ok=False)
    demo=ROOT/'results/demos/peg_v2_pilot/standard_20260905_001'
    protected={str(p):sha(p) for p in demo.iterdir() if p.is_file()}
    ep=json.loads((demo/'episode.json').read_text());allrows=[json.loads(x) for x in (demo/'transitions.jsonl').open()]
    assert bindings()==ep['hashes']
    initial=np.array(ep['initial']['obs'],dtype=np.float32)
    goal=initial[:3]+initial[13:16];lo=np.minimum(initial[:3],goal)-np.float32(.05);hi=np.maximum(initial[:3],goal)+np.float32(.05)
    rows=[]
    for r in allrows:
        if not 100<=r['step']<=126:continue
        b,a=r['before'],r['after'];action=np.clip(np.array(r['action'],dtype=np.float32),-1,1)
        raw=np.array(b['target_pose'][:3],dtype=np.float32)+action[:3]*.002
        clipped=np.clip(raw,lo,hi);np.testing.assert_array_equal(clipped,np.array(a['target_pose'][:3],dtype=np.float32))
        target=np.array(a['target_pose']);base_clipped=np.clip(target,[-10]*3+[-6.28]*4,[10]*3+[6.28]*4)
        rows.append(dict(step=r['step'],actual_ee_pose=a['actual_ee_pose'],adapter_target_pose=a['target_pose'],
            controller_target_pose_static_inference=base_clipped.tolist(),controller_target_directly_logged=False,
            action=r['action'],requested_delta=(action[:3]*.002).tolist(),
            workspace_before_clip=raw.tolist(),workspace_after_clip=clipped.tolist(),workspace_clip_flags=(raw!=clipped).tolist(),
            base_action_clip_flags=(target!=base_clipped).tolist(),goal_relative_position=a['obs'][13:16],goal_quaternion=a['obs'][16:20],
            qpos=None,ik_q_target=None,q_target_minus_current_q=None,ik_residual=None,ik_status=None,
            distance_to_joint_limits=None,actuator_ctrl=None,actuator_clip_flags=None,contact_pairs=None,
            requested_axial_delta=float(action[0]*.002),ik_implied_axial_progress=None,
            realized_axial_delta=float(a['actual_ee_pose'][0]-b['actual_ee_pose'][0]),
            adapter_target_axial_delta=float(a['target_pose'][0]-b['target_pose'][0]),
            target_tracking_error=float(np.linalg.norm(target[:3]-a['actual_ee_pose'][:3])),
            missing_reason='Not recorded. EE pose does not uniquely recover joint trajectory, solver state, actuator controls or contacts.'))
    xml=ROOT/'runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml'
    tree=ET.parse(xml).getroot()
    geometry=[]
    def visit(node,path='world'):
        if node.tag=='body':path+='/'+node.get('name','<unnamed_body>')
        for child in node:
            if child.tag=='geom':geometry.append(dict(body_path=path,name=child.get('name'),attributes=child.attrib,observed_contact=False))
            visit(child,path)
    visit(tree)
    static=dict(xml_path=str(xml),sha256=sha(xml),scope='Current generated XML only; not a time-resolved contact log or proof of runtime joint binding',
      joints=[e.attrib for e in tree.findall('.//joint') if e.get('name')],actuators=[e.attrib for e in tree.findall('./actuator/*')],
      geoms=geometry,includes=[e.attrib for e in tree.findall('.//include')],defaults=[ET.tostring(e,encoding='unicode') for e in tree.findall('./default')])
    log=(ROOT/'results/validation/peg/f3_v2_pilot_20260905/standard_recorder.log').read_text()
    insertion=[r for r in rows if r['step']>=107]
    result=dict(PEG_F3_AXIAL_TARGET_AUDIT='PASS',AXIAL_TARGET_ACCUMULATION='PASS',WORKSPACE_CLIP_BLOCKING='NO',
      PEG_F3_IK_AUDIT='FAIL',IK_TARGET_PROGRESS='UNRESOLVED',IK_STAGNATION_STEP=None,IK_RESIDUAL_TREND='UNAVAILABLE',
      PEG_F3_JOINT_LIMIT_AUDIT='FAIL',JOINT_LIMIT_BLOCKING='UNRESOLVED',ACTUATOR_CLIPPING='UNRESOLVED',
      PEG_F3_GEOMETRY_CONTACT_AUDIT='UNRESOLVED',PEG_HOLE_CONTACT='UNRESOLVED',ROBOT_FIXTURE_CONTACT='UNRESOLVED',CONTACT_BLOCKING='UNRESOLVED',
      primary_axial_blocker='UNRESOLVED',secondary_axial_blocker='UNRESOLVED',root_cause_classification='UNRESOLVED',
      first_blocking_step=None,first_observed_axial_nonresponse_step=107,
      audit_fail_semantics='IK/joint audit FAIL means insufficient historical telemetry, not confirmed IK/limit failure.',
      requested_axial_total=sum(r['requested_axial_delta'] for r in insertion),
      realized_axial_total=sum(r['realized_axial_delta'] for r in insertion),
      target_axial_total=sum(r['adapter_target_axial_delta'] for r in insertion),
      target_error_start=insertion[0]['target_tracking_error'],target_error_end=insertion[-1]['target_tracking_error'],
      ik_exception_message_present='IK求解失败' in log,
      ik_exception_absence_not_solver_success=True,
      known_localization='Downstream of accumulated Cartesian target; IK vs actuation vs physical constraint cannot be separated.',
      next_minimal_fix='No causal repair justified. Separately authorize telemetry-only instrumentation for qpos/qvel, IK input/output/status/residual/Jacobian, joint/actuator bindings/limits/ctrl, and named contact pairs before any new bounded diagnostic. No replay/rollout authorized here.',
      demo_recording_started=False,training_started=False,checkpoint_modified=False,rollout_started=False)
    assert not any(any(r['workspace_clip_flags']) or any(r['base_action_clip_flags']) for r in rows)
    assert protected=={str(p):sha(p) for p in demo.iterdir() if p.is_file()}
    assert bindings()==ep['hashes']
    for name,value in [('steps100_126.json',rows),('static_model_inventory.json',static),('result.json',result),('input_hashes.json',protected)]:
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
