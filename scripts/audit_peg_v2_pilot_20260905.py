"""Read-only audit of the stopped V2 standard pilot; no simulator imports."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import bindings,sha,validate_snapshot,validate_transition
OUT=ROOT/'results/validation/peg/f3_v2_pilot_20260905'
DEMO=ROOT/'results/demos/peg_v2_pilot/standard_20260905_001'
def main():
    before={p.name:sha(p) for p in DEMO.iterdir() if p.is_file()}
    episode=json.loads((DEMO/'episode.json').read_text());status=json.loads((DEMO/'status.json').read_text())
    rows=[json.loads(line) for line in (DEMO/'transitions.jsonl').open()]
    assert len(rows)==126
    binding=episode['hashes'];assert bindings()==binding
    previous=episode['initial'];validate_snapshot(previous)
    for i,row in enumerate(rows,1):
        validate_transition(row,previous,i,'standard',binding)
        np.testing.assert_allclose(np.array(row['after']['obs'][:3])+np.array(row['after']['obs'][13:16]),[.5601,0,.526],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(row['after']['obs'][16:20],episode['initial']['obs'][16:20])
        previous=row['after']
    assert not any(r['success'] or r['truncated'] for r in rows)
    assert status['status']=='INTERRUPTED' and status['eligible'] is False
    assert before=={p.name:sha(p) for p in DEMO.iterdir() if p.is_file()}
    start=rows[105]['after'];end=rows[-1]['after']
    insertion=rows[106:]
    position_errors=[float(np.linalg.norm(np.array(r['after']['target_pose'][:3])-r['after']['actual_ee_pose'][:3])) for r in insertion]
    summary=dict(
        audit='PASS',pilot_acceptance='FAIL',stop_reason='CONTROLLER_TRACKING_DIVERGENCE',
        source='AI_AGENT_REALTIME_STATE_FEEDBACK_SIMULATED_DEMONSTRATION_NOT_HUMAN',
        transitions=len(rows),success=False,truncated=False,quarantined=True,
        source_config_runtime_env_recorder_hashes_unchanged=True,
        fixed_goal_unchanged=True,contract_sensor_continuity_finite_reward_audit='PASS',
        initial_metrics=episode['initial']['metrics'],final_metrics=end['metrics'],
        best_lateral_error=min(r['after']['metrics']['lateral_error'] for r in rows),
        near_hole_transitions=sum(r['after']['metrics']['lateral_error']<=.01 and r['after']['metrics']['axial_error']<=.02 for r in rows),
        insertion_window=dict(first_step=107,last_step=126,
          requested_x_delta=sum(r['action'][0]*.002 for r in insertion),
          target_x_delta=end['target_pose'][0]-start['target_pose'][0],
          realized_x_delta=end['actual_ee_pose'][0]-start['actual_ee_pose'][0],
          tracking_errors=position_errors,tracking_error_final=position_errors[-1],
          orientation_error_before=start['metrics']['orientation_error'],orientation_error_after=end['metrics']['orientation_error']),
        artifact_hashes=before,
        note='Recorder status says operator quit because q was used for controlled safety stop. This sidecar documents the actual tracking stop reason without changing recorded data. No simulator crash, retry or next pilot.')
    (OUT/'standard_audit.json').write_text(json.dumps(summary,indent=2))
    result=dict(PEG_F3_V2_PILOT_STANDARD='FAIL',PEG_F3_V2_PILOT_LATERAL_PLUS='NOT_RUN',
      PEG_F3_V2_PILOT_LATERAL_MINUS='NOT_RUN',PEG_F3_V2_PILOT_ROTATION_ROLL='NOT_RUN',PEG_F3_V2_PILOT_ROTATION_PITCH='NOT_RUN',
      PEG_F3_V2_PILOT_COLLECTION='FAIL',successful_pilots=0,failed_pilots=1,unattempted_pilots=4,
      demo_recording_started=True,training_started=False,checkpoint_modified=False,pilot_promoted_to_training=False,
      retry_count=0,stop_reason=summary['stop_reason'])
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
