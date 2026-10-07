"""Freeze and validate review specifications; pure stdlib, no rollout."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/validation/peg/f3_data_review_20260904'
def read(n):return json.loads((OUT/n).read_text())
pilot=read('v2_pilot_protocol.json');spec=read('v2_dataset_acceptance_spec.json');summary=read('data_review_summary.json');episodes=read('episode_audit.json')
assert len(pilot['pilots'])==5 and len(episodes)==20 and sum(r['transitions'] for r in episodes)==5512
assert all(r['success'] for r in episodes) and summary['rotation_nonzero_total']==0
assert pilot['invariants']['max_steps']==500 and pilot['obs_contract']=='PEG_OBS_CONTRACT_V2_SENSOR_FIXED'
assert set(spec['checks'])=={'shape','finite','continuity','contract','no_mixing','hashes','orientation_correction','success','near_hole','lateral_correction','provenance','release'}
paths=[OUT/'v2_pilot_protocol.json',OUT/'v2_dataset_acceptance_spec.json',OUT/'analysis_protocol.json',
 ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml',ROOT/'configs/peg_insertion/runtime_gpu_formal.env',
 ROOT/'training/peg_insertion/observation_contract.py',ROOT/'configs/peg_insertion/legacy_observation_contracts.json']
pkg=ROOT/'external/packages/peg_insertion/peg_insertion/code'
paths += [pkg/'example/async_rlpd_peg_sim/record_peg_serl_demos.py',pkg/'example/async_rlpd_peg_sim/peg_env_adapter.py',
 pkg/'env/mujoco_env/mujoco_env/utils/sensor_utils.py',pkg/'env/mujoco_env/mujoco_env/controllers/cartesian_ik.py',pkg/'env/mujoco_env/mujoco_env/tasks/peg_insertion/peg_insertion.py']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'frozen_spec_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={k:summary[k] for k in ('PEG_F3_V1_DEMO_AUDIT','PEG_F3_ROTATION_DATA_COVERAGE')}
result.update(PEG_F3_V2_DEMO_PROTOCOL='PASS',PEG_F3_V2_DATASET_ACCEPTANCE_SPEC='PASS',V2_DEMO_COLLECTION='BLOCKED',
 reason='Protocol frozen, but existing recorder lacks V2/raw sensor/hash/phase enforcement. No recording authorization consumed.',
 training_started=False,demo_recording_started=False,checkpoint_modified=False,simulator_rollout_started=False,
 final_dataset_accepted=False)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
