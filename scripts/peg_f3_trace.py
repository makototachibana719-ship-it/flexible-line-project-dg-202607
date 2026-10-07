"""Post-step passive telemetry. No environment or policy writes."""
import json
import numpy as np
def record(out,checkpoint,step,obs,action,reward,terminated,truncated,info,env):
    ratios=[float(info[k])/tol for k,tol in [('lateral_error',.002),('axial_error',.003),('orientation_error',.05)]]
    components=dict(lateral=-80*float(info['lateral_error']),axial=-20*float(info['axial_error']),orientation=-2*float(info['orientation_error']),
                    alignment_bonus=2.*(info['lateral_error']<.01),success_bonus=25.*bool(info['is_success']))
    row=dict(step=step,sim_time=step*.02,eef_position=obs[:3].tolist(),target_position=(obs[:3]+obs[13:16]).tolist(),relative_position=obs[13:16].tolist(),
             action=action.tolist(),action_saturation_flags=(np.abs(action)>=.99).tolist(),reward_total=float(reward),reward_components=components,
             terminated=bool(terminated),truncated=bool(truncated),success=bool(info['is_success']),lateral_ratio=ratios[0],axial_ratio=ratios[1],orientation_ratio=ratios[2],simultaneous_threshold_ratio=max(ratios))
    for key in ('position_error','lateral_error','axial_error','orientation_error','force_norm'):row[key]=float(info[key])
    # Auxiliary passive contact/sensor facts; not a calibrated rejection classifier.
    row['contact_count']=int(env.env.data.ncon)
    row['observed_force_xyz']=obs[7:10].tolist()
    with (out/f'trajectory_checkpoint{checkpoint}.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
