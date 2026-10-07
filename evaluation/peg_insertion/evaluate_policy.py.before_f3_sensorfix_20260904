"""Independent Peg evaluation with source-identical success semantics."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import jax,numpy as np,yaml
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.checkpoint import restore_fresh_state
ROOT=Path(__file__).resolve().parents[2];CFG=ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml';PKG=ROOT/'external/packages/peg_insertion/peg_insertion';sys.path[:0]=[str(PKG/'code/example/async_rlpd_peg_sim'),str(PKG/'code/env'),str(ROOT/'external/serl/serl_launcher')]
from peg_env_adapter import make_peg_state_env
from serl_launcher.utils.launcher import make_sac_agent
FIELDS=['episode','success','return','episode_length','simulated_control_seconds','wall_seconds','final_lateral_error','final_axial_error','final_orientation_error','final_position_error','mean_force','max_force','terminated','truncated','failure_category']
def category(success,terminated,truncated,info):
 if success:return 'SUCCESS'
 vals=[info.get(k,np.nan) for k in ('lateral_error','axial_error','orientation_error')]
 if not np.isfinite(vals).all():return 'NONFINITE'
 if truncated:return 'TIMEOUT'
 if vals[0]>.002:return 'LATERAL_MISS'
 if vals[1]>.003:return 'AXIAL_MISS'
 if vals[2]>.05:return 'ORIENTATION_MISS'
 return 'OTHER'
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CFG);p.add_argument('--checkpoint',type=Path);p.add_argument('--episodes',type=int,default=50);p.add_argument('--output-dir',type=Path,default=ROOT/'results/single_skill/peg/formal_seed0');p.add_argument('--dry-run-one-step',action='store_true');a=p.parse_args();c=yaml.safe_load(a.config.read_text())
 with portable_peg_runtime(ROOT):env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=500)
 obs,_=env.reset(seed=c['seed'])
 if a.dry_run_one_step:
  result=env.step(np.zeros(7,np.float32));print('RESET_ONE_STEP_DRY_RUN: PASS');print('CONTROL_DT_SECONDS: 0.02');print('STEP_API_LENGTH:',len(result));env.close();return
 if not a.checkpoint:raise ValueError('--checkpoint required')
 agent=make_sac_agent(c['seed'],np.zeros(20,np.float32),np.zeros(7,np.float32));step=int(a.checkpoint.name.split('_')[-1]);agent=agent.replace(state=restore_fresh_state(a.checkpoint.parent,agent.state,step));rows=[]
 for ep in range(a.episodes):
  obs,_=env.reset(seed=c['seed']+ep);ret=0.;forces=[];info={};terminated=truncated=False;t0=time.perf_counter()
  for i in range(500):
   action=np.asarray(jax.device_get(agent.sample_actions(observations=obs,argmax=True)));obs,reward,terminated,truncated,info=env.step(action);ret+=float(reward);forces.append(float(info.get('force_norm',0.)))
   if terminated or truncated:break
  wall=time.perf_counter()-t0;success=bool(info.get('is_success',terminated and not truncated));rows.append({'episode':ep,'success':success,'return':ret,'episode_length':i+1,'simulated_control_seconds':(i+1)*0.02,'wall_seconds':wall,'final_lateral_error':float(info.get('lateral_error',np.nan)),'final_axial_error':float(info.get('axial_error',np.nan)),'final_orientation_error':float(info.get('orientation_error',np.nan)),'final_position_error':float(info.get('position_error',np.nan)),'mean_force':float(np.mean(forces)),'max_force':float(np.max(forces)),'terminated':bool(terminated),'truncated':bool(truncated),'failure_category':category(success,terminated,truncated,info)})
 a.output_dir.mkdir(parents=True,exist_ok=False)
 with (a.output_dir/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 success=np.asarray([r['success'] for r in rows]);returns=np.asarray([r['return'] for r in rows]);length=np.asarray([r['episode_length'] for r in rows]);summary={'episodes':len(rows),'success_count':int(success.sum()),'success_rate':float(success.mean()),'return_mean':float(returns.mean()),'return_std':float(returns.std()),'episode_length_mean':float(length.mean()),'episode_length_std':float(length.std()),'control_dt_seconds':0.02,'simulated_time_definition':'episode_length * control_dt','wall_seconds_mean':float(np.mean([r['wall_seconds'] for r in rows])),'final_lateral_error_mean':float(np.mean([r['final_lateral_error'] for r in rows])),'final_axial_error_mean':float(np.mean([r['final_axial_error'] for r in rows])),'final_orientation_error_mean':float(np.mean([r['final_orientation_error'] for r in rows])),'max_force_mean':float(np.mean([r['max_force'] for r in rows])),'failure_categories':{x:sum(r['failure_category']==x for r in rows) for x in sorted(set(r['failure_category'] for r in rows))}}
 (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');env.close();print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
