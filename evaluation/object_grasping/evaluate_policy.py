"""Independent Object policy evaluation schema and bounded dry run."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from training.object_grasping.run_object_training import ROOT, ENV_SRC, adapt, assert_contract, load_config, DEFAULT_CONFIG
FIELDS=['episode','success','return','episode_length','max_lift_height','final_block_height','failure_case']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--checkpoint'); p.add_argument('--episodes',type=int,default=50); p.add_argument('--output-dir',type=Path,default=ROOT/'results/single_skill/object'); p.add_argument('--dry-run-one-step',action='store_true'); a=p.parse_args(); cfg=load_config(a.config); sys.path.insert(0,str(ENV_SRC)); import gym,franka_sim
 env=gym.make(cfg['environment']); native,_=env.reset(seed=cfg['seed']); obs=adapt(env,native); assert_contract(obs,env.action_space,cfg)
 if a.dry_run_one_step:
  result=env.step(np.zeros((4,),np.float32)); print('RESET_ONE_STEP_DRY_RUN: PASS'); print('STEP_API_LENGTH:',len(result)); env.close(); return 0
 if not a.checkpoint: raise ValueError('--checkpoint required for formal evaluation')
 raise NotImplementedError('trained-policy restore must pass the new-checkpoint gate on GPU before formal evaluation')
if __name__=='__main__': raise SystemExit(main())
