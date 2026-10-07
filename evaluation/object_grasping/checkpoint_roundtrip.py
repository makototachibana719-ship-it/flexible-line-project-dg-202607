"""Fresh Object DrQ legacy-Flax save/rebuild/restore/inference roundtrip."""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from flax import config as flax_config
from flax.training import checkpoints
from training.object_grasping.run_object_training import ROOT, ENV_SRC, adapt, assert_contract, load_config, DEFAULT_CONFIG
sys.path.insert(0,str(ROOT/'external/serl/serl_launcher'))
from serl_launcher.utils.launcher import make_drq_agent
def signature(tree): return [(jax.tree_util.keystr(p),list(np.shape(v)),str(np.asarray(jax.device_get(v)).dtype)) for p,v in jax.tree_util.tree_flatten_with_path(tree)[0]]
def make(env,cfg,obs): return make_drq_agent(seed=cfg['seed'],sample_obs=obs,sample_action=env.action_space.sample(),image_keys=['front','wrist'],encoder_type=cfg['training']['encoder_type'])
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--output-dir',type=Path,default=ROOT/'runtime/object_handoff/checkpoint_roundtrip'); a=p.parse_args(); out=a.output_dir.resolve()
 if ROOT not in out.parents or (out.exists() and any(out.iterdir())): raise ValueError('output must be empty and project-local')
 out.mkdir(parents=True,exist_ok=True); cfg=load_config(a.config); sys.path.insert(0,str(ENV_SRC)); import gym,franka_sim
 env=gym.make(cfg['environment']); native,_=env.reset(seed=cfg['seed']); obs=adapt(env,native); assert_contract(obs,env.action_space,cfg); old=flax_config.flax_use_orbax_checkpointing
 try:
  first=make(env,cfg,obs); sig=signature(first.state); flax_config.update('flax_use_orbax_checkpointing',False); saved=checkpoints.save_checkpoint(out,first.state,0,keep=1,overwrite=False); del first; gc.collect(); second=make(env,cfg,obs); restored=checkpoints.restore_checkpoint(out,second.state,step=0); agent=second.replace(state=restored); sig_pass=sig==signature(restored); action=np.asarray(jax.device_get(agent.sample_actions(observations=jax.device_put(obs),argmax=True))); infer=action.shape==(4,) and np.isfinite(action).all() and np.all(action>=-1) and np.all(action<=1); report={'checkpoint_save':'PASS','checkpoint_restore':'PASS' if sig_pass else 'FAIL','structure_match':sig_pass,'leaf_count':len(sig),'restored_step':int(np.asarray(restored.step)),'policy_inference':'PASS' if infer else 'FAIL','action_shape':list(action.shape),'saved_path':str(saved)}; (out/'roundtrip_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return 0 if sig_pass and infer else 1
 finally: flax_config.update('flax_use_orbax_checkpointing',old); env.close()
if __name__=='__main__': raise SystemExit(main())
