"""Fail-closed Object resume bundles built on the validated Cable sidecar format."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from flax import config as flax_config
from flax.training import checkpoints
from training.cable.resume import (canonical_hash, install_actor_step_tracking,
 load_resume_bundle, restore_online_replay, sha256_file, sidecar_paths,
 snapshot_online_replay, write_resume_metadata)

REQUIRED_METADATA={"checkpoint_step","learner_update_steps","actor_steps","seed",
 "environment","algorithm","observation","action","demo_sha256",
 "demo_transition_count","config_hash","wandb_project","wandb_name",
 "wandb_run_id","online_replay_sha256","created_at","source_snapshot"}

def legacy_save(directory:Path,target:Any,step:int,keep:int=20)->str:
 old=flax_config.flax_use_orbax_checkpointing; flax_config.update('flax_use_orbax_checkpointing',False)
 try:return checkpoints.save_checkpoint(directory,target,step=step,keep=keep,overwrite=False)
 finally:flax_config.update('flax_use_orbax_checkpointing',old)
def legacy_restore(directory:Path,target:Any,step:int):
 old=flax_config.flax_use_orbax_checkpointing; flax_config.update('flax_use_orbax_checkpointing',False)
 try:return checkpoints.restore_checkpoint(directory,target,step=step)
 finally:flax_config.update('flax_use_orbax_checkpointing',old)
def validate_metadata_schema(metadata:dict[str,Any])->None:
 missing=REQUIRED_METADATA-set(metadata)
 if missing: raise ValueError(f'missing resume metadata fields: {sorted(missing)}')
 if metadata['checkpoint_step']!=metadata['learner_update_steps']: raise ValueError('learner/checkpoint step mismatch')
def object_bundle(checkpoint_path:Path,config:dict[str,Any],manifest_path:Path,run_id:str|None):
 metadata=load_resume_bundle(checkpoint_path,expected_environment=config['environment'],expected_algorithm=config['algorithm'],expected_seed=int(config['seed']),expected_config_hash=canonical_hash(config),expected_demo_manifest_hash=sha256_file(manifest_path),expected_observation=config['observation'],expected_action=config['action'],expected_wandb_project=config['wandb']['project'],expected_wandb_name=config['wandb']['run_name'],explicit_wandb_run_id=run_id)
 validate_metadata_schema(metadata)
 if metadata['demo_sha256']!=config['demo']['sha256'] or metadata['demo_transition_count']!=config['demo']['raw_transitions']: raise ValueError('Demo identity mismatch')
 return metadata
def save_bundle(directory:Path,state:Any,replay:Any,step:int,actor_steps:int,config:dict[str,Any],manifest_path:Path,wandb_run_id:str,source_snapshot:dict[str,Any])->Path:
 directory=directory.resolve(); cp=directory/f'checkpoint_{step}'; meta,replay_path=sidecar_paths(cp)
 for path in (cp,meta,replay_path):
  if path.exists(): raise FileExistsError(f'refusing checkpoint collision: {path}')
 saved=Path(legacy_save(directory,state,step)); replay.project_actor_steps=int(actor_steps); info=snapshot_online_replay(replay,replay_path)
 write_resume_metadata(meta,{"format_version":1,"checkpoint_step":step,"learner_update_steps":step,"actor_steps":actor_steps,"seed":int(config['seed']),"environment":config['environment'],"algorithm":config['algorithm'],"observation":config['observation'],"action":config['action'],"demo_sha256":config['demo']['sha256'],"demo_transition_count":config['demo']['raw_transitions'],"demo_manifest_hash":sha256_file(manifest_path),"config_hash":canonical_hash(config),"wandb_project":config['wandb']['project'],"wandb_name":config['wandb']['run_name'],"wandb_run_id":wandb_run_id,"online_replay_file":info['path'],"online_replay_sha256":info['sha256'],"online_replay_size":info['size'],"online_replay_capacity":info['capacity'],"source_snapshot":source_snapshot})
 return saved
