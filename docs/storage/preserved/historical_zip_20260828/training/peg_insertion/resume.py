"""Fail-closed Peg state-replay resume bundles."""
from __future__ import annotations
import hashlib,json,os,pickle
from datetime import datetime,timezone
from pathlib import Path
from types import MethodType
from typing import Any
import numpy as np
from training.peg_insertion.checkpoint import save_fresh_state,restore_fresh_state

REQUIRED_METADATA={"checkpoint_step","learner_update_steps","actor_steps","seed","environment","algorithm","observation","action","demo_sha256","demo_transition_count","config_hash","wandb_project","wandb_name","wandb_run_id","online_replay_sha256","created_at","source_snapshot"}
def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical_hash(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sidecars(cp:Path):return cp.parent/f'{cp.name}_resume_metadata.json',cp.parent/f'{cp.name}_online_replay.pkl'
def install_actor_step_tracking(buf,steps=0):
 if hasattr(buf,'project_original_insert'):raise RuntimeError('tracking already installed')
 buf.project_actor_steps=int(steps);buf.project_original_insert=buf.insert
 def tracked(self,data):self.project_original_insert(data);self.project_actor_steps+=1
 buf.insert=MethodType(tracked,buf)
def _rng(buf):
 r=buf.np_random;return {'kind':'generator','state':r.bit_generator.state} if hasattr(r,'bit_generator') else {'kind':'random_state','state':r.get_state()}
def _set_rng(buf,s):
 r=buf.np_random
 if s['kind']=='generator' and hasattr(r,'bit_generator'):r.bit_generator.state=s['state']
 elif s['kind']=='random_state' and hasattr(r,'set_state'):r.set_state(s['state'])
 else:raise ValueError('replay RNG mismatch')
def snapshot_replay(buf,path:Path):
 if path.exists():raise FileExistsError(path)
 n=len(buf);payload={'format_version':1,'capacity':int(buf._capacity),'size':n,'insert_index':int(buf._insert_index),'dataset_dict':{k:np.asarray(v)[:n].copy() for k,v in buf.dataset_dict.items()},'rng_state':_rng(buf),'actor_steps':int(buf.project_actor_steps)}
 tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('xb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);return {'sha256':sha256_file(path),'size':n,'capacity':int(buf._capacity),'actor_steps':int(buf.project_actor_steps)}
def restore_replay(buf,path:Path):
 p=pickle.loads(path.read_bytes())
 if p.get('format_version')!=1 or int(p['capacity'])!=int(buf._capacity):raise ValueError('replay format/capacity mismatch')
 n=int(p['size']);
 if not 0<=n<=buf._capacity or set(p['dataset_dict'])!=set(buf.dataset_dict):raise ValueError('replay payload mismatch')
 for k,v in p['dataset_dict'].items():
  t=np.asarray(buf.dataset_dict[k]);s=np.asarray(v)
  if t.shape[1:]!=s.shape[1:] or t.dtype!=s.dtype:raise ValueError(f'replay {k} mismatch')
  t[:n]=s
 buf._size=n;buf._insert_index=int(p['insert_index']);_set_rng(buf,p['rng_state']);install_actor_step_tracking(buf,p['actor_steps'])
 return {'size':n,'insert_index':int(buf._insert_index),'actor_steps':int(p['actor_steps']),'sha256':sha256_file(path)}
def save_bundle(directory:Path,state,replay,step:int,actor_steps:int,cfg:dict,manifest:Path,run_id:str,source:dict)->Path:
 directory=directory.resolve();cp=directory/f'checkpoint_{step}';meta,rp=sidecars(cp)
 for p in (cp,meta,rp):
  if p.exists():raise FileExistsError(f'checkpoint collision: {p}')
 cp=save_fresh_state(directory,state,step);replay.project_actor_steps=int(actor_steps);ri=snapshot_replay(replay,rp)
 payload={'format_version':1,'checkpoint_step':step,'learner_update_steps':step,'actor_steps':actor_steps,'seed':int(cfg['seed']),'environment':cfg['environment'],'algorithm':cfg['algorithm'],'observation':cfg['observation'],'action':cfg['action'],'demo_sha256':cfg['demo']['sha256'],'demo_transition_count':cfg['demo']['transitions'],'demo_manifest_hash':sha256_file(manifest),'config_hash':canonical_hash(cfg),'wandb_project':cfg['wandb']['project'],'wandb_name':cfg['wandb']['run_name'],'wandb_run_id':run_id,'online_replay_file':rp.name,'online_replay_sha256':ri['sha256'],'online_replay_size':ri['size'],'online_replay_capacity':ri['capacity'],'source_snapshot':source,'created_at':datetime.now(timezone.utc).isoformat()}
 tmp=meta.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,meta);return cp
def load_bundle(cp:Path,cfg:dict,manifest:Path,run_id:str|None):
 cp=cp.resolve();meta,rp=sidecars(cp)
 if not cp.is_file() or not meta.is_file() or not rp.is_file():raise FileNotFoundError('resume requires checkpoint + replay + metadata')
 m=json.loads(meta.read_text());missing=REQUIRED_METADATA-set(m)
 if missing:raise ValueError(f'missing metadata: {sorted(missing)}')
 expected={'checkpoint_step':int(cp.name.split('_')[-1]),'learner_update_steps':int(cp.name.split('_')[-1]),'seed':int(cfg['seed']),'environment':cfg['environment'],'algorithm':cfg['algorithm'],'observation':cfg['observation'],'action':cfg['action'],'demo_sha256':cfg['demo']['sha256'],'demo_transition_count':cfg['demo']['transitions'],'demo_manifest_hash':sha256_file(manifest),'config_hash':canonical_hash(cfg),'wandb_project':cfg['wandb']['project'],'wandb_name':cfg['wandb']['run_name']}
 for k,v in expected.items():
  if m.get(k)!=v:raise ValueError(f'{k} mismatch')
 if run_id and m['wandb_run_id']!=run_id:raise ValueError('W&B mismatch')
 if sha256_file(rp)!=m['online_replay_sha256']:raise ValueError('replay hash mismatch')
 m.update(checkpoint_path=cp,online_replay_path=rp);return m
