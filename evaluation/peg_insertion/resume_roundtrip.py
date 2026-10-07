"""Cross-process bounded Peg safe-resume validation; no learner update."""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,shutil,subprocess,sys
from pathlib import Path
import jax,numpy as np,yaml
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import *
from training.peg_insertion import resume as resume_lib
from training.peg_insertion.checkpoint import restore_fresh_state
ROOT=Path(__file__).resolve().parents[2];CFG=ROOT/'configs/peg_insertion/peg_rlpd_baseline_seed0.yaml';MANIFEST=ROOT/'deliverables/gpu_training_handoff/peg_insertion/demo_manifest.json';OUT=ROOT/'runtime/peg_handoff/resume_roundtrip_m1de_v2'
PKG=ROOT/'external/packages/peg_insertion/peg_insertion';sys.path[:0]=[str(PKG/'code/example/async_rlpd_peg_sim'),str(PKG/'code/env'),str(ROOT/'external/serl/serl_launcher')]
from peg_env_adapter import make_peg_state_env
from serl_launcher.utils.launcher import make_sac_agent,make_replay_buffer
def cfg():return yaml.safe_load(CFG.read_text())
def hashes(tree):return [hashlib.sha256(str(np.asarray(x).shape).encode()+str(np.asarray(x).dtype).encode()+np.asarray(jax.device_get(x)).tobytes()).hexdigest() for x in jax.tree_util.tree_leaves(tree)]
def objects(c,capacity=128):
 env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=5);agent=make_sac_agent(c['seed'],np.zeros(20,np.float32),np.zeros(7,np.float32));buf=make_replay_buffer(env,capacity=capacity,type='replay_buffer');return env,agent,buf
def transitions(c):return pickle.loads((ROOT/c['demo']['path']).read_bytes())
def create(c,out):
 out.mkdir(parents=True,exist_ok=False)
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  env,agent,buf=objects(c);install_actor_step_tracking(buf,13)
  for t in transitions(c)[:4]:buf.insert(t)
  cp=save_bundle(out,agent.state,buf,123,17,c,MANIFEST,'peg-resume-unit-run',{'identifier':'peg-m1de-bounded'})
  _,rp=sidecars(cp);(out/'create_evidence.json').write_text(json.dumps({'hashes':hashes(agent.state),'checkpoint_hash':sha256_file(cp),'replay_hash':sha256_file(rp),'size':len(buf),'index':buf._insert_index},indent=2));env.close()
def copy_bundle(out,name,replay=True):
 d=out/name;d.mkdir();cp=d/'checkpoint_123';shutil.copy2(out/'checkpoint_123',cp);m,r=sidecars(cp);shutil.copy2(out/'checkpoint_123_resume_metadata.json',m)
 if replay:shutil.copy2(out/'checkpoint_123_online_replay.pkl',r)
 return cp
def failures(c,out):
 result={}
 def closed(name,fn):
  try:fn()
  except Exception as e:result[name]={'result':'FAIL_CLOSED','error':type(e).__name__}
  else:raise AssertionError(name)
 closed('missing_replay_sidecar',lambda:load_bundle(copy_bundle(out,'fail_missing',False),c,MANIFEST,'peg-resume-unit-run'))
 def alter(name,key,value):
  cp=copy_bundle(out,name);m,_=sidecars(cp);x=json.loads(m.read_text());x[key]=value;m.write_text(json.dumps(x));return cp
 closed('wrong_replay_hash',lambda:load_bundle(alter('fail_replay','online_replay_sha256','0'*64),c,MANIFEST,'peg-resume-unit-run'))
 closed('wrong_demo_hash',lambda:load_bundle(alter('fail_demo','demo_sha256','0'*64),c,MANIFEST,'peg-resume-unit-run'))
 bad=dict(c);bad['seed']=99;closed('config_mismatch',lambda:load_bundle(out/'checkpoint_123',bad,MANIFEST,'peg-resume-unit-run'))
 closed('wandb_mismatch',lambda:load_bundle(out/'checkpoint_123',c,MANIFEST,'wrong-run'))
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  env,a,b=objects(c);install_actor_step_tracking(b,17);closed('checkpoint_collision',lambda:save_bundle(out,a.state,b,123,17,c,MANIFEST,'peg-resume-unit-run',{}));env.close()
 return result
def resume(c,out):
 cp=out/'checkpoint_123';before=sha256_file(cp);e=json.loads((out/'create_evidence.json').read_text());bundle=load_bundle(cp,c,MANIFEST,'peg-resume-unit-run')
 with portable_peg_runtime(ROOT),repaired_replay_datastore_insert():
  env,agent,buf=objects(c);state=restore_fresh_state(out,agent.state,123)
  if hashes(state)!=e['hashes']:raise AssertionError('agent mismatch')
  ri=restore_replay(buf,bundle['online_replay_path']);before_rng=pickle.loads(bundle['online_replay_path'].read_bytes())['rng_state'];after_rng=resume_lib._rng(buf)
  if pickle.dumps(before_rng)!=pickle.dumps(after_rng) or ri['size']!=e['size'] or ri['insert_index']!=e['index'] or ri['actor_steps']!=17:raise AssertionError('replay progress')
  batch=buf.sample(batch_size=2)
  demo=make_replay_buffer(env,capacity=10000,type='replay_buffer')
  for t in transitions(c):demo.insert(t)
  if len(demo)!=5512:raise AssertionError('Demo rebuild')
  obs,_=env.reset(seed=0);action=np.asarray(jax.device_get(agent.replace(state=state).sample_actions(observations=obs,argmax=True)))
  if action.shape!=(7,) or not np.isfinite(action).all() or not np.all(np.abs(action)<=1):raise AssertionError('inference')
  save_bundle(out,state,buf,125,17,c,MANIFEST,'peg-resume-unit-run',{'identifier':'resumed-next'})
  env.close()
 f=failures(c,out)
 if sha256_file(cp)!=before:raise AssertionError('old checkpoint changed')
 report={'agent_values':'MATCH','learner_step':123,'actor_step':17,'online_replay_size':ri['size'],'online_replay_index':ri['insert_index'],'online_replay_rng':'MATCH','sample_shapes':{k:list(np.asarray(v).shape) for k,v in batch.items()},'demo_rebuild_size':len(demo),'wandb_identity':'MATCH','restored_inference':'PASS','next_checkpoint_step':125,'old_checkpoint_unchanged':True,'failure_tests':f,'PEG_RESUME_GATE':'VALIDATED'}
 (out/'resume_roundtrip_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=('create','resume'));p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();c=cfg();out=a.output_dir.resolve()
 if ROOT not in out.parents:raise ValueError('project local output required')
 if a.phase=='create':create(c,out)
 elif a.phase=='resume':resume(c,out)
 else:
  if out.exists() and any(out.iterdir()):raise FileExistsError(out)
  base=[sys.executable,'-m','evaluation.peg_insertion.resume_roundtrip','--output-dir',str(out)];subprocess.run(base+['--phase','create'],check=True);subprocess.run(base+['--phase','resume'],check=True);print('PROCESS_RECONSTRUCTION: PASS')
if __name__=='__main__':main()
