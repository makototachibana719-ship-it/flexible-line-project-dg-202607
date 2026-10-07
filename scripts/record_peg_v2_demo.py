"""V2 pilot recorder. Default is read-only preflight, record requires two flags.

Human console controls: enter phase followed by7 normalized action values for
exactly one control step; q interrupts. No policy loading or automatic actions.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.peg_insertion.v2_recorder import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-type',choices=TYPES,default='standard')
    p.add_argument('--record',action='store_true');p.add_argument('--authorize-recording',action='store_true');a=p.parse_args()
    # All pilot data confined to a new V2 tree, never the old demos directory.
    base=(ROOT/'results/demos/peg_v2_pilot').resolve();out=a.output.resolve()
    require(base in out.parents,'output must be a new subdirectory of results/demos/peg_v2_pilot')
    result=preflight(out);print(json.dumps(result,indent=2))
    if not a.record:
        require(not a.authorize_recording,'record flag missing');return
    require(a.authorize_recording,'explicit recording authorization flag required')
    import os
    runtime=ROOT/'configs/peg_insertion/runtime_gpu_formal.env'
    os.environ.update(dict(s.split('=',1) for s in runtime.read_text().splitlines() if s.strip()))
    sys.path[:0]=[str(PKG/'code/example/async_rlpd_peg_sim'),str(PKG/'code/env')]
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    env=None;writer=None
    try:
        with portable_peg_runtime(ROOT):env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=500)
        env.reset(seed=0);initial=snapshot(env);validate_snapshot(initial)
        writer=EpisodeWriter(out,a.episode_type,initial,result['hashes'],bindings)
        print('Fixed reset; all steps including setup are retained. Phases:',PHASES)
        for index in range(1,501):
            print(json.dumps(dict(step=index,metrics=writer.previous['metrics'],actual_ee_pose=writer.previous['actual_ee_pose'])))
            command=input('phase a0 a1 a2 a3 a4 a5 a6 (q=interrupt): ').strip()
            if command=='q':raise InterruptedError('operator quit')
            parts=command.split();require(len(parts)==8 and parts[0] in PHASES,'phase plus7 actions required')
            action=vector([float(x) for x in parts[1:]],7).astype(np.float32);require(np.max(np.abs(action))<=1,'action bounds')
            before=snapshot(env);require(before==writer.previous,'unexpected state mutation')
            obs,reward,terminated,truncated,info=env.step(action);after=snapshot(env)
            require(np.array_equal(obs,np.asarray(after['obs'],dtype=np.float32)),'returned observation mismatch')
            row=dict(observation_contract_version=V2,episode_type=a.episode_type,step=index,phase='success' if info['is_success'] else parts[0],
                obs=before['obs'],action=action.tolist(),next_obs=after['obs'],reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=bool(info['is_success']),before=before,after=after,hashes=result['hashes'])
            writer.append(row)
            if terminated or truncated:writer.finish();break
        if not writer.closed:raise InterruptedError('horizon without terminal state')
    except BaseException as e:
        if writer is not None:writer.abort(repr(e))
        raise
    finally:
        if env is not None:env.close()
if __name__=='__main__':main()
