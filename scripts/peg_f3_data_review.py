"""Read-only V1 dataset analysis. No simulator, JAX, learner or recorder imports."""
import csv,hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'external/packages/peg_insertion/peg_insertion'
OUT=ROOT/'results/validation/peg/f3_data_review_20260904'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def stats(x):
    a=np.asarray(x)
    return dict(count=int(a.size),mean=float(a.mean()),min=float(a.min()),max=float(a.max()),median=float(np.median(a)),p95=float(np.percentile(a,95))) if a.size else dict(count=0,mean=None,min=None,max=None,median=None,p95=None)
def metrics(o):
    q=o[:,3:7].astype(float);g=o[:,16:20].astype(float)
    q/=np.linalg.norm(q,axis=1)[:,None];g/=np.linalg.norm(g,axis=1)[:,None]
    rel=o[:,13:16].astype(float)
    return np.linalg.norm(rel,axis=1),np.linalg.norm(rel[:,1:],axis=1),np.abs(rel[:,0]),2*np.arccos(np.clip(np.abs(np.sum(q*g,axis=1)),0,1))
def main():
    OUT.mkdir(exist_ok=False)
    # Frozen before computing near-hole counts. Region does not filter orientation.
    write('analysis_protocol.json',dict(near_hole=dict(lateral_le_m=.01,axial_le_m=.02,orientation_filter=None),
        definition='Post-transition next_obs, lateral<=10mm AND axial<=20mm; diagnostic region only, success thresholds unchanged.',
        step_counts='post-transition only; reset excluded',min_position='includes initial state',rotation_nonzero='any rotation component !=0 exactly',
        tracking='same-step measured world translation versus action[:3]*0.002; accumulated-target dynamics prevent exact equality expectations'))
    pkl=PKG/'demos/peg_insert_20_demos_state.pkl';files=sorted((PKG/'demos/peg_demos_v1').glob('demo_*.npz'))
    assert len(files)==20 and sha(pkl)=='44ee651768d9f231caccf81d5a58460c8f44bc17463de7e08ea157b3c2a53ce7'
    protected={str(p):sha(p) for p in [pkl,*files]};data=pickle.loads(pkl.read_bytes());offset=0;episodes=[];near_rows=[];orientation=[];errors=[]
    for ep,p in enumerate(files):
        with np.load(p,allow_pickle=False) as d:
            o=d['observations'];no=d['next_observations'];a=d['actions'];n=len(a);meta=json.loads(str(d['metadata_json']))
            assert o.shape==no.shape==(n,20) and a.shape==(n,7)
            assert all(np.isfinite(x).all() for x in (o,no,a,d['rewards']))
            assert np.array_equal(o[0],d['initial_observation']) and np.array_equal(no[:-1],o[1:])
            for i in range(n):
                for k in ('observations','next_observations','actions','rewards','dones','masks'):assert np.array_equal(data[offset+i][k],d[k][i]),(ep,i,k)
            pos,lat,ax,ori=metrics(no);initial=metrics(o[:1]);orientation.extend(ori.tolist())
            for name,v in [('position_error',pos),('lateral_error',lat),('axial_error',ax),('orientation_error',ori)]:
                residual=float(np.max(np.abs(d[name]-v)));errors.append(residual);assert residual<1e-5,(p,name,residual)
            # Use original stored metrics for exact threshold counts (float32 recomputation checked above).
            pos=d['position_error'];lat=d['lateral_error'];ax=d['axial_error'];ori=d['orientation_error']
            success=(lat<=.002)&(ax<=.003)&(ori<=.05)
            near=(lat<=.01)&(ax<=.02);rn=np.linalg.norm(a[:,3:6],axis=1)
            assert bool(success[-1])==bool(meta['success']) and bool(d['terminated'][-1]) and not bool(d['truncated'][-1])
            expected=-80*lat-20*ax-2*ori+2*(lat<.01)+25*success
            assert np.allclose(expected,d['rewards'],atol=1e-5,rtol=1e-6)
            req=a[:,:3]*.002;real=no[:,:3]-o[:,:3];reqn=np.linalg.norm(req,axis=1);realn=np.linalg.norm(real,axis=1);mask=(reqn>1e-8)&(realn>1e-8)
            cosine=np.sum(req[mask]*real[mask],axis=1)/(reqn[mask]*realn[mask])
            row=dict(episode=ep,file=p.name,transitions=n,success=bool(success[-1]),initial_position_error=float(initial[0][0]),final_position_error=float(pos[-1]),min_position_error=float(min(pos.min(),initial[0][0])),
                initial_orientation_error=float(initial[3][0]),mean_orientation_error=float(ori.mean()),min_orientation_error=float(ori.min()),max_orientation_error=float(ori.max()),final_orientation_error=float(ori[-1]),
                lateral_threshold_steps=int((lat<=.002).sum()),axial_threshold_steps=int((ax<=.003).sum()),orientation_threshold_steps=int((ori<=.05).sum()),all_three_steps=int(success.sum()),
                rotation_action_nonzero_count=int((rn!=0).sum()),rotation_action_norm=stats(rn),near_hole_transitions=int(near.sum()),
                translation_tracking=dict(active_pairs=int(mask.sum()),cosine=stats(cosine),realized_requested_norm_ratio=stats(realn[mask]/reqn[mask]),residual_m=stats(np.linalg.norm(real-req,axis=1))),rotation_tracking='UNVERIFIED_NO_EXCITATION')
            episodes.append(row)
            for i in np.flatnonzero(near):near_rows.append(dict(episode=ep,step=int(i+1),lateral_error=float(lat[i]),axial_error=float(ax[i]),orientation_error=float(ori[i]),translation_action=a[i,:3].tolist(),rotation_action=a[i,3:6].tolist(),successful_terminal=bool(success[i] and d['terminated'][i])))
            offset+=n
    assert offset==len(data)==5512
    assert protected=={str(p):sha(p) for p in [pkl,*files]}
    write('episode_audit.json',episodes)
    with (OUT/'episode_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=episodes[0].keys());w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,dict) else v for k,v in row.items()} for row in episodes)
    with (OUT/'near_hole_transitions.jsonl').open('w') as f:
        for r in near_rows:f.write(json.dumps(r)+'\n')
    near=dict(transitions=len(near_rows),episodes_with_coverage=sum(e['near_hole_transitions']>0 for e in episodes),successful_terminal_transitions=sum(r['successful_terminal'] for r in near_rows))
    for k in ('lateral_error','axial_error','orientation_error'):near[k]=stats([r[k] for r in near_rows])
    for k in ('translation_action','rotation_action'):near[k]={str(i):stats([r[k][i] for r in near_rows]) for i in range(3)}
    write('near_hole_summary.json',near);write('protected_hashes.json',protected)
    summary=dict(PEG_F3_V1_DEMO_AUDIT='PASS',PEG_F3_ROTATION_DATA_COVERAGE='INADEQUATE',episodes=20,transitions=5512,successes=sum(e['success'] for e in episodes),
        rotation_nonzero_total=sum(e['rotation_action_nonzero_count'] for e in episodes),orientation_error=stats(orientation),orientation_threshold_steps=sum(e['orientation_threshold_steps'] for e in episodes),
        initial_orientation_errors=stats([e['initial_orientation_error'] for e in episodes]),episode_max_orientation_errors=stats([e['max_orientation_error'] for e in episodes]),near_hole=near,
        interpretation='Zero rotation is valid for successful fixed-scene orientation holding, not inherently wrong. Insufficient evidence for active recovery from rotational deviation seen in learned policy.',
        metric_recompute_max_absolute_error=max(errors),rotation_tracking='UNVERIFIED',training_started=False,demo_recording_started=False,checkpoint_modified=False)
    write('data_review_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
