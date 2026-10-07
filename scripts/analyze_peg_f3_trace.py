"""Offline analysis/figures only; never imports the evaluator or JAX."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/validation/peg/f3_diagnostic_20260904'
allrows={};summary={}
for checkpoint in (5000,9000):
    rows=[json.loads(x) for x in (OUT/f'trajectory_checkpoint{checkpoint}.jsonl').read_text().splitlines()]
    assert len(rows)==500 and [r['step'] for r in rows]==list(range(1,501))
    assert all(np.isclose(sum(r['reward_components'].values()),r['reward_total']) and r['success']==(r['simultaneous_threshold_ratio']<=1) for r in rows)
    outresult=json.loads((OUT/f'checkpoint_{checkpoint}/result.json').read_text())
    assert outresult['PEG_F3_CHECKPOINT_9000_SMOKE']=='PASS' and outresult['episodes']==1 and not outresult['checkpoint_modified']
    allrows[checkpoint]=rows
    with (OUT/f'trajectory_checkpoint{checkpoint}.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    keys={}
    for name,field in [('t_min_position','position_error'),('t_min_lateral','lateral_error'),('t_min_axial','axial_error'),('t_min_orientation','orientation_error'),('t_best_simultaneous_threshold','simultaneous_threshold_ratio')]:keys[name]=min(rows,key=lambda r:r[field])
    keys['t_max_force']=max(rows,key=lambda r:r['force_norm']);keys['t_final']=rows[-1]
    best=keys['t_best_simultaneous_threshold'];sat=np.asarray([r['action_saturation_flags'] for r in rows]);actions=np.asarray([r['action'] for r in rows])
    modes=[]
    if keys['t_min_position']['position_error']>.01:modes.append('A')
    if best['lateral_ratio']>1:modes.append('B')
    if best['orientation_ratio']>1:modes.append('C')
    if sat.mean()>=.1:modes.append('E')
    value=dict(failure_mode='MIXED ('+'+'.join(modes)+')' if len(modes)>1 else (modes[0] if modes else 'UNRESOLVED'),best_simultaneous_ratio=best['simultaneous_threshold_ratio'],
        min_position=keys['t_min_position']['position_error'],max_force=keys['t_max_force']['force_norm'],critical_timesteps=keys,
        fraction_saturated=float(sat.mean()),saturation_by_action_dimension=sat.mean(axis=0).tolist(),
        action_sign_changes=((actions[1:]*actions[:-1])<0).sum(axis=0).tolist(),
        threshold_pass_steps={k:sum(r[k]<=1 for r in rows) for k in ('lateral_ratio','axial_ratio','orientation_ratio','simultaneous_threshold_ratio')},
        reward_component_totals={k:sum(r['reward_components'][k] for r in rows) for k in rows[0]['reward_components']},
        force_rejection='UNRESOLVED: force alone does not demonstrate rejection; episode truncates, not force-triggered termination',
        first_contact_step=next((r['step'] for r in rows if r['contact_count']>0),None),
        policy_state_and_bundle_checks='PASS',classification_scope='descriptive heuristic; A means not within1cm, E denotes saturation not proven oscillation')
    summary[str(checkpoint)]=value

def svg(name,series,log=False):
    import html
    vals=[float(v) for _,data in series for v in data]
    def trans(v):return np.log10(max(v,1e-6)) if log else v
    lo=min(trans(v) for v in vals);hi=max(trans(v) for v in vals)
    if log:lo=min(lo,0)
    pad=max((hi-lo)*.08,.01);lo-=pad;hi+=pad
    def y(v):return 340-(trans(float(v))-lo)/(hi-lo)*290
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="450"><rect width="1100" height="450" fill="white"/>',f'<text x="80" y="24" font-size="17">{html.escape(name)}</text>']
    for j in range(6):
        v=lo+(hi-lo)*j/5;v=10**v if log else v
        parts.append(f'<path d="M80,{y(v)}H1050" stroke="#ddd"/><text x="5" y="{y(v)}" font-size="12">{v:.4g}</text>')
    for st in (1,100,200,300,400,500):parts.append(f'<text x="{80+(st-1)/499*970}" y="362" font-size="12">{st}</text>')
    colors=['#1565c0','#d84315','#388e3c','#7b1fa2','#00838f','#ad1457','#546e7a']
    for j,(label,data) in enumerate(series):
        points=' '.join(f'{80+i/499*970:.2f},{y(v):.2f}' for i,v in enumerate(data));color=colors[j%7]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/><text x="{80+j%3*330}" y="{395+j//3*20}" fill="{color}" font-size="12">{html.escape(label)}</text>')
    if log:parts.append(f'<path d="M80,{y(1)}H1050" stroke="black" stroke-dasharray="5,4"/>')
    parts.append('</svg>');(OUT/(name+'.svg')).write_text(''.join(parts),encoding='utf-8')
for field,name in [('position_error','position_error_vs_step'),('force_norm','force_vs_step'),('position_error','checkpoint5000_vs9000_position'),('simultaneous_threshold_ratio','checkpoint5000_vs9000_simultaneous_threshold')]:
    svg(name,[(str(cp),[r[field] for r in rows]) for cp,rows in allrows.items()],'ratio' in field)
for cp,rows in allrows.items():
    svg('threshold_ratios_vs_step_'+str(cp),[(k,[r[k] for r in rows]) for k in ('lateral_ratio','axial_ratio','orientation_ratio','simultaneous_threshold_ratio')],True)
    svg('actions_vs_step_'+str(cp),[(f'a{i}',[r['action'][i] for r in rows]) for i in range(7)])
result=dict(PEG_F3_DIAGNOSTIC='PASS',checkpoints=summary,training_started=False,checkpoint_modified=False,checkpoint_resumed=False,screening_started=False,final_evaluation_started=False)
with (OUT/'diagnostic_summary.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
for checkpoint,s in summary.items():
    print(checkpoint,json.dumps({k:v for k,v in s.items() if k!='critical_timesteps'}))
    for key,row in s['critical_timesteps'].items():print(key,json.dumps({k:row[k] for k in ('step','position_error','lateral_ratio','axial_ratio','orientation_ratio','force_norm','contact_count','action')}))
