"""Fixed-reset screening supervisor, never final evaluation or training."""
import json, subprocess, sys, hashlib, math, ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/validation/peg/f3_stage2_20260904'
STEPS=(5000,6000,7000,8000,9000)
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def mean(rows,key):return sum(r[key] for r in rows)/len(rows)
def main():
    OUT.mkdir(exist_ok=False)
    source=(ROOT/'scripts/peg_f3_stage1.py').read_text()
    worker=(ROOT/'scripts/peg_f3_stage2_worker.py').read_text()
    # Prove worker differs only in checkpoint/output parameterization.
    expected=source.replace("OUT=ROOT/'results/validation/peg/f3_stage1_20260904'", "STEP=int(sys.argv[1])\ncheck_steps=(5000,6000,7000,8000,9000)\nif STEP not in check_steps:raise ValueError('not screening allowlist')\nOUT=ROOT/f'results/validation/peg/f3_stage2_20260904/checkpoint_{STEP}'")
    expected=expected.replace('checkpoint_9000','checkpoint_{STEP}').replace("CP=ROOT/'results/checkpoints/peg_rlpd_baseline_seed0/checkpoint_{STEP}'", "CP=ROOT/f'results/checkpoints/peg_rlpd_baseline_seed0/checkpoint_{STEP}'").replace("(ROOT/'results/checkpoints/peg_rlpd_baseline_seed0/checkpoint_{STEP}')","(ROOT/f'results/checkpoints/peg_rlpd_baseline_seed0/checkpoint_{STEP}')").replace("x['step']==9000","x['step']==STEP").replace('CP.parent,agent.state,9000','CP.parent,agent.state,STEP')
    assert ast.dump(ast.parse(worker))==ast.dump(ast.parse(expected)), 'worker contract drift'
    protocol=dict(checkpoints=STEPS,episodes_per_checkpoint=10,seeds=list(range(10)),reset='fixed default; no random option',argmax=True,max_steps=500,
        metrics='identical to Stage1',successful_metrics_when_no_success=None,
        ranking=['success_rate descending','successful steps ascending (positive success only)','successful cycle_time ascending (positive success only)',
                 'mean_min_position_err ascending (precision proxy, not final insertion precision)','mean_approach_reduction descending (zero-success branch only)',
                 'mean_min_lateral_err ascending (zero-success branch only)','mean_min_axial_err ascending (zero-success branch only)',
                 'mean_max_force ascending','mean_action_saturation ascending','mean_return descending','step ascending (exact tie)'],
        zero_success_improvement_rule='B if any later checkpoint beats5000 mean_min_position_err by at least0.002m AND10%; otherwise C. Descriptive heuristic, not statistical significance.',
        stage1_sha256=hashlib.sha256(source.encode()).hexdigest(),worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),contract_diff_verified=True)
    write('protocol.json',protocol)
    summaries=[];executions=[];errors=[]
    for step in STEPS:
        print('START checkpoint',step,flush=True)
        with (OUT/f'checkpoint_{step}.log').open('x') as log:
            try:
                run=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/peg_f3_stage2_worker.py'),str(step)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                code=run.returncode
            except subprocess.TimeoutExpired:code=124
        executions.append(dict(checkpoint=step,exit_code=code));write('execution.json',executions)
        if code:errors.append(f'checkpoint{step} worker exit{code}');break
        result=json.loads((OUT/f'checkpoint_{step}/result.json').read_text())
        if result['PEG_F3_CHECKPOINT_9000_SMOKE']!='PASS':errors.append('worker audit failed');break
        rows=[json.loads(x) for x in (OUT/f'checkpoint_{step}/episodes.jsonl').read_text().splitlines()]
        assert len(rows)==10 and [r['seed'] for r in rows]==list(range(10))
        successes=[r for r in rows if r['success']]
        summary=dict(checkpoint=step,episodes=10,successes=len(successes),success_rate=len(successes)/10,
            mean_successful_steps=mean(successes,'steps') if successes else None,
            mean_successful_cycle_time=mean(successes,'cycle_time') if successes else None)
        for key in ('steps','cycle_time','return','first_position_err','min_position_err','approach_reduction','min_lateral_err','min_axial_err','max_force','action_saturation'):
            summary['mean_'+key]=mean(rows,key)
        summaries.append(summary);print(json.dumps(summary),flush=True)
    any_success=any(s['successes'] for s in summaries)
    def rank(s):
        if s['successes']:
            return (-s['success_rate'],s['mean_successful_steps'],s['mean_successful_cycle_time'],s['mean_min_position_err'],s['mean_max_force'],s['mean_action_saturation'],-s['mean_return'],s['checkpoint'])
        return (0,s['mean_min_position_err'],-s['mean_approach_reduction'],s['mean_min_lateral_err'],s['mean_min_axial_err'],s['mean_max_force'],s['mean_action_saturation'],-s['mean_return'],s['checkpoint'])
    ranking=sorted(summaries,key=rank)
    complete=len(summaries)==5 and not errors
    improvement=complete and any(s['mean_min_position_err']<=summaries[0]['mean_min_position_err']*.9 and summaries[0]['mean_min_position_err']-s['mean_min_position_err']>=.002 for s in summaries[1:])
    category='A' if any_success else ('B' if improvement else 'C')
    final=dict(PEG_F3_SCREENING='PASS' if complete else 'FAIL',category=category if complete else 'UNRESOLVED',
        PEG_F3_FINAL_COMPARISON='READY' if complete and any_success else 'BLOCKED',
        PEG_F3_POLICY_QUALITY_REVIEW=('RECOMMENDED' if improvement else 'REQUIRED') if complete and not any_success else 'NOT_REQUIRED_BY_THIS_RULE',
        checkpoints=summaries,ranking=[s['checkpoint'] for s in ranking],Top2=[s['checkpoint'] for s in ranking[:2]] if complete and any_success else None,
        diagnostic_top2=[s['checkpoint'] for s in ranking[:2]],errors=errors,training_started=False,checkpoint_resumed=False,cloud_modified=False,final_evaluation_started=False,
        limitation='Repeated fixed deterministic scene, not independent scenarios or generalization success rate. Minima can occur at different timesteps. Successful metrics undefined=null for zero success.')
    write('screening_summary.json',final);print(json.dumps(final,indent=2),flush=True)
    return 0 if complete else 1
if __name__=='__main__':raise SystemExit(main())
