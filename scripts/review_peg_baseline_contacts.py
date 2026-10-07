"""One authorized fixed reset; no action calls and no Demo output."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
from run_peg_axial_bounded_diagnostic import durable_save,persist_initial_evidence_and_guard
from training.peg_insertion.v2_recorder import ROOT,PKG,bindings,sha,snapshot,validate_snapshot
from training.peg_insertion.axial_telemetry import AxialTelemetry

def main():
    out=ROOT/'results/diagnostics/peg/f3_baseline_contact_review_retry_20260905'
    out.mkdir(parents=True,exist_ok=False)
    b=bindings()
    reference=json.loads((ROOT/'results/demos/peg_v2_pilot/standard_20260905_001/episode.json').read_text())
    assert b==reference['hashes']
    gate=json.loads((ROOT/'results/validation/peg/f3_axial_telemetry_20260905/binding.json').read_text())
    assert all(sha(ROOT/f)==h for f,h in gate['telemetry_files'].items())
    assert sha(gate['installed_solver_file'])==gate['installed_solver_sha256']
    procs=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    for line in procs.splitlines():
        if any(k in line for k in ('scripts/record_peg_v2_demo.py','scripts/run_peg_training.py','scripts/run_peg_axial_bounded_diagnostic.py')):
            raise RuntimeError('conflicting process: '+line)
    durable_save(out/'preflight.json',dict(binding=b,telemetry_binding=gate,processes=procs,runner_sha256=sha(__file__)))
    for line in (ROOT/'configs/peg_insertion/runtime_gpu_formal.env').read_text().splitlines():
        if line.strip():k,v=line.split('=',1);os.environ[k]=v
    sys.path[:0]=[str(PKG/'code/example/async_rlpd_peg_sim'),str(PKG/'code/env')]
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    env=None;observer=None;triggered=False;error=None
    try:
        with portable_peg_runtime(ROOT):env=make_peg_state_env(render_mode='rgb_array',max_episode_steps=500)
        env.reset(seed=0)
        initial=snapshot(env);validate_snapshot(initial)
        np.testing.assert_array_equal(initial['obs'],reference['initial']['obs'])
        observer=AxialTelemetry(env,out/'telemetry')
        try:persist_initial_evidence_and_guard(observer,initial,out)
        except RuntimeError as exc:
            if str(exc)!='dangerous initial penetration' or not (out/'initial_guard_event.json').exists():raise
            triggered=True
        m=env.env.model;d=env.env.data
        geoms=[]
        for i in range(m.ngeom):
            bid=int(m.geom_bodyid[i]);mesh=int(m.geom_dataid[i])
            geoms.append(dict(id=i,name=m.geom(i).name,body_id=bid,body_name=m.body(bid).name,
                type=int(m.geom_type[i]),data_id=mesh,mesh_name=m.mesh(mesh).name if int(m.geom_type[i])==7 and mesh>=0 else None,
                contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),
                size=m.geom_size[i].tolist(),position=d.geom_xpos[i].tolist()))
        durable_save(out/'geom_inventory.json',geoms)
        assert observer.count==0
    except BaseException as exc:error=repr(exc);raise
    finally:
        if observer:observer.close()
        if env:env.close()
        durable_save(out/'execution.json',dict(error=error,initial_guard_triggered=triggered,
            simulator_actions_executed=0,observer_count=observer.count if observer else None,
            demo_recording_started=False,training_started=False,training_data_eligible=False,
            production_hashes_unchanged=bindings()==b))
    print(json.dumps(json.loads((out/'initial_snapshot.json').read_text())['contacts'],indent=2))
    print((out/'execution.json').read_text())
if __name__=='__main__':main()
