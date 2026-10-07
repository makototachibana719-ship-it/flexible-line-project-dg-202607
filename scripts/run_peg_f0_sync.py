"""Single B arm; historical F0-Observe is A. Never promotes formal Gate."""
import os
import verify_peg_f0 as f0

if __name__=='__main__':
    os.environ['JAX_LOG_COMPILES']='1'
    f0.OUT=f0.ROOT/'results/validation/peg/f0_sync_20260904'
    raise SystemExit(f0.run_once(write_gate=False,extra_args=('--observe-update1','--per-leaf-sync')))
