"""One fresh instrumented production-path attempt; never promotes a Gate."""
import os
import verify_peg_f0 as f0

if __name__=='__main__':
    os.environ['JAX_LOG_COMPILES']='1'
    f0.OUT=f0.ROOT/'results/validation/peg/f0_observe_20260904'
    raise SystemExit(f0.run_once(write_gate=False,extra_args=('--observe-update1',)))
