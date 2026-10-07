#!/usr/bin/env bash
# Sourced by all production launchers; fixed project interpreter and paths.
PY="$HOME/miniconda3/envs/serl_peg_gpu/bin/python"
if [[ ! -x "$PY" ]]; then echo 'FORMAL_LAUNCH_BLOCKED: locked Python missing' >&2; return 2; fi
export PYTHONPATH="$ROOT/external/serl/serl_launcher:$ROOT/external/serl/franka_sim:$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export MUJOCO_GL=egl
unset JAX_DISABLE_JIT JAX_PLATFORMS JAX_PLATFORM_NAME CUDA_VISIBLE_DEVICES
