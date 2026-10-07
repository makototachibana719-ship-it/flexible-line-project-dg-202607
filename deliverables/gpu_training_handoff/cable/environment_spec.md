# Cable GPU Environment Specification

`GPU_ENV_SPEC = PREPARED`  
`GPU_ENV_RUNTIME = UNVERIFIED`

## Validated reference environment

Python 3.10.20; JAX/JAXlib 0.4.35; Flax 0.8.5; Gym 0.26.2; Gymnasium 1.3.0; MuJoCo 2.3.7; W&B 0.28.0; Orbax 0.6.4; TensorStore 0.1.51. This is CPU/reference evidence, not proof of GPU compatibility. The delivered `metadata/pip_freeze.txt` contains later TensorStore/W&B versions and must not silently override these project-tested checkpoint constraints.

## Source/install contract

- Project root is supplied as `PROJECT_ROOT`; no `/home/ubuntu` path is allowed.
- Editable SERL launcher source: `$PROJECT_ROOT/external/serl/serl_launcher`.
- Editable Cable environment source: `$PROJECT_ROOT/external/packages/cable_routing_v2/CableRoute_最终交接包_V2_20260820_221714/franka_sim`.
- Training entry remains `$PROJECT_ROOT/training/cable/run_cable_training.py`.
- MuJoCo uses EGL for headless execution.
- `configs/cable/environment_gpu.yaml` is a base specification, not a GPU-validated lock file.

## GPU-specific JAX rule

Do not copy the CPU jaxlib wheel mechanically. Install the JAX 0.4.35 CUDA build compatible with the target driver/CUDA using the official JAX installation matrix. Record the exact command and resolved wheel.

Run before smoke:

```bash
nvidia-smi
python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

At least one CUDA/GPU device must appear and `jax.default_backend()` must not equal `cpu`. Otherwise stop before formal training.
