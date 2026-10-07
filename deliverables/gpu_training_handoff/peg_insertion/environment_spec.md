# Peg GPU Environment Specification

Reference Python is 3.10.20. The locally validated `serl_mujoco` versions are JAX/JAXlib 0.4.35, Flax 0.8.5, Optax 0.2.3, Gym 0.26.2, Gymnasium 1.3.0, MuJoCo 3.10.0, W&B 0.28.0, Orbax Checkpoint 0.6.4, and TensorStore 0.1.78. Delivered package evidence differs for Gym 0.23.1, Gymnasium 1.1.1, MuJoCo 3.1.5, and W&B 0.28.1; the locally validated lifecycle is authoritative for the reference handoff, while deviations must be recorded.

Create the independent environment with `conda env create -f configs/peg_insertion/environment_gpu.yaml` and activate `serl_peg_gpu`. It must not reuse the Cable or Object environment.

Do not copy the CPU JAXlib wheel mechanically. Select the official GPU build compatible with the target WSL2 NVIDIA driver/CUDA. Before any smoke, both `nvidia-smi` and `jax.default_backend() == "gpu"` must pass.

Install/discover `mujoco_env` so Python can import it. The project runtime adapter discovers its actual module location and creates a runtime-local asset symlink; no username-specific path is permitted. Use the full SERL launcher on `PYTHONPATH`; do not use the package's partial `modified_serl` directory as a replacement.

Checkpointing uses project-scoped legacy Flax files plus online replay and metadata sidecars. Delivered OCDBT checkpoints are legacy artifacts and are never loaded.
