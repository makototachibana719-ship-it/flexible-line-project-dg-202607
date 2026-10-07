# Object GPU Environment

Reference: Python 3.10.20, JAX/JAXlib 0.4.35, Flax 0.8.5, Gym 0.26.2, Gymnasium 1.3.0, MuJoCo 2.3.7, Optax 0.2.3, Orbax 0.6.4, TensorStore 0.1.51. Install the JAX/JAXlib CUDA wheel appropriate to the WSL2 driver separately. TensorStore 0.1.51 is selected because the project-side legacy-Flax new-checkpoint path is tested independently of legacy `checkpoint_4500`; 0.1.78 is not required for fresh training.
