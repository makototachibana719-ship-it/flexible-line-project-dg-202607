from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

register(
    id="PandaCableRoute-v0",
    entry_point="franka_sim.envs:PandaCableRouteGymEnv",
    max_episode_steps=500,
)

register(
    id="PandaCableRouteVision-v0",
    entry_point="franka_sim.envs:PandaCableRouteGymEnv",
    max_episode_steps=500,
    kwargs={"image_obs": True},
)
