#!/usr/bin/env python3

import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer

import franka_sim

class CompatFlattenObservation(gym.Wrapper):
    """
    扁平化 Dict observation，并统一为 Gymnasium 风格接口。

    reset() -> observation, info
    step()  -> observation, reward, terminated, truncated, info
    """

    def __init__(self, env):
        super().__init__(env)

        self._original_observation_space = env.observation_space

        flat_space = gym.spaces.flatten_space(
            self._original_observation_space
        )

        self.observation_space = gym.spaces.Box(
            low=np.asarray(flat_space.low, dtype=np.float32),
            high=np.asarray(flat_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def _flatten_observation(self, observation):
        flattened = gym.spaces.flatten(
            self._original_observation_space,
            observation,
        )

        return np.asarray(flattened, dtype=np.float32)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            observation, info = result
        else:
            observation = result
            info = {}

        if info is None:
            info = {}

        observation = self._flatten_observation(observation)

        return observation, info

    def step(self, action):
        result = self.env.step(action)

        if not isinstance(result, tuple):
            raise TypeError(
                "env.step() 必须返回 tuple，"
                f"当前类型为 {type(result)}"
            )

        if len(result) == 5:
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = result

        elif len(result) == 4:
            observation, reward, done, info = result

            if info is None:
                info = {}

            truncated = bool(
                info.get("TimeLimit.truncated", False)
            )
            terminated = bool(done) and not truncated

        else:
            raise RuntimeError(
                "无法识别 env.step() 返回格式，"
                f"返回值数量为 {len(result)}"
            )

        if info is None:
            info = {}

        observation = self._flatten_observation(observation)

        return (
            observation,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )


from peg_env_adapter import make_peg_state_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")
flags.DEFINE_integer("utd_ratio",8,"Update-to-data ratio.")
flags.DEFINE_integer("max_steps", 10000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: SACAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # SERL_PEG_EVAL_ENV_PATCH_V2
    eval_env = make_peg_state_env(
        render_mode="rgb_array",
        max_episode_steps=500,
    )

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            # SERL_TERMINATION_PATCH_V1
            next_obs, reward, terminated, truncated, info = env.step(actions)
            terminated = bool(terminated)
            truncated = bool(truncated)
            episode_done = terminated or truncated
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)

            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    # 仅真正终止时停止 bootstrap；
                    # 时间上限 truncated 仍允许 bootstrap。
                    masks=np.float32(
                        0.0 if terminated else 1.0
                    ),
                    dones=episode_done,
                )
            )

            obs = next_obs
            if episode_done:
                running_return = 0.0
                obs, _ = env.reset()

        if FLAGS.render:
            env.render()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: SACAgent, replay_buffer, replay_iterator):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # Train the networks
        with timer.context("sample_replay_buffer"):
            batch = next(replay_iterator)

        with timer.context("train"):
            agent, update_info = agent.update_high_utd(batch, utd_ratio=FLAGS.utd_ratio)
            agent = jax.block_until_ready(agent)

            # publish the updated network
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    # SERL_PEG_ENV_PATCH_V2
    env = make_peg_state_env(
        render_mode="human" if FLAGS.render else "rgb_array",
        max_episode_steps=500,
    )

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="replay_buffer",
            preload_rlds_path=FLAGS.preload_rlds_path,
        )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator=replay_iterator,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
