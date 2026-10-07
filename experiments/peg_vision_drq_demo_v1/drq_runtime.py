"""Thin runtime adapter around the unchanged Object DrQ implementation.

The shared ``train_utils`` module imports TensorFlow and W&B for optional video
and logging helpers.  On this WSL host those unused imports conflict with the
OSMesa renderer.  This module supplies only the three exact utilities DrQ uses,
then imports the original ``DrQAgent`` unchanged.
"""

from __future__ import annotations

from collections import defaultdict
import pickle
from pathlib import Path
import sys
import types
from typing import Any

import jax
from jax import nn
import jax.numpy as jnp
from flax.core import frozen_dict


OBJECT_RESNET10 = Path(
    "/home/makoto/flexible_line_project/external/packages/object_grasping/"
    "object_grasp_package/code/example/resnet10_params.pkl"
)


def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)
    if not isinstance(offline_batch, dict):
        offline_batch = offline_batch.unfreeze()
    if not isinstance(online_batch, dict):
        online_batch = online_batch.unfreeze()
    for key, value in offline_batch.items():
        if type(value) is dict:
            batch[key] = concat_batches(offline_batch[key], online_batch[key], axis=axis)
        else:
            batch[key] = jnp.concatenate((offline_batch[key], online_batch[key]), axis=axis)
    return frozen_dict.freeze(batch)


def unpack(batch):
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]
            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )
    return batch


def load_resnet10_params(agent, image_keys=("image",), public=True):
    del public
    with OBJECT_RESNET10.open("rb") as stream:
        encoder_params = pickle.load(stream)
    new_params = agent.state.params
    for image_key in image_keys:
        new_encoder_params = new_params["modules_actor"]["encoder"][f"encoder_{image_key}"]
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for key in new_encoder_params:
            if key in encoder_params:
                new_encoder_params[key] = encoder_params[key]
    return agent.replace(state=agent.state.replace(params=new_params))


def _install_minimal_serl_modules() -> None:
    # The upstream typing module imports the full TensorFlow runtime only to
    # include ``tf.Tensor`` in a type alias.  Runtime DrQ code needs no
    # TensorFlow symbol, so preserve its public names without loading TF.
    typing_name = "serl_launcher.common.typing"
    if typing_name not in sys.modules:
        typing_module = types.ModuleType(typing_name)
        typing_module.PRNGKey = Any
        typing_module.Params = frozen_dict.FrozenDict
        typing_module.Data = Any
        typing_module.Batch = Any
        sys.modules[typing_name] = typing_module

    train_name = "serl_launcher.utils.train_utils"
    if train_name not in sys.modules:
        train_module = types.ModuleType(train_name)
        train_module._unpack = unpack
        train_module.concat_batches = concat_batches
        train_module.load_resnet10_params = load_resnet10_params
        sys.modules[train_name] = train_module


_install_minimal_serl_modules()

from serl_launcher.agents.continuous.drq import DrQAgent  # noqa: E402


def make_object_drq_agent(seed, sample_obs, sample_action, image_keys=("front", "wrist")):
    """Exact ``make_drq_agent`` recipe from the Object training entry point."""
    return DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type="resnet-pretrained",
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.96,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )
