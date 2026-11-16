"""Verifies single-program SHAC training and structural inspection on the whip plant."""

import jax.numpy as jnp

from nodejax import nn
from examples.whip.learn import HIDDEN, MEMORY, WhipFeatures, WhipMean, whip_critic
from examples.whip.shac_whip import train_whip_shac, whip_training_program
from examples.whip.whip import whip_plant


def test_shac_trains_the_whip_as_one_program() -> None:
    result = train_whip_shac(
        whip_plant().parameterize(), WhipFeatures(tip_sensor=False),
        n_episodes=1, n_worlds=2, n_chunks_per_episode=1, n_steps_per_chunk=2, n_critic_updates=1,
    )
    cost = result.history.mean_cost
    assert cost.shape == (1,)
    assert jnp.all(jnp.isfinite(cost))
    assert result.policy.cyclic


def test_program_listing_prints_the_whole_tree() -> None:
    """Verifies that `tree_view` and `describe` report the whip training `Node` hierarchy, statics, and plant parameter shapes."""
    plant = whip_plant().parameterize()
    program = whip_training_program(
        WhipMean(WhipFeatures(tip_sensor=False), nn.GRU(MEMORY), HIDDEN), whip_critic(plant), plant, 1,
        n_worlds=2, n_chunks_per_episode=1, n_steps_per_chunk=2, n_critic_updates=1,
    )
    tree = program.tree_view()
    listing = program.describe()
    plant_listing = plant.describe()
    print(tree)
    print(listing)

    assert 'plant: whip_step' in tree
    assert "trainable='policy'" in listing
    assert 'rope.solve.bending.index = int32(8, 3)' in plant_listing


def test_shac_trains_on_the_ideal_actuator() -> None:
    result = train_whip_shac(
        whip_plant(ideal_actuator=True).parameterize(), WhipFeatures(tip_sensor=False),
        n_episodes=1, n_worlds=2, n_chunks_per_episode=1, n_steps_per_chunk=2, n_critic_updates=1,
    )
    assert jnp.all(jnp.isfinite(result.history.mean_cost))
