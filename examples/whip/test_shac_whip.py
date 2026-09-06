"""SHAC on the whip: one program from one key, through motor and rope."""

import jax
import jax.numpy as jnp

from nodejax import nn
from examples.whip.learn import HIDDEN, MEMORY, WhipFeatures, WhipMean, whip_critic
from examples.whip.shac_whip import N_EPISODES, train_whip_shac, whip_training_program
from examples.whip.whip import whip_plant


def test_shac_trains_the_whip_as_one_program() -> None:
    result = train_whip_shac(
        whip_plant().parameterize(), WhipFeatures(tip_sensor=False),
        n_episodes=2, n_worlds=4, n_chunks_per_episode=2, n_steps_per_chunk=10,
    )
    cost = result.history.mean_cost
    assert cost.shape == (2,)  # one entry per episode
    assert jnp.all(jnp.isfinite(cost))
    assert result.policy.cyclic


def test_program_listing_prints_the_whole_tree() -> None:
    """Run with ``-s`` to read the program: the node tree with each node's
    statics, then the parameter tree with shapes."""
    plant = whip_plant().parameterize()
    program = whip_training_program(
        WhipMean(WhipFeatures(tip_sensor=False), nn.GRU(MEMORY), HIDDEN), whip_critic(plant), plant, N_EPISODES)
    tree = program.tree_view()
    listing = program.parameterize(rng=jax.random.PRNGKey(1)).describe()
    print(tree)
    print(listing)

    assert 'plant: whip_step' in tree
    assert "trainable='policy'" in listing
    assert 'rollout.plant.rope.solve.bending.index = int32(8, 3)' in listing


def test_shac_trains_on_the_ideal_actuator() -> None:
    result = train_whip_shac(
        whip_plant(ideal_actuator=True).parameterize(), WhipFeatures(tip_sensor=False),
        n_episodes=1, n_worlds=4, n_chunks_per_episode=2, n_steps_per_chunk=10,
    )
    assert jnp.all(jnp.isfinite(result.history.mean_cost))
