"""PPO on the whip: one program from one key, no gradient through the plant."""

import jax
import jax.numpy as jnp

from examples.whip.learn import WhipFeatures, evaluation_starts, summarize, whip_evaluation
from examples.whip.whip import whip_plant
from examples.whip.ppo_whip import mean_policy, train_whip_ppo


def test_ppo_trains_the_whip_as_one_program() -> None:
    plant = whip_plant().parameterize()
    result = train_whip_ppo(
        plant, WhipFeatures(tip_sensor=False), iterations=2, n_worlds=4, n_steps_per_chunk=8,
        n_minibatches_per_epoch=2, n_chunks_per_minibatch=1,
    )
    assert result.history.mean_cost.shape == (2,)
    assert jnp.all(jnp.isfinite(result.history.mean_cost))
    evaluation = whip_evaluation(
        mean_policy(result.policy), plant, evaluation_starts(plant), jax.random.PRNGKey(3), steps=5)
    assert evaluation.reward.shape == (9,)  # three starts by three targets
    assert jnp.all(jnp.isfinite(evaluation.reward))
    summarize(evaluation, 'PPO, two iterations')


def test_ppo_trains_with_the_tip_sensed() -> None:
    plant = whip_plant(tip_sensor=True).parameterize()
    result = train_whip_ppo(
        plant, WhipFeatures(tip_sensor=True), iterations=1, n_worlds=4, n_steps_per_chunk=8,
        n_minibatches_per_epoch=2, n_chunks_per_minibatch=1,
    )
    evaluation = whip_evaluation(
        mean_policy(result.policy), plant, evaluation_starts(plant), jax.random.PRNGKey(3), steps=5)
    assert jnp.all(jnp.isfinite(evaluation.reward))
