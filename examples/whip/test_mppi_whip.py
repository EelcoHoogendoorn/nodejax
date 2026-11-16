"""Verifies MPPI critic training and receding-horizon planning on the whip plant."""

import jax
import jax.numpy as jnp

from examples.rl import control
from examples.whip.learn import evaluation_starts, whip_critic, whip_evaluation
from examples.whip.mppi_whip import learned_controller, train_whip_mppi
from examples.whip.whip import whip_plant


def test_mppi_trains_a_critic_and_plans_on_the_whip() -> None:
    plant = whip_plant(ideal_actuator=True).parameterize()
    result = train_whip_mppi(
        plant, whip_critic(plant, hidden=16), n_iterations=1, n_worlds=2, n_candidates=2,
        n_refinements=1, n_steps_per_plan=2, n_steps_per_iteration=2, n_critic_updates=1,
    )
    assert result.history.mean_cost.shape == (1,)
    assert jnp.all(jnp.isfinite(result.history.mean_cost))

    controller = learned_controller(
        result.critic, plant, n_candidates=2, n_refinements=1, n_steps_per_plan=2)
    starts = jax.tree.map(lambda value: value[:2], evaluation_starts(plant))
    evaluation = whip_evaluation(
        controller, plant, starts, jax.random.PRNGKey(3), steps=2, step=control.PlannedStep)
    # Evaluates 2 initial states over 2 control steps across both arm joints.
    assert evaluation.action.shape == (2, 2, 2)
    assert jnp.all(jnp.isfinite(evaluation.reward))
