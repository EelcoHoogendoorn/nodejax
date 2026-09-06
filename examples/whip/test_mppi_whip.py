"""MPPI on the whip: the planner rolls the plant's own model out from its state."""

import jax
import jax.numpy as jnp

from examples.rl import control
from examples.whip.learn import evaluation_starts, whip_critic, whip_evaluation
from examples.whip.mppi_whip import learned_controller, train_whip_mppi
from examples.whip.whip import whip_plant


def test_mppi_trains_a_critic_and_plans_on_the_whip() -> None:
    plant = whip_plant().parameterize()
    result = train_whip_mppi(
        plant, whip_critic(plant), n_iterations=1, n_worlds=2, n_candidates=4, n_refinements=1,
        n_steps_per_plan=4, n_steps_per_iteration=5, n_critic_updates=1,
    )
    assert result.history.mean_cost.shape == (1,)
    assert jnp.all(jnp.isfinite(result.history.mean_cost))

    controller = learned_controller(result.critic, plant, n_candidates=4, n_refinements=1, n_steps_per_plan=4)
    evaluation = whip_evaluation(
        controller, plant, evaluation_starts(plant), jax.random.PRNGKey(3), steps=3, step=control.PlannedStep)
    assert evaluation.action.shape == (9, 3, 2)  # nine worlds, three steps, two joints
    assert jnp.all(jnp.isfinite(evaluation.reward))
