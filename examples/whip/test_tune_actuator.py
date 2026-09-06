"""The actuator tuner: a genome read off the plant, written back, scored, and evolved."""

import jax.numpy as jnp

from nodejax import Struct
from examples.whip.tune_actuator import GENOME, plant_genome, scenario_cost, stroke_commands, tune, tuned_plant
from examples.whip.whip import CURRENT_KI, OBSERVER_TAU_VEL, whip_plant


def test_the_genome_starts_from_the_plant_and_writes_back_into_it() -> None:
    base = whip_plant().parameterize()
    initial = plant_genome(base)
    assert jnp.allclose(initial.tau_vel, OBSERVER_TAU_VEL) and jnp.allclose(initial.current_ki, CURRENT_KI)
    genome = jnp.log(jnp.asarray(3.0))
    tuned = tuned_plant(base, Struct(**{name: genome for name in GENOME}))
    assert jnp.allclose(tuned.param.actuator.mechanical_est.observer.tau_vel, 3.0)
    assert jnp.allclose(tuned.param.actuator.current_ctrl.controller.ki, 3.0)


def test_evolution_runs_a_generation_and_reports_finite_costs() -> None:
    base = whip_plant().parameterize()
    result = tune(base, n_generations=2, population=4)
    assert result.history.shape == (2,)
    assert jnp.all(jnp.isfinite(result.history))
    assert jnp.all(jnp.isfinite(scenario_cost(base, stroke_commands()).total))
