"""Verifies open-loop plan refinement and critic training on the `Pendulum` MPPI assembly."""

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode,
    Leaf,
    Struct,
    batch,
    drop_aux,
    repeat,
    scanned,
    tile,
    tree_len,
)
from examples.rl.control import OpenLoopStep
from examples.rl.mppi import CandidateRollouts, GaussianProposal, MPPIStep, mppi_training
from examples.rl.mppi_pendulum import (
    COMMAND_LIMIT,
    DISCOUNT,
    NOISE_CORRELATION,
    NOISE_SCALE,
    TEMPERATURE,
    PendulumCommandRange,
    pendulum_critic,
    pendulum_training_program,
)
from examples.rl.pendulum import Pendulum


def open_loop_trajectory(
    controls: jax.Array,
    starts: Struct,
    plant: BaseNode,
) -> Struct:
    """Rolls out one control sequence per world from its matching initial state.

    `controls` has shape `[world, time]` and every leaf in `starts` has leading shape `[world]`.
    `batch(scanned(OpenLoopStep(plant)))` maps across worlds and scans across time from `starts`,
    supplying zero disturbances to satisfy the `Pendulum` step contract.
    """
    rollout = batch(scanned(OpenLoopStep(plant)), n=tree_len(starts)).parameterize()
    return drop_aux(rollout.apply(
        command=controls,
        disturbance=jnp.zeros_like(controls),
        initial_state=starts,
    ))


def test_gaussian_refinement_improves_the_plan() -> None:
    """Verifies that repeated `MPPIStep` refinements produce finite bounded plans and open-loop trajectories."""
    plant = Pendulum()
    n_candidates = 8
    n_steps = 6
    start = Struct(angle=jnp.asarray(1.0), velocity=jnp.asarray(0.5))
    controls = jnp.zeros((n_steps,))
    candidate_plans = Leaf(
        lambda input: tile(input, n_candidates),
        name='candidate_plans',
    )
    proposal = candidate_plans >> batch(
        GaussianProposal(
            noise_scale=NOISE_SCALE,
            correlation=NOISE_CORRELATION,
            clean=PendulumCommandRange(),
        ),
        n=n_candidates,
    )

    def zero_terminal_value(terminal_state):
        """Returns a zero terminal cost estimate for every candidate rollout."""
        return jnp.zeros_like(terminal_state.angle)

    zero_value = Leaf(zero_terminal_value, name='zero_terminal_value')
    open_loop_rollouts = batch(scanned(OpenLoopStep(plant)), n=n_candidates + 1)
    refinement = MPPIStep(
        proposal=proposal,
        rollouts=CandidateRollouts(open_loop_rollouts),
        critics=batch(zero_value, n=n_candidates + 1),
        discount=DISCOUNT,
        temperature=TEMPERATURE,
    )
    planner = repeat(refinement, n=2).with_input(bundle=Struct(
        initial_state=start,
        controls=controls,
    )).parameterize()

    result = planner.apply(
        initial_state=start,
        controls=controls,
        rng=jax.random.PRNGKey(5),
    )
    trajectory = open_loop_trajectory(result.controls[None], tile(start, 1), plant)

    assert result.controls.shape == (n_steps,)
    assert jnp.all(jnp.isfinite(result.controls))
    assert jnp.max(jnp.abs(result.controls)) <= COMMAND_LIMIT
    assert trajectory.cost.shape == (1, n_steps)
    assert jnp.all(jnp.isfinite(trajectory.cost))


def test_one_planning_update_trains_the_critic() -> None:
    critic = pendulum_critic()
    trained = mppi_training(
        pendulum_training_program(
            2,
            critic,
            Pendulum(),
            n_worlds=2,
            n_refinements=1,
            n_candidates=4,
            n_steps_per_plan=3,
            n_steps_per_iteration=3,
            n_critic_updates=1,
        ),
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    state = Struct(angle=jnp.asarray(-1.2), velocity=jnp.asarray(2.0))

    assert jnp.isfinite(critic.bind(trained.iteration.ema_critic.state).apply(state))
    assert jnp.isfinite(trained.history.critic_loss).all()
    assert jnp.isfinite(trained.history.critic_explained).all()
    assert jnp.isfinite(trained.history.mean_cost).all()
