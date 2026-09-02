"""Focused integration checks for the Pendulum MPPI example."""

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
from examples.rl.losses import bootstrapped_costs
from examples.rl.mppi import CandidateRollouts, GaussianProposal, MPPIStep, mppi_training
from examples.rl.mppi_pendulum import (
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
    """Roll out one world-major control plan from each matching start.

    ``controls`` has shape ``[world, time]`` and every leaf in ``starts`` has
    leading shape ``[world]``. The rollout is ``scanned(batch(step))``, so its
    sequence input is time-major: scan consumes the first axis and batch
    consumes the second. The starts are the step's state input, given once.
    Pendulum has no disturbance in this check, but its step contract still
    requires the field.
    """
    n_worlds = tree_len(starts)
    n_steps = controls.shape[1]
    rollout = scanned(batch(OpenLoopStep(plant), n=n_worlds)).parameterize()
    return drop_aux(rollout.apply(
        command=jnp.swapaxes(controls, 0, 1),
        disturbance=jnp.zeros((n_steps, n_worlds)),
        initial_state=starts,
    ))


def test_gaussian_refinement_improves_the_plan() -> None:
    plant = Pendulum()
    n_candidates = 64
    n_steps = 10
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
        """Assign no cost beyond the sampled trajectory."""
        return jnp.zeros_like(terminal_state.angle)

    zero_value = Leaf(zero_terminal_value, name='zero_terminal_value')
    open_loop_rollouts = scanned(
        batch(OpenLoopStep(plant), n=n_candidates + 1),
    )
    refinement = MPPIStep(
        proposal=proposal,
        rollouts=CandidateRollouts(open_loop_rollouts),
        critics=batch(zero_value, n=n_candidates + 1),
        discount=DISCOUNT,
        temperature=TEMPERATURE,
    )
    planner = repeat(refinement, n=12).with_input(bundle=Struct(
        initial_state=start,
        controls=controls,
    )).parameterize()

    result = planner.apply(
        initial_state=start,
        controls=controls,
        rng=jax.random.PRNGKey(5),
    )
    before = open_loop_trajectory(controls[None], tile(start, 1), plant)
    after = open_loop_trajectory(result.controls[None], tile(start, 1), plant)
    before_cost = bootstrapped_costs(
        before.cost,
        jnp.zeros((1,)),
        discount=DISCOUNT,
    )[0]
    after_cost = bootstrapped_costs(
        after.cost,
        jnp.zeros((1,)),
        discount=DISCOUNT,
    )[0]

    assert after_cost < before_cost


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

    assert jnp.isfinite(critic.bind(trained.learner.ema_critic.state).apply(state))
    assert jnp.isfinite(trained.history.critic_loss).all()
    assert jnp.isfinite(trained.history.mean_cost).all()
