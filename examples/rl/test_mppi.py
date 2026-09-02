"""Focused behavior checks for the generic MPPI Nodes."""

import jax
import jax.numpy as jnp

from nodejax import Leaf, Struct, control
from examples.rl.losses import bootstrapped_costs
from examples.rl.mppi import (
    GaussianProposal,
    MPPIStep,
    RecedingMPPI,
    mppi_weights,
)


def test_gaussian_proposal_draws_one_bounded_plan() -> None:
    controls = jnp.zeros((8,))
    clean = Leaf(
        lambda input: jnp.clip(input, -0.25, 0.25),
        name='bounded_plan',
    )
    proposal = GaussianProposal(
        noise_scale=2.0,
        correlation=0.5,
        clean=clean,
    ).parameterize()

    proposed = proposal.apply(controls, rng=jax.random.PRNGKey(1))

    assert proposed.shape == controls.shape
    assert jnp.any(proposed != controls)
    assert jnp.max(jnp.abs(proposed)) <= 0.25


def test_mppi_step_matches_its_explicit_candidate_costs() -> None:
    target = 0.75
    initial_state = jnp.asarray(-0.25)
    controls = jnp.zeros((4,))
    proposed = jnp.asarray((
        (0.4, 0.4, -0.2, -0.2),
        (-0.5, 0.1, 0.3, 0.0),
    ))

    def quadratic_rollout(initial_state, candidates):
        """Accumulate each plan and charge squared distance from the target."""
        state = initial_state + jnp.cumsum(candidates.T, axis=0)
        return Struct(cost=(state - target) ** 2, next_state=state)

    refinement = MPPIStep(
        proposal=Leaf(
            lambda input: proposed,
            name='fixed_candidates',
        ),
        rollouts=Leaf(quadratic_rollout, name='quadratic_rollouts'),
        critics=Leaf(
            lambda input: jnp.zeros_like(input),
            name='zero_values',
        ),
        discount=0.97,
        temperature=0.3,
    ).parameterize()

    result = refinement.apply(
        initial_state=initial_state,
        controls=controls,
    )
    candidates = jnp.concatenate((controls[None], proposed), axis=0)
    states = initial_state + jnp.cumsum(candidates.T, axis=0)
    costs = bootstrapped_costs(
        (states - target) ** 2,
        jnp.zeros((candidates.shape[0],)),
        discount=0.97,
    )
    expected = jnp.sum(
        mppi_weights(costs, 0.3)[:, None] * candidates,
        axis=0,
    )

    assert jnp.allclose(result.controls, expected)
    assert result.initial_state == initial_state


def test_receding_mppi_executes_then_shifts_its_plan() -> None:
    increments = jnp.asarray((1.0, 2.0, 3.0))
    refinements = Leaf(
        lambda initial_state, controls: Struct(
            initial_state=initial_state,
            controls=controls + increments,
        ),
        name='fixed_refinements',
    )
    controller = RecedingMPPI(
        plan=control.Delay().with_input(jnp.zeros((3,))),
        refinements=refinements,
    ).parameterize().initialize()

    controller, first = controller(jnp.asarray(0.0))
    assert jnp.allclose(first, 1.0)
    assert jnp.allclose(controller.state.plan, jnp.asarray((2.0, 3.0, 0.0)))

    controller, second = controller(jnp.asarray(0.0))
    assert jnp.allclose(second, 3.0)
    assert jnp.allclose(controller.state.plan, jnp.asarray((5.0, 3.0, 0.0)))
