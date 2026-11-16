"""Trains a terminal value critic and evaluates receding-horizon MPPI planning on the whip plant."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax import BaseNode, Leaf, Node, PNode, Struct, carried, node, serial, tree_broadcast_axis
from nodejax import nn
from examples.rl import control
from examples.rl.mppi import mppi_controller, mppi_iteration, mppi_training
from examples.whip.learn import (
    evaluation_starts,
    random_starts,
    summarize,
    timed,
    whip_critic,
    whip_evaluation,
    write_artifacts,
)
from examples.whip.plots import plot_training_curves
from examples.whip.whip import N_LINKS, whip_plant


DISCOUNT = 0.98
TRACE = 0.95
NOISE_SCALE = 20.0  # rad/s on the velocity setpoint
NOISE_CORRELATION = 0.9
TEMPERATURE = 30.0

N_ITERATIONS = 40
N_TRAINING_WORLDS = 16
N_TRAINING_REFINEMENTS = 1
N_TRAINING_CANDIDATES = 16
N_STEPS_PER_PLAN = 10
N_STEPS_PER_ITERATION = 150
N_CRITIC_UPDATES = 96
EMA_CRITIC_DECAY = 0.95
CRITIC_RATE = 1e-3

N_EVALUATION_CANDIDATES = 32
N_EVALUATION_REFINEMENTS = 2


def whip_mppi_controller(
    critic: BaseNode,
    plant: PNode,
    *,
    n_candidates: int,
    n_refinements: int,
    n_steps_per_plan: int,
) -> Node | PNode:
    """Builds a receding-horizon MPPI controller for `plant` with unbounded `(N_LINKS,)` velocity setpoints."""
    return mppi_controller(
        critic,
        plant,
        clean=nn.identity,
        noise_scale=NOISE_SCALE,
        noise_correlation=NOISE_CORRELATION,
        temperature=TEMPERATURE,
        discount=DISCOUNT,
        n_candidates=n_candidates,
        n_refinements=n_refinements,
        n_steps_per_plan=n_steps_per_plan,
        command_shape=(N_LINKS,),
    )


@node
def WhipPlanningStarts(
    plant: PNode, n_iterations: int, n_worlds: int, n_steps_per_iteration: int,
) -> PNode:
    """Samples random `(n_iterations, n_worlds)` whip starts broadcast across `n_steps_per_iteration` with zero disturbances."""
    def apply(rng) -> Struct:
        starts = random_starts(plant, rng.next(), (n_iterations, n_worlds))
        return Struct(
            initial_plant_state=tree_broadcast_axis(starts, n_steps_per_iteration, axis=2),
            disturbance=jnp.zeros((n_iterations, n_worlds, n_steps_per_iteration)),
        )

    return Leaf(apply)


def whip_mppi_program(
    n_iterations: int,
    critic: Node,
    plant: PNode,
    *,
    n_worlds: int = N_TRAINING_WORLDS,
    n_refinements: int = N_TRAINING_REFINEMENTS,
    n_candidates: int = N_TRAINING_CANDIDATES,
    n_steps_per_plan: int = N_STEPS_PER_PLAN,
    n_steps_per_iteration: int = N_STEPS_PER_ITERATION,
    n_critic_updates: int = N_CRITIC_UPDATES,
) -> Node:
    """Assembles the outer `carried` MPPI critic-training `Node` over `n_iterations`."""
    controller = whip_mppi_controller(
        critic, plant,
        n_candidates=n_candidates, n_refinements=n_refinements, n_steps_per_plan=n_steps_per_plan)
    iteration = mppi_iteration(
        controller,
        critic,
        plant,
        critic_optimizer=optax.adam(CRITIC_RATE),
        ema_critic_decay=EMA_CRITIC_DECAY,
        discount=DISCOUNT,
        trace=TRACE,
        n_worlds=n_worlds,
        n_steps_per_iteration=n_steps_per_iteration,
        n_critic_updates=n_critic_updates,
    )
    data = WhipPlanningStarts(plant, n_iterations, n_worlds, n_steps_per_iteration)
    return serial(data=data, training=carried(iteration))


def train_whip_mppi(
    plant: PNode,
    critic: Node,
    *,
    n_iterations: int = N_ITERATIONS,
    n_worlds: int = N_TRAINING_WORLDS,
    n_refinements: int = N_TRAINING_REFINEMENTS,
    n_candidates: int = N_TRAINING_CANDIDATES,
    n_steps_per_plan: int = N_STEPS_PER_PLAN,
    n_steps_per_iteration: int = N_STEPS_PER_ITERATION,
    n_critic_updates: int = N_CRITIC_UPDATES,
) -> Struct:
    """Trains `critic` as the terminal value function for MPPI planning on `plant` and returns EMA-bound `critic` and `history`."""
    program = whip_mppi_program(
        n_iterations, critic, plant,
        n_worlds=n_worlds, n_refinements=n_refinements, n_candidates=n_candidates,
        n_steps_per_plan=n_steps_per_plan, n_steps_per_iteration=n_steps_per_iteration,
        n_critic_updates=n_critic_updates)
    trained = mppi_training(
        program,
        parameter_key=jax.random.PRNGKey(1),
        training_key=jax.random.PRNGKey(11),
    )
    return Struct(
        critic=critic.bind(trained.iteration.ema_critic.state),
        history=trained.history,
    )


def learned_controller(
    critic: PNode,
    plant: PNode,
    *,
    n_candidates: int = N_EVALUATION_CANDIDATES,
    n_refinements: int = N_EVALUATION_REFINEMENTS,
    n_steps_per_plan: int = N_STEPS_PER_PLAN,
) -> PNode:
    """Builds the evaluation-budget MPPI controller parameterized around a trained `critic`."""
    return whip_mppi_controller(
        critic, plant,
        n_candidates=n_candidates, n_refinements=n_refinements, n_steps_per_plan=n_steps_per_plan,
    ).parameterize()


def main() -> None:
    plant = whip_plant().parameterize()
    run = timed(lambda: train_whip_mppi(plant, whip_critic(plant)))
    print('mean cost per step by iteration:', run.result.history.mean_cost)
    print('critic explained variance by iteration:', run.result.history.critic_explained)
    curve = plot_training_curves(
        {
            'MPPI': Struct(curve=run.result.history.mean_cost, seconds=run.seconds),
            'critic explained variance': Struct(
                curve=run.result.history.critic_explained, seconds=run.seconds),
        },
        'whip_mppi_training.png')
    print(f'training curve: {curve}')
    controller = learned_controller(run.result.critic, plant)
    evaluation = whip_evaluation(
        controller, plant, evaluation_starts(plant), jax.random.PRNGKey(21), step=control.PlannedStep)
    summarize(evaluation, 'MPPI')
    write_artifacts(evaluation, 'whip_mppi')


if __name__ == '__main__':
    main()
