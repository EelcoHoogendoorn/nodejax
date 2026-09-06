"""Train a terminal critic for MPPI on the whip, the planner beside the two
learned policies.

MPPI plans through the plant's own model from the plant's true state, so
it is the privileged controller here: no policy network, only a critic
that learns what a state is worth past the planning horizon. The plant,
the critic, and the evaluation are the ones the policies use.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Leaf, Node, PNode, Struct, carried, node, serial, tree_broadcast_axis
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
NOISE_SCALE = 20.0  # rad/s on the setpoint, the plant's own unit
NOISE_CORRELATION = 0.9
TEMPERATURE = 30.0  # candidate costs are summed tip energies in the hundreds

N_ITERATIONS = 40
N_TRAINING_WORLDS = 16
N_TRAINING_REFINEMENTS = 1
N_TRAINING_CANDIDATES = 16
N_STEPS_PER_PLAN = 10  # a tenth of a second of policy steps
# An iteration runs each world for a second and a half, the evaluation's length: at half a
# second the far starts never reached a target, the critic learned that far was worthless, and
# the planner had nothing to travel toward.
N_STEPS_PER_ITERATION = 150
# Critic fits per iteration: the planning rollouts cost far more than a fit, and at four the
# critic never converged on the whip.
N_CRITIC_UPDATES = 96
EMA_CRITIC_DECAY = 0.95
CRITIC_RATE = 1e-3

N_EVALUATION_CANDIDATES = 32
N_EVALUATION_REFINEMENTS = 2


def whip_mppi_controller(
    critic: Node | PNode,
    plant: PNode,
    *,
    n_candidates: int,
    n_refinements: int,
    n_steps_per_plan: int,
) -> Node:
    """The receding MPPI controller with the whip's command shape and noise;
    the setpoint is unbounded, so no projection cleans the plans."""
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
) -> Node:
    """Random starts of ``plant``, a rope at rest and a target, per
    (iteration, world), repeated across time since the step consumes one
    only when a run begins, and zero disturbances shaped (iteration, world,
    time)."""
    def apply(rng):
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
    """Assemble the complete whip MPPI critic-training tree."""
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
    """Train ``critic`` as the terminal critic of ``plant`` under fixed
    keys. The planner reads the true state, so the plant's sensing does
    not reach it, only its actuators do. The result holds the critic bound
    to its EMA parameters and the history."""
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
    """The receding controller planning under a learned critic."""
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
