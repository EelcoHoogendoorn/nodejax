"""Implements Model Predictive Path Integral control with an injected terminal critic.

Composes candidate proposal batches, open-loop plant rollouts, terminal critic
batches, repeated plan refinements, and TD-lambda critic updates from explicit
collaborators. Exposes `mppi_controller` for receding-horizon planning,
`mppi_iteration` for online critic fitting under an EMA target critic, and
`mppi_training` for running a compiled critic-learning loop.
"""

import math

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    Struct,
    Wrapper,
    batch,
    control,
    drop_aux,
    externalize,
    iterated,
    node,
    repeat,
    scanned,
    split_aux,
    tile,
    train_step,
    tree_last,
)
from nodejax import nn
from examples.rl.control import OpenLoopStep, PlannedStep, initial_state
from examples.rl.losses import bootstrapped_costs, mse, td_lambda


def mppi_weights(costs: jax.Array, temperature: float) -> jax.Array:
    """Normalizes `(candidate,)` rollout costs into softmax weights at `temperature`."""
    return jax.nn.softmax(-costs / temperature)


@node
def GaussianProposal(
    noise_scale: float,
    correlation: float = 0.0,
    clean: BaseNode = nn.identity,
) -> Node | PNode:
    """Perturbs one `[time, *command]` open-loop plan with AR(1)-correlated Gaussian noise and applies `clean`."""
    if not math.isfinite(correlation) or not 0.0 <= correlation < 1.0:
        raise ValueError('GaussianProposal correlation must be in [0, 1)')

    def apply(self, input, rng):
        white = jax.random.normal(rng.next(), input.shape)
        if correlation == 0.0:
            noise = white
        else:
            innovation_scale = jnp.sqrt(1.0 - correlation**2)

            def smooth(previous, innovation):
                current = correlation * previous + innovation_scale * innovation
                return current, current

            tail = jax.lax.scan(smooth, white[0], white[1:])[1]
            noise = jnp.concatenate((white[:1], tail), axis=0)
        return self.clean(input + noise_scale * noise)

    return Wrapper(clean=clean)(apply)


@node
def MPPIStep(
    proposal: BaseNode,
    rollouts: BaseNode,
    critics: BaseNode,
    *,
    discount: float,
    temperature: float,
) -> Node | PNode:
    """Refines one open-loop `controls` sequence from a fixed `initial_state`.

    Concatenates the warm plan with `proposal(controls)` candidates, evaluates open-loop
    `rollouts` and terminal `critics`, and returns softmax-weighted `controls` paired
    with `initial_state`.
    """
    members = Composite(
        proposal=proposal,
        rollouts=rollouts,
        critics=critics,
    )

    def apply(self, initial_state, controls):
        proposed = self.proposal(controls)
        candidates = jnp.concatenate((controls[None], proposed), axis=0)
        trajectories = self.rollouts(
            initial_state=initial_state,
            candidates=candidates,
        )
        terminals = drop_aux(
            self.critics(tree_last(trajectories.next_state, axis=1)))
        costs = bootstrapped_costs(
            trajectories.cost,
            terminals,
            discount=discount,
        )
        weights = mppi_weights(costs, temperature)
        weights = weights.reshape((-1,) + (1,) * (candidates.ndim - 1))
        refined = jnp.sum(weights * candidates, axis=0)
        return Struct(initial_state=initial_state, controls=refined)

    return members(apply)


@node
def RecedingMPPI(plan: BaseNode, refinements: BaseNode) -> Node | PNode:
    """Refines a carried `[time, *command]` `plan` from the current plant state, emits its first command, and shifts the tail."""
    members = Composite(plan=plan, refinements=refinements)

    def apply(self, input):
        refined = self.refinements(
            initial_state=input,
            controls=self.plan.state,
        )
        controls = refined.controls
        shifted = jnp.concatenate((controls[1:], jnp.zeros_like(controls[:1])))
        self.plan(shifted)
        return controls[0]

    return members(apply)


@node
def MPPIIteration(
    sampler: BaseNode,
    target_critic: BaseNode,
    critic_trainer: BaseNode,
    ema_critic: BaseNode,
    *,
    discount: float,  # per-step discount on cost and critic value
    trace: float,  # TD-lambda trace where 0 is one-step and 1 is Monte Carlo
) -> Node:
    """Samples one closed-loop control chunk under the EMA target critic, fits `critic_trainer` to TD-lambda targets, and updates `ema_critic`."""
    members = Composite(
        sampler=sampler,
        target_critic=target_critic,
        critic_trainer=critic_trainer,
        ema_critic=ema_critic,
    )

    def apply(self, initial_plant_state, disturbance):
        critic_param = self.critic_trainer.params()
        ema_critic_param = self.ema_critic(critic_param)
        trajectory = drop_aux(self.sampler(
            initial_plant_state=initial_plant_state,
            disturbance=disturbance,
            critics=ema_critic_param,
        ))
        next_values = self.target_critic(
            input=trajectory.next_state,
            critic=ema_critic_param,
        )
        targets = td_lambda(
            trajectory.cost,
            next_values,
            discount=discount,
            trace=trace,
        )
        self.critic_trainer(input=trajectory.state, target=targets)

        mean_cost = jnp.mean(trajectory.cost)
        return mean_cost, Aux(mean_cost=mean_cost, target_variance=jnp.var(targets))

    return members(apply)


@node
def CandidateRollouts(rollouts: BaseNode) -> Node | PNode:
    """Tiles a single `initial_state` across `(candidate, time, *command)` `candidates` and evaluates open-loop `rollouts` with zero disturbance."""
    def apply(self, initial_state, candidates):
        return self.rollouts(
            command=candidates,
            disturbance=jnp.zeros(candidates.shape[:2]),
            initial_state=tile(initial_state, candidates.shape[0]),
        )

    return Wrapper(rollouts=rollouts)(apply)


def mppi_controller(
    critic: BaseNode,
    plant: BaseNode,
    *,
    clean: BaseNode,  # projects each candidate plan onto the admissible command range
    noise_scale: float,  # proposal noise std in command units
    noise_correlation: float,  # AR(1) coefficient along the plan where 0 is white noise
    temperature: float,  # softmax temperature over candidate costs
    discount: float,  # per-step discount on rollout cost and terminal critic value
    n_candidates: int,
    n_refinements: int,  # MPPI refinements of the warm plan per control step
    n_steps_per_plan: int,  # horizon in control steps
    command_shape: tuple[int, ...] = (),  # shape of one command, () for scalar
) -> Node | PNode:
    """Assembles a receding-horizon MPPI controller from `critic` and `plant`.

    Perturbs the carried plan into `n_candidates` trajectories, scores them with
    discounted running costs plus terminal `critic` values across `n_refinements`
    iterations, emits the first command, and shifts the warm plan forward.
    """
    candidate_plans = Leaf(lambda input: tile(input, n_candidates), name='candidate_plans')
    proposal = candidate_plans >> batch(
        GaussianProposal(noise_scale=noise_scale, correlation=noise_correlation, clean=clean),
        n=n_candidates,
    )
    rollouts = CandidateRollouts(batch(scanned(OpenLoopStep(plant)), n=n_candidates + 1))
    refinement = MPPIStep(
        proposal=proposal,
        rollouts=rollouts,
        critics=batch(critic, n=n_candidates + 1),
        discount=discount,
        temperature=temperature,
    )
    plan = control.Delay().with_input(jnp.zeros((n_steps_per_plan,) + tuple(command_shape)))
    return RecedingMPPI(
        plan=plan,
        refinements=repeat(refinement, n=n_refinements),
    ).with_input(initial_state(plant))


def mppi_iteration(
    controller: BaseNode,
    critic: Node,
    plant: BaseNode,
    *,
    critic_optimizer: optax.GradientTransformation,
    ema_critic_decay: float,  # EMA decay of the target critic parameters
    discount: float,  # per-step discount on cost and critic value
    trace: float,  # TD-lambda trace where 0 is one-step and 1 is Monte Carlo
    n_worlds: int,
    n_steps_per_iteration: int,  # physical steps run per world per iteration
    n_critic_updates: int,  # critic fits per iteration
) -> Node:
    """Assembles one critic-fitting MPPI iteration around `controller`, `critic`, and `plant`.

    Externalizes the planner's critic parameters so `controller` plans under the EMA
    target critic while `critic` fits TD-lambda targets computed under that same target.
    """
    sampler = externalize(
        batch(scanned(PlannedStep(controller, plant)), n=n_worlds),
        'policy.refinements.critics',
    )
    trajectory_critic = batch(batch(critic, n=n_steps_per_iteration), n=n_worlds)
    critic_trainer = iterated(
        train_step(trajectory_critic, mse, critic_optimizer),
        n=n_critic_updates,
    )
    return MPPIIteration(
        sampler=sampler,
        target_critic=externalize(trajectory_critic, field='critic'),
        critic_trainer=critic_trainer,
        ema_critic=nn.EMA(tau=ema_critic_decay, warm=True),
        discount=discount,
        trace=trace,
    )


def mppi_training(
    program: Node,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Parameterizes and executes an MPPI critic-training `program`, returning the final iteration state and loss history."""
    final, aux = split_aux(
        jax.jit(program.parameterize(rng=parameter_key).apply)(rng=training_key),
    )
    critic_loss = aux.training.critic_trainer.loss[..., -1].reshape(-1)
    target_variance = aux.training.target_variance.reshape(-1)
    return Struct(
        iteration=final,
        history=Struct(
            critic_loss=critic_loss,
            critic_explained=1.0 - critic_loss / target_variance,
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )
