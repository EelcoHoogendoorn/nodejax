"""Model Predictive Path Integral control with an injected terminal critic.

The Nodes here own the MPPI operations and receive their collaborators
assembled: proposal batches, plant rollouts, critic batches, repeated
refinements, and training transforms, so the tree shows the candidate,
planning, control, and training axes where they are chosen. The canonical
assembly is here too: ``mppi_controller`` builds the receding planner from a
critic and a plant, ``mppi_learner`` builds one critic-fitting iteration
around it, and ``mppi_training`` runs a program and returns what it trained.
A plant-specific file supplies the plant, the critic, the command range, the
data, and the numbers.
"""

import math

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
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
from examples.rl.control import ControlledStep, OpenLoopStep
from examples.rl.losses import bootstrapped_costs, mse, td_lambda


def mppi_weights(costs: jax.Array, temperature: float) -> jax.Array:
    """Softmax costs shaped (candidate,) into weights normalized over candidates."""
    return jax.nn.softmax(-costs / temperature)


@node
def GaussianProposal(
    noise_scale: float,
    correlation: float = 0.0,
    clean: BaseNode = nn.identity,
) -> Node:
    """Perturb one open-loop plan with optionally correlated Gaussian noise.

    Input and output are arrays shaped ``[time]``. ``clean`` must map one
    ``[time]`` array to another. Correlation is the AR(1) coefficient along
    time. Candidate populations are expressed by batching this Node at the
    assembly site.
    """
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
    proposal: Node,
    rollouts: Node,
    critics: Node,
    *,
    discount: float,
    temperature: float,
) -> Node:
    """Refine one explicit plan from one fixed initial plant state.

    ``proposal`` maps controls shaped ``[time]`` to candidates shaped
    ``[candidate, time]``. ``rollouts`` accepts that candidate array and one
    initial plant state; it returns ``cost`` shaped ``[candidate, time]`` and
    a matching ``next_state`` pytree. The rollout Node owns any environment
    inputs and axis preparation. ``critics`` maps the candidate-axis terminal
    state pytree to values shaped ``[candidate]``. The current plan is included
    as an additional candidate. The output is ``Struct(initial_state,
    controls)`` with controls shaped ``[time]``.
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
        refined = jnp.sum(weights[:, None] * candidates, axis=0)
        return Struct(initial_state=initial_state, controls=refined)

    return members(apply)


@node
def RecedingMPPI(plan: Node, refinements: Node) -> Node:
    """Refine a stored plan, execute its first command, then shift it.

    ``plan`` is a cyclic Node whose state is the current ``[time]`` plan and
    whose input replaces that state. ``refinements`` accepts ``initial_state``
    and ``controls`` shaped ``[time]`` and returns
    ``Struct(initial_state, controls)`` with the same shapes. This Node accepts
    one plant-state pytree and returns one scalar command.
    """
    members = Composite(plan=plan, refinements=refinements)

    def apply(self, input):
        refined = self.refinements(
            initial_state=input,
            controls=self.state.plan,
        )
        controls = refined.controls
        shifted = jnp.concatenate((controls[1:], jnp.zeros_like(controls[:1])))
        self.plan(shifted)
        return controls[0]

    return members(apply)


@node
def MPPIUpdate(
    sampler: Node,
    target_critic: Node,
    critic_trainer: Node,
    ema_critic: Node,
    *,
    discount: float,  # per-step discount on cost and on the critic's value
    trace: float,  # TD-lambda trace; 0 is one-step, 1 is Monte Carlo
) -> Node:
    """Sample one control chunk, fit its critic, and update the EMA target.

    ``SHACUpdate`` has this same apply with a policy trainer in the sampler's
    place; the two stay separate so each example reads on its own.
    ``sampler`` accepts ``initial_state`` leaves shaped (world, time, ...) and
    ``disturbance`` shaped (world, time). It also accepts the target critic's
    parameter pytree through its externalized ``critics`` field and returns a
    trajectory whose state leaves begin (world, time, ...) and whose ``cost``
    is shaped (world, time). ``target_critic`` values that trajectory's next
    states, its whole parameter tree arriving in its ``critic`` field.
    ``critic_trainer`` is cyclic, exposes the fitted parameters as
    ``params()``, and accepts trajectory state as ``input`` plus the
    TD-lambda target array as ``target``. ``ema_critic`` smooths those
    parameters into the target critic, which is the one value function the
    planner steers by and the targets are formed under; the fitted critic
    reaches the planner only through it, so one fit cannot move the next
    batch of data.
    """
    members = Composite(
        sampler=sampler,
        target_critic=target_critic,
        critic_trainer=critic_trainer,
        ema_critic=ema_critic,
    )

    def apply(self, initial_state, disturbance):
        critic_param = self.critic_trainer.params()
        ema_critic_param = self.ema_critic(critic_param)
        trajectory = drop_aux(self.sampler(
            initial_state=initial_state,
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
        return mean_cost, Aux(mean_cost=mean_cost)

    return members(apply)


@node
def CandidateRollouts(rollouts: Node) -> Node:
    """Roll one start out open loop under every candidate plan.

    ``candidates`` is shaped (candidate, time), while ``initial_state`` is one
    unbatched plant-state pytree. Returned state leaves begin
    (candidate, time, ...), and ``cost`` is shaped (candidate, time).

    The wrapped Node is ``batch(scanned(OpenLoopStep))``: batch consumes
    candidates and scan consumes time, the order the plans arrive in. The
    one start is repeated over candidates, and the disturbance is zero at
    every step.
    """
    def apply(self, initial_state, candidates):
        return self.rollouts(
            command=candidates,
            disturbance=jnp.zeros_like(candidates),
            initial_state=tile(initial_state, candidates.shape[0]),
        )

    return Wrapper(rollouts=rollouts)(apply)


def mppi_controller(
    critic: BaseNode,
    plant: BaseNode,
    *,
    clean: BaseNode,  # projects each candidate plan onto the admissible command range
    noise_scale: float,  # proposal noise std in command units
    noise_correlation: float,  # AR(1) coefficient of that noise along the plan; 0 is white
    temperature: float,  # softmax temperature over candidate costs; lower trusts the best more
    discount: float,  # per-step discount on rollout cost and on the critic's terminal value
    n_candidates: int,
    n_refinements: int,  # MPPI refinements of the warm plan per control step
    n_steps_per_plan: int,  # horizon in control steps; one scalar command per step
) -> Node:
    """Assemble the receding MPPI controller from a critic and a plant.

    Every control step perturbs the carried plan into candidates, rolls each
    out open loop, scores them with the discounted cost plus the critic's
    terminal value, and moves the plan toward the exponentially weighted
    mean; the first command is executed and the plan shifts.
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
    plan = control.Delay().with_input(jnp.zeros((n_steps_per_plan,)))
    return RecedingMPPI(
        plan=plan,
        refinements=repeat(refinement, n=n_refinements),
    ).with_input(plant.initialize().state)


def mppi_learner(
    controller: Node,
    critic: Node,
    plant: BaseNode,
    *,
    critic_optimizer,  # Optax transformation for the critic parameters
    ema_critic_decay: float,  # EMA decay of the target critic's parameters
    discount: float,  # per-step discount on cost and on the critic's value
    trace: float,  # TD-lambda trace; 0 is one-step, 1 is Monte Carlo
    n_worlds: int,
    n_steps_per_iteration: int,  # physical steps every world runs per iteration
    n_critic_updates: int,  # critic fits per iteration
) -> Node:
    """Assemble one critic-fitting iteration around a receding controller.

    The returned iteration accepts ``initial_state`` leaves shaped
    (world, time, ...) and ``disturbance`` shaped (world, time).

    The controller plans under the EMA target critic, whose parameters reach
    it as data through the planner's critic batch, and the sampled
    trajectories fit the online critic to TD-lambda targets under that same
    target critic.
    """
    sampler = externalize(
        batch(scanned(ControlledStep(controller, plant)), n=n_worlds),
        'policy.refinements.critics',
    )
    trajectory_critic = batch(batch(critic, n=n_steps_per_iteration), n=n_worlds)
    critic_trainer = iterated(
        train_step(trajectory_critic, mse, critic_optimizer),
        n=n_critic_updates,
    )
    return MPPIUpdate(
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
    """Run an MPPI critic-training program and return what it trained.

    ``program`` carries an MPPI iteration over its data. The result holds the
    iteration bound to its final state, from which a caller binds a critic
    to the EMA state, and the history.
    """
    final, aux = split_aux(
        jax.jit(program.parameterize(rng=parameter_key).apply)(rng=training_key),
    )
    return Struct(
        learner=final,
        history=Struct(
            critic_loss=aux.training.critic_trainer.loss[..., -1].reshape(-1),
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )
