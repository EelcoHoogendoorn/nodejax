"""Short-Horizon Actor-Critic from complete collaborators.

The differentiated Node is a rollout that carries its own terminal values:
a critic evaluates the final plant states, and the policy objective is a
plain function of that output. The critic's parameters are externalized,
so every update supplies the EMA target critic as data and the gradient
reaches the terminal values only through the plant's final state, which
is the algorithm. The critic then fits TD-lambda targets formed under the
same target critic. The update Node receives the two trainers, the target
Node, and the EMA as members and holds only the glue between them; the
caller assembles those from the policy, the critic, and the plant with
stock transforms.
"""

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    Struct,
    batch,
    externalize,
    iterated,
    node,
    scan,
    split_aux,
    state_reinit,
    train_step,
    tree_last,
)
from nodejax import nn
from examples.rl.control import ControlledStep
from examples.rl.losses import bootstrapped_costs, mse, td_lambda


@node
def bootstrapped_cost(discount: float) -> Node:
    """SHAC's policy objective over a rollout that carries its terminal values.

    ``output`` is a trajectory with ``cost`` shaped (world, time) and
    ``terminal`` shaped (world,). Time is discounted independently per world,
    then the resulting values are averaged over worlds. No separate target
    enters this objective.
    """
    def apply(output: Struct) -> jax.Array:
        return jnp.mean(bootstrapped_costs(
            output.cost,
            output.terminal,
            discount=discount,
        ))

    return Leaf(apply)


@node
def TerminalCritic(rollout: Node, critic: Node) -> Node:
    """Value the final states of a rollout with a critic.

    ``input`` fields and rollout state leaves are shaped (world, time, ...),
    with ``cost`` shaped (world, time). ``critic`` maps the final state from
    each world to ``terminal`` shaped (world,). Externalizing ``critic`` lets
    a caller supply the target critic's parameters on every call.
    """
    members = Composite(rollout=rollout, critic=critic)

    def apply(self, input):
        trajectories = self.rollout(bundle=input)
        terminal = self.critic(tree_last(trajectories.next_state, axis=1))
        return trajectories.replace(terminal=terminal)

    return members(apply)


@node
def SHACIteration(
    policy_trainer: Node,
    target_critic: Node,
    critic_trainer: Node,
    ema_critic: Node,
    *,
    discount: float,  # per-step discount on cost and on the critic's value
    trace: float,  # TD-lambda trace; 0 is one-step, 1 is Monte Carlo
) -> Node:
    """One short-horizon policy update and fitted-value update.

    ``disturbance`` is shaped (world, time), and ``initial_plant_state`` leaves
    begin (world, time, ...). Trajectories, target-critic values, and TD-lambda
    targets retain that axis order.

    ``MPPIIteration`` has this same apply with a plain sampler in the policy
    trainer's place; the two stay separate so each example reads on its own.
    ``policy_trainer`` differentiates a rollout that carries its terminal
    values under an externalized ``critic`` and returns the trajectories it
    stepped through. ``target_critic`` values those trajectories' next
    states, its whole parameter tree arriving in its ``critic`` field.
    ``critic_trainer`` fits the critic on trajectory states to the TD-lambda
    targets and exposes the fitted parameters as ``params()``.
    ``ema_critic`` smooths those parameters, and the smoothed copy is the
    target critic the other two members receive. The output is the
    trajectories of the pre-update forward pass, as for any train step; the
    mean cost per step rides on aux.
    """
    members = Composite(
        policy_trainer=policy_trainer,
        target_critic=target_critic,
        critic_trainer=critic_trainer,
        ema_critic=ema_critic,
    )

    def apply(self, disturbance, initial_plant_state):
        # The policy rollout and both critic uses share this lagged snapshot.
        ema_critic_param = self.ema_critic(self.critic_trainer.params())
        rollout_input = Struct(
            disturbance=disturbance,  # per-step external plant forcing
            initial_plant_state=initial_plant_state,
        )
        trajectories = self.policy_trainer(
            input=rollout_input,
            critic=ema_critic_param,  # fills TerminalCritic's externalized critic member
        )
        next_values = self.target_critic(
            input=trajectories.next_state,
            critic=ema_critic_param,  # fills the externalized critic member
        )
        targets = td_lambda(
            trajectories.cost,
            next_values,
            discount=discount,
            trace=trace,
        )
        self.critic_trainer(input=trajectories.state, target=targets)
        return trajectories, Aux(mean_cost=jnp.mean(trajectories.cost))

    return members(apply)


def shac_iteration(
    policy: Node,
    critic: Node,
    plant: BaseNode,
    *,
    discount: float,  # per-step discount on cost and on the critic's value
    trace: float,  # TD-lambda trace; 0 is one-step, 1 is Monte Carlo
    actor_optimizer,  # Optax transformation for the policy parameters
    critic_optimizer,  # Optax transformation for the critic parameters
    ema_critic_decay: float,  # EMA decay of the target critic's parameters
    n_worlds: int,
    n_steps_per_chunk: int,  # differentiated horizon in plant steps
    n_critic_updates: int,  # critic fits per policy update
) -> Node:
    """Assemble one SHAC iteration from a policy, a critic, and a plant.

    Plant and policy state carry across chunks and reinitialize from
    ``initial_plant_state`` when an enclosing scan claims the episode. The
    critic is batched to the rollout's world axis for terminal values and to
    both its axes for targets; both uses receive the EMA target parameters as
    data, and the same trajectory critic is what the critic trainer fits.
    """
    transition = state_reinit(ControlledStep(policy, plant), boundary='episode')
    rollout = batch(scan(transition, n=n_steps_per_chunk), n=n_worlds)
    terminal_critic = batch(critic, n=n_worlds)
    trajectory_critic = batch(batch(critic, n=n_steps_per_chunk), n=n_worlds)
    policy_trainer = train_step(
        externalize(TerminalCritic(rollout, terminal_critic), 'critic'),
        bootstrapped_cost(discount=discount),
        actor_optimizer,
    )
    critic_trainer = iterated(
        train_step(trajectory_critic, mse, critic_optimizer),
        n=n_critic_updates,
    )
    return SHACIteration(
        policy_trainer=policy_trainer,
        target_critic=externalize(trajectory_critic, field='critic'),
        critic_trainer=critic_trainer,
        ema_critic=nn.EMA(tau=ema_critic_decay, warm=True),
        discount=discount,
        trace=trace,
    )


def shac_training(
    program: Node,
    *,
    parameter_key: jax.Array,
    training_key: jax.Array,
) -> Struct:
    """Run a SHAC program and return what it trained.

    ``program`` carries a SHAC iteration over its data. The result holds the
    trained policy, read out of the policy trainer's trained model, the
    iteration bound to its final state, from which a caller binds a critic
    to the EMA state, and the loss history.
    """
    final, aux = split_aux(
        jax.jit(program.parameterize(rng=parameter_key).apply)(rng=training_key),
    )
    return Struct(
        policy=final.policy_trainer.trained().pnode.rollout.policy,
        iteration=final,
        history=Struct(
            policy_loss=aux.training.policy_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss[..., -1].reshape(-1),
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )
