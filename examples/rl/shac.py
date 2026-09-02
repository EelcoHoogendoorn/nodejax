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

    ``output`` is a trajectory with ``cost`` on a leading time axis and
    ``terminal`` for the same worlds. No separate target enters this objective.
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

    ``rollout`` maps a chunk input to trajectories with ``cost`` and
    ``next_state`` on a leading time axis; ``critic`` maps a state pytree on
    the remaining axes to a scalar per world. The output is the trajectories
    with the critic's values of the last ``next_state`` added as
    ``terminal``. Externalizing ``critic`` lets a caller supply the target
    critic's parameters on every call.
    """
    members = Composite(rollout=rollout, critic=critic)

    def apply(self, input):
        trajectories = self.rollout(bundle=input)
        terminal = self.critic(tree_last(trajectories.next_state))
        return trajectories.replace(terminal=terminal)

    def init(param, input):
        """The rollout's state, primed from the chunk input. Declared so
        that priming never applies the critic, whose slot may be external."""
        return Struct(rollout=rollout.bind(param.rollout).init(input=input))

    return members(apply, init=init)


@node
def SHACUpdate(
    policy_trainer: Node,
    target_critic: Node,
    critic_trainer: Node,
    ema_critic: Node,
    *,
    discount: float,  # per-step discount on cost and on the critic's value
    trace: float,  # TD-lambda trace; 0 is one-step, 1 is Monte Carlo
) -> Node:
    """One short-horizon policy update and fitted-value update.

    ``MPPIUpdate`` has this same apply with a plain sampler in the policy
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

    def apply(self, disturbance, initial_state):
        # The policy rollout and both critic uses share this lagged snapshot.
        ema_critic_param = self.ema_critic(self.critic_trainer.params())
        rollout_input = Struct(
            disturbance=disturbance,  # per-step external plant forcing
            initial_state=initial_state,
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


def shac_learner(
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
    """Assemble one SHAC update from a policy, a critic, and a plant.

    Physical and policy state carry across chunks and reset when an
    enclosing scan claims the episode. The critic is batched to the
    rollout's world axis for terminal values and to its time axis for
    targets; both uses receive the EMA target parameters as data, and the
    same time-batched critic is what the critic trainer fits.
    """
    transition = state_reinit(ControlledStep(policy, plant), boundary='episode')
    rollout = scan(batch(transition, n=n_worlds), n=n_steps_per_chunk)
    terminal_critic = batch(critic, n=n_worlds)
    trajectory_critic = batch(terminal_critic, n=n_steps_per_chunk, axis='time')
    policy_trainer = train_step(
        externalize(TerminalCritic(rollout, terminal_critic), 'critic'),
        bootstrapped_cost(discount=discount),
        actor_optimizer,
    )
    critic_trainer = iterated(
        train_step(trajectory_critic, mse, critic_optimizer),
        n=n_critic_updates,
    )
    return SHACUpdate(
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

    ``program`` carries a SHAC update over its data. The result holds the
    trained policy, read out of the policy trainer's trained model, the
    learner bound to its final state, from which a caller binds a critic
    to the EMA state, and the loss history.
    """
    final, aux = split_aux(
        jax.jit(program.parameterize(rng=parameter_key).apply)(rng=training_key),
    )
    return Struct(
        policy=final.policy_trainer.trained().pnode.rollout.policy,
        learner=final,
        history=Struct(
            policy_loss=aux.training.policy_trainer.loss.reshape(-1),
            critic_loss=aux.training.critic_trainer.loss[..., -1].reshape(-1),
            mean_cost=aux.training.mean_cost.reshape(-1),
        ),
    )
