"""Implements Short-Horizon Actor-Critic from explicit rollout and critic collaborators.

Differentiates a short-horizon closed-loop rollout augmented with terminal
critic values (`TerminalCritic`) under externalized EMA target critic parameters
so policy gradients flow through the plant transitions and terminal state
without differentiating with respect to critic weights. Fits the online critic
to TD-lambda targets evaluated under that same EMA target critic.
"""

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
def bootstrapped_cost(discount: float) -> PNode:
    """Computes SHAC's discounted running-plus-terminal cost objective averaged across worlds."""
    def apply(output: Struct) -> jax.Array:
        return jnp.mean(bootstrapped_costs(
            output.cost,
            output.terminal,
            discount=discount,
        ))

    return Leaf(apply)


@node
def TerminalCritic(rollout: BaseNode, critic: BaseNode) -> Node | PNode:
    """Runs `rollout` and attaches `critic` evaluations of each world's final `next_state` as `terminal`."""
    members = Composite(rollout=rollout, critic=critic)

    def apply(self, input):
        trajectories = self.rollout(bundle=input)
        terminal = self.critic(tree_last(trajectories.next_state, axis=1))
        return trajectories.replace(terminal=terminal)

    return members(apply)


@node
def SHACIteration(
    policy_trainer: BaseNode,
    target_critic: BaseNode,
    critic_trainer: BaseNode,
    ema_critic: BaseNode,
    *,
    discount: float,
    trace: float,
) -> Node:
    """Executes one short-horizon policy gradient step followed by TD-lambda critic updates under an EMA target critic."""
    members = Composite(
        policy_trainer=policy_trainer,
        target_critic=target_critic,
        critic_trainer=critic_trainer,
        ema_critic=ema_critic,
    )

    def apply(self, disturbance, initial_plant_state):
        # Policy rollout and TD-lambda target computation share the lagged EMA critic snapshot.
        ema_critic_param = self.ema_critic(self.critic_trainer.params())
        rollout_input = Struct(
            disturbance=disturbance,
            initial_plant_state=initial_plant_state,
        )
        trajectories = self.policy_trainer(
            input=rollout_input,
            critic=ema_critic_param,
        )
        next_values = self.target_critic(
            input=trajectories.next_state,
            critic=ema_critic_param,
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
    discount: float,
    trace: float,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    ema_critic_decay: float,
    n_worlds: int,
    n_steps_per_chunk: int,
    n_critic_updates: int,
) -> Node:
    """Assembles one SHAC iteration from `policy`, `critic`, and `plant`.

    Carries plant and policy state across chunks while reinitializing at `'episode'`
    boundaries, differentiates through `plant` while training only `'policy'`
    parameters, and updates `critic` over `n_critic_updates` passes.
    """
    transition = state_reinit(ControlledStep(policy, plant), boundary='episode')
    rollout = batch(scan(transition, n=n_steps_per_chunk), n=n_worlds)
    terminal_critic = batch(critic, n=n_worlds)
    trajectory_critic = batch(batch(critic, n=n_steps_per_chunk), n=n_worlds)
    policy_trainer = train_step(
        externalize(TerminalCritic(rollout, terminal_critic), 'critic'),
        bootstrapped_cost(discount=discount),
        actor_optimizer,
        trainable='policy',
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
    """Parameterizes and executes a SHAC `program`, returning the trained policy, final iteration, and loss history."""
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
