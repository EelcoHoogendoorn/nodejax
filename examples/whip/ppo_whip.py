"""Trains a whip policy with PPO alongside the SHAC learner.

Uses the same plant, policy body, and evaluation while avoiding gradients through
the motor or rope.
"""

import math

import jax
import jax.numpy as jnp
import optax

from nodejax import Leaf, Node, PNode, Struct, carried, node, serial
from nodejax import nn
from examples.rl import control
from examples.rl.distributions import LearnedGaussian, StateIndependentLogStd
from examples.rl.losses import mse
from examples.rl.ppo import ppo_iteration, ppo_training
from examples.whip.learn import (
    HIDDEN,
    MEMORY,
    WhipFeatures,
    WhipMean,
    evaluation_starts,
    random_starts,
    summarize,
    timed,
    whip_evaluation,
    write_artifacts,
)
from examples.whip.plots import plot_training_curves
from examples.whip.whip import whip_plant


N_WORLDS = 32
N_STEPS_PER_CHUNK = 16
N_EPOCHS = 4
N_MINIBATCHES_PER_EPOCH = 4
N_CHUNKS_PER_MINIBATCH = 2
N_CRITIC_PASSES = 4
CLIP = 0.2
DISCOUNT = 0.98
TRACE = 0.95
ENTROPY_WEIGHT = 1e-3
ACTOR_RATE = 1e-3
CRITIC_RATE = 1e-3
INITIAL_STD = 10.0  # rad/s of exploration noise on the setpoint, in the policy's own unit
N_ITERATIONS = 60


@node
def WhipValue(features: Node | PNode, hidden: int) -> Node:
    """Predicts a scalar advantage baseline from `features` of the observation."""
    return serial(
        features=features,
        body=(
            nn.Linear(hidden)
            >> nn.tanh
            >> nn.Linear(hidden)
            >> nn.tanh
            >> nn.Projection()
        ),
    )


@node
def WhipPPOTrainingData(
    plant: PNode,
    iterations: int,
    *,
    n_worlds: int,
    n_chunks: int,
    n_steps_per_chunk: int,
) -> PNode:
    """Samples random `plant` start states per `(iteration, world)` and zero torque disturbances."""
    def apply(rng):
        return Struct(
            initial_plant_state=random_starts(plant, rng.next(), (iterations, n_worlds)),
            disturbance=jnp.zeros((iterations, n_worlds, n_chunks, n_steps_per_chunk)),
        )

    return Leaf(apply)


def whip_policy(features: Node | PNode, memory: Node, initial_std: float = INITIAL_STD) -> Node:
    """Wraps the whip policy mean under a learned Gaussian with initial noise `initial_std` (rad/s)."""
    return LearnedGaussian(
        WhipMean(features, memory, HIDDEN), StateIndependentLogStd(initial=math.log(initial_std)))


def whip_ppo_program(
    policy: Node,
    value: Node,
    plant: PNode,
    iterations: int,
    *,
    n_worlds: int = N_WORLDS,
    n_steps_per_chunk: int = N_STEPS_PER_CHUNK,
    n_epochs: int = N_EPOCHS,
    n_minibatches_per_epoch: int = N_MINIBATCHES_PER_EPOCH,
    n_chunks_per_minibatch: int = N_CHUNKS_PER_MINIBATCH,
    n_critic_passes: int = N_CRITIC_PASSES,
) -> Node:
    """Assembles the complete whip PPO training Node."""
    iteration = ppo_iteration(
        policy,
        value,
        plant,
        value_loss=mse,
        clip=CLIP,
        entropy_weight=ENTROPY_WEIGHT,
        actor_optimizer=optax.adam(ACTOR_RATE),
        critic_optimizer=optax.adam(CRITIC_RATE),
        discount=DISCOUNT,
        trace=TRACE,
        n_worlds=n_worlds,
        n_steps_per_chunk=n_steps_per_chunk,
        n_epochs=n_epochs,
        n_minibatches_per_epoch=n_minibatches_per_epoch,
        n_chunks_per_minibatch=n_chunks_per_minibatch,
        n_critic_passes=n_critic_passes,
    )
    data = WhipPPOTrainingData(
        plant,
        iterations,
        n_worlds=n_worlds,
        n_chunks=n_minibatches_per_epoch * n_chunks_per_minibatch,
        n_steps_per_chunk=n_steps_per_chunk,
    )
    return serial(data=data, training=carried(iteration))


def train_whip_ppo(
    plant: PNode,
    features: Node | PNode,
    *,
    iterations: int = N_ITERATIONS,
    seed: int = 0,
    initial_std: float = INITIAL_STD,
    n_worlds: int = N_WORLDS,
    n_steps_per_chunk: int = N_STEPS_PER_CHUNK,
    n_epochs: int = N_EPOCHS,
    n_minibatches_per_epoch: int = N_MINIBATCHES_PER_EPOCH,
    n_chunks_per_minibatch: int = N_CHUNKS_PER_MINIBATCH,
    n_critic_passes: int = N_CRITIC_PASSES,
) -> Struct:
    """Trains a policy and value network reading `features` of `plant` observations with PPO."""
    program = whip_ppo_program(
        whip_policy(features, nn.GRU(MEMORY), initial_std),
        WhipValue(features, HIDDEN),
        plant,
        iterations,
        n_worlds=n_worlds,
        n_steps_per_chunk=n_steps_per_chunk,
        n_epochs=n_epochs,
        n_minibatches_per_epoch=n_minibatches_per_epoch,
        n_chunks_per_minibatch=n_chunks_per_minibatch,
        n_critic_passes=n_critic_passes,
    )
    outcome = ppo_training(
        program,
        parameter_key=jax.random.fold_in(jax.random.PRNGKey(0), seed),
        training_key=jax.random.fold_in(jax.random.PRNGKey(100), seed),
    )
    return Struct(policy=outcome.policy, history=outcome.history)


def mean_policy(policy: PNode) -> PNode:
    """Extracts the deterministic mean policy from a Gaussian `policy` for evaluation."""
    return policy >> control.ProposalMean()


def main() -> None:
    """Trains a whip policy with PPO and saves training and evaluation figures."""
    plant = whip_plant().parameterize()
    run = timed(lambda: train_whip_ppo(plant, WhipFeatures(tip_sensor=False)))
    print('mean cost per step by iteration:', run.result.history.mean_cost)
    curve = plot_training_curves(
        {'PPO': Struct(curve=run.result.history.mean_cost, seconds=run.seconds)},
        'whip_ppo_training.png')
    print(f'training curve: {curve}')
    evaluation = whip_evaluation(
        mean_policy(run.result.policy), plant, evaluation_starts(plant), jax.random.PRNGKey(21))
    summarize(evaluation, 'PPO')
    write_artifacts(evaluation, 'whip_ppo')


if __name__ == '__main__':
    main()
