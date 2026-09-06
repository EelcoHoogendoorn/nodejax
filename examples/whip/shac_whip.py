"""Train a whip policy with SHAC through the actuator and the rope.

The learner is the RL module's SHAC unchanged, the plant is the whip. The
policy sees the actuator's own estimates, the critic sees the rope, and
the gradient runs through the motor's current loops and the rope's
constraint projection alike.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Leaf, Node, PNode, Struct, carried, node, scan, serial, tree_broadcast_axis
from nodejax import nn
from examples.rl.shac import shac_iteration, shac_training
from examples.whip.learn import (
    HIDDEN,
    MEMORY,
    WhipFeatures,
    WhipMean,
    evaluation_starts,
    random_starts,
    summarize,
    timed,
    untrained_policy,
    whip_critic,
    whip_evaluation,
    write_artifacts,
)
from examples.whip.plots import plot_training_curves
from examples.whip.whip import whip_plant


DISCOUNT = 0.98
TRACE = 0.95
N_STEPS_PER_CHUNK = 20
N_WORLDS = 16
N_CHUNKS_PER_EPISODE = 8
N_EPISODES = 40
N_CRITIC_UPDATES = 4
EMA_CRITIC_DECAY = 0.99
ACTOR_RATE = 0.0005
ACTOR_GRADIENT_CLIP = 1.0  # global norm: the crack makes rollout gradients spike
ACTOR_OPTIMIZER = optax.chain(
    optax.clip_by_global_norm(ACTOR_GRADIENT_CLIP), optax.adam(ACTOR_RATE))
CRITIC_RATE = 0.001
DISTURBANCE_SCALE = 0.5


@node
def WhipTrainingData(
    plant: PNode,
    *,
    n_episodes: int,
    n_chunks_per_episode: int,
    n_steps_per_chunk: int,
    n_worlds: int,
    disturbance_scale: float,
) -> Node:
    """Random starts of ``plant``, a rope at rest and a target, and torque
    disturbances for chunked SHAC episodes, shaped (episode, chunk, world,
    time). The plant derives the rest of its state from the start at the
    episode boundary, so the start is repeated over chunks and time."""
    def apply(rng):
        disturbance = disturbance_scale * jax.random.normal(
            rng.next(),
            (n_episodes, n_chunks_per_episode, n_worlds, n_steps_per_chunk),
        )
        starts = random_starts(plant, rng.next(), (n_episodes, n_worlds))
        initial_plant_state = tree_broadcast_axis(
            tree_broadcast_axis(starts, n_chunks_per_episode, axis=1), n_steps_per_chunk, axis=3)
        return Struct(disturbance=disturbance, initial_plant_state=initial_plant_state)

    return Leaf(apply)


def whip_training_program(
    policy: Node,
    critic: Node,
    plant: PNode,
    n_episodes: int,
    *,
    actor_optimizer=ACTOR_OPTIMIZER,
    n_worlds: int = N_WORLDS,
    n_chunks_per_episode: int = N_CHUNKS_PER_EPISODE,
    n_steps_per_chunk: int = N_STEPS_PER_CHUNK,
) -> Node:
    """Assemble the complete whip SHAC Node tree; ``actor_optimizer`` is
    the optax transformation on the policy's parameters."""
    iteration = shac_iteration(
        policy,
        critic,
        plant,
        discount=DISCOUNT,
        trace=TRACE,
        actor_optimizer=actor_optimizer,
        critic_optimizer=optax.adam(CRITIC_RATE),
        ema_critic_decay=EMA_CRITIC_DECAY,
        n_worlds=n_worlds,
        n_steps_per_chunk=n_steps_per_chunk,
        n_critic_updates=N_CRITIC_UPDATES,
    )
    episode = scan(iteration, boundary='episode', n=n_chunks_per_episode)
    data = WhipTrainingData(
        plant,
        n_episodes=n_episodes,
        n_chunks_per_episode=n_chunks_per_episode,
        n_steps_per_chunk=n_steps_per_chunk,
        n_worlds=n_worlds,
        disturbance_scale=DISTURBANCE_SCALE,
    )
    return serial(data=data, training=carried(episode))


def train_whip_shac(
    plant: PNode,
    features: Node,
    *,
    n_episodes: int = N_EPISODES,
    seed: int = 0,
    actor_optimizer=ACTOR_OPTIMIZER,
    n_worlds: int = N_WORLDS,
    n_chunks_per_episode: int = N_CHUNKS_PER_EPISODE,
    n_steps_per_chunk: int = N_STEPS_PER_CHUNK,
) -> Struct:
    """Train a policy reading ``features`` of what ``plant`` observes; the
    critic sees the plant's rope either way. ``seed`` picks the keys. The
    history holds the mean cost per step by episode."""
    program = whip_training_program(
        WhipMean(features, nn.GRU(MEMORY), HIDDEN), whip_critic(plant), plant, n_episodes,
        actor_optimizer=actor_optimizer, n_worlds=n_worlds,
        n_chunks_per_episode=n_chunks_per_episode, n_steps_per_chunk=n_steps_per_chunk)
    outcome = shac_training(
        program,
        parameter_key=jax.random.fold_in(jax.random.PRNGKey(1), seed),
        training_key=jax.random.fold_in(jax.random.PRNGKey(11), seed),
    )
    # The history arrives per chunk; one training round here is an episode.
    per_episode = outcome.history.mean_cost.reshape(n_episodes, -1).mean(axis=1)
    return Struct(policy=outcome.policy, history=outcome.history.replace(mean_cost=per_episode))


def main() -> None:
    plant = whip_plant().parameterize()
    features = WhipFeatures(tip_sensor=False)
    run = timed(lambda: train_whip_shac(plant, features))
    print('mean cost per step by episode:', run.result.history.mean_cost)
    curve = plot_training_curves(
        {'SHAC': Struct(curve=run.result.history.mean_cost, seconds=run.seconds)},
        'whip_shac_training.png')
    print(f'training curve: {curve}')
    starts = evaluation_starts(plant)
    baseline = untrained_policy(plant, features, jax.random.PRNGKey(2))
    summarize(whip_evaluation(baseline, plant, starts, jax.random.PRNGKey(21)), 'untrained')
    evaluation = whip_evaluation(run.result.policy, plant, starts, jax.random.PRNGKey(22))
    summarize(evaluation, 'SHAC')
    write_artifacts(evaluation, 'whip_shac')


if __name__ == '__main__':
    main()
