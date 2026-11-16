"""Demonstrates third-order meta-learning across a deeply transformed recurrent Node.

Stacks residual `MinGRU` blocks paired with `BatchNorm` and `tanh`, wraps each layer in `next_step_ttt` inside a `tie`-shared encoder and decoder, scans across sequences, adapts over support trajectories via `finetune`, ensembles rematerialized learners via `ensemble(remat(...))`, batches across tasks, and meta-trains every initial weight and inner learning rate end-to-end with `trained(train_step(...))`. Because smooth operations nest inner TTT, support fine-tuning, and outer meta-updates directly, third-order derivatives propagate through the entire composition.

`BatchNorm` inside each block exercises collective `'batch'` running statistics across `stack`, `tie`, `finetune`, `ensemble`, and outer `trained` scans. Its `1 / sqrt(var)` backward gain also bounds sequence length in differentiated inner loops: short streams (`STREAM = 8`) remain well-conditioned, whereas long unnormalized-gradient horizons compound normalization gains across inner updates.

Run directly via `python -m examples.third_order_learning`.
"""

import os

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Leaf,
    Node,
    Struct,
    batch,
    ensemble,
    finetune,
    next_step_ttt,
    node,
    reduce,
    remat,
    residual,
    scanned,
    serial,
    stack,
    taps,
    tie,
    train_step,
    trained,
)
from nodejax import nn
from nodejax.transforms.learning import learned_sgd
from nodejax.core.types import PyTree


SMOKE = os.environ.get('NODEJAX_SMOKE') == '1'

FEATURES = 2
WIDTH = 3
DEPTH = 2
MEMBERS = 2
SUPPORT_SEQUENCES = 1 if SMOKE else 4
STREAM = 3 if SMOKE else 8
TASKS = 2
META_STEPS = 2 if SMOKE else 120
SPECTRAL_RADIUS = 0.9

TTT_RATE = 0.5
FINETUNE_RATE = 0.3
META_RATE = 0.02


@node
def Encoder(features: int, width: int) -> Node:
    """Projects `features`-dimensional observations into `width` features."""
    def param(rng) -> Struct:
        scale = jnp.sqrt(features)
        return Struct(weight=jax.random.normal(rng.next(), (features, width)) / scale)

    def apply(param, input) -> jax.Array:
        return input @ param.weight

    return Leaf(apply, param=param)


@node
def Decoder(features: int, width: int) -> Node:
    """Projects `width` features back to `features` via the transposed matrix shared under `tie`."""
    def param(rng) -> Struct:
        scale = jnp.sqrt(features)
        return Struct(weight=jax.random.normal(rng.next(), (features, width)) / scale)

    def apply(param, input) -> jax.Array:
        return input @ param.weight.T

    return Leaf(apply, param=param)


def token_loss(output: PyTree, target: jax.Array) -> jax.Array:
    return jnp.mean((output - target) ** 2)


def sequence_loss(output: PyTree, target: jax.Array) -> jax.Array:
    return jnp.mean((output[..., 1:, :] - target[..., 1:, :]) ** 2)


def predictor() -> Node:
    block = residual(nn.MinGRU(WIDTH)) >> nn.BatchNorm(momentum=0.1) >> nn.tanh
    adaptive = next_step_ttt(train_step(block, token_loss, learned_sgd(TTT_RATE)))
    recurrent = stack(adaptive, n=DEPTH)
    observed = taps(recurrent >> nn.tanh)
    pipe = Encoder(FEATURES, WIDTH) >> observed >> Decoder(FEATURES, WIDTH)
    return tie(pipe, 'encoder', 'decoder')


def third_order_learning() -> Node:
    online = scanned(predictor())
    adapted = finetune(train_step(online, sequence_loss, learned_sgd(FINETUNE_RATE)))
    committee = ensemble(remat(adapted), n=MEMBERS) >> reduce(jnp.mean)
    outer = train_step(batch(committee), sequence_loss, optax.adam(META_RATE))
    return trained(outer)


# Synthetic input data for the dynamical-system task.


def make_sequence(matrix: jax.Array, initial: jax.Array) -> jax.Array:
    """Generates one trajectory of `state_next = tanh(matrix @ state)`."""
    def step(state: jax.Array, unused: None) -> tuple[jax.Array, jax.Array]:
        next_state = jnp.tanh(matrix @ state)
        return next_state, next_state

    _, following = jax.lax.scan(step, initial, None, length=STREAM - 1)
    return jnp.concatenate((initial[None], following), axis=0)


def make_tasks(rng: jax.Array) -> tuple[Struct, jax.Array]:
    """Samples one batch of random contraction-matrix dynamical tasks per meta step."""
    matrix_rng, support_rng, query_rng = jax.random.split(rng, 3)
    matrices = jax.random.uniform(
        matrix_rng,
        (META_STEPS, TASKS, FEATURES, FEATURES),
        minval=-1.0, maxval=1.0)
    largest = jnp.linalg.norm(matrices, ord=2, axis=(-2, -1))
    matrices = matrices * (SPECTRAL_RADIUS / largest)[..., None, None]
    support_initial = jax.random.uniform(
        support_rng,
        (META_STEPS, TASKS, SUPPORT_SEQUENCES, FEATURES),
        minval=-1.0, maxval=1.0)
    query_initial = jax.random.uniform(
        query_rng, (META_STEPS, TASKS, FEATURES), minval=-1.0, maxval=1.0)

    support = jax.vmap(jax.vmap(jax.vmap(make_sequence, in_axes=(None, 0))))(
        matrices, support_initial)
    query = jax.vmap(jax.vmap(make_sequence))(matrices, query_initial)
    episodes = Struct(
        support=Struct(input=support, target=support),
        query=query,
    )
    return episodes, query


def run() -> Struct:
    episodes, target = make_tasks(jax.random.PRNGKey(1))
    learner = third_order_learning().with_input(
        bundle=Struct(
            support=episodes.support,
            query=episodes.query,
            target=target,
        )
    ).parameterize(rng=jax.random.PRNGKey(0))
    final, aux = jax.jit(learner.apply)(
        support=episodes.support,
        query=episodes.query,
        target=target,
    )
    return Struct(initial=learner, final=final, aux=aux)


def main() -> None:
    result = run()
    leaves = jax.tree.leaves(result.final.param)
    paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(result.final.param)[0]
    }
    decoder_leaves = sum('.decoder.weight' in path for path in paths)
    tapped = [
        leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(result.aux)[0]
        if jax.tree_util.keystr(path).endswith('.tanh')
    ]
    activation_trace, = tapped
    print(result.initial.summary())
    print()

    quarter = max(1, META_STEPS // 4)
    start = float(result.aux.loss[0])
    settled = float(jnp.mean(result.aux.loss[-quarter:]))

    print(f'meta loss over fresh tasks: {start:.6f} -> settled {settled:.6f}')
    print(f'trainable leaves: {len(leaves)}')
    print(f'tied decoder parameter leaves: {decoder_leaves}')
    print(f'tapped activation trace: {activation_trace.shape}')
    print(f'all finite: {bool(jnp.all(jnp.isfinite(result.aux.loss)))}')

    if not SMOKE:
        assert settled < 0.5 * start
    assert decoder_leaves == 0
    assert bool(jnp.all(jnp.isfinite(result.aux.loss)))


if __name__ == '__main__':
    main()
