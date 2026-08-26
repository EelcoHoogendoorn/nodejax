"""Third-order learning across a deeply transformed recurrent Node.

Each stacked residual minGRU has its own running batch statistics and performs test-time training
inside a tied encoder and readout. The cell's state transition is contractive by construction, so
gradient transport along the sequence stays bounded at every derivative order; the tanh after each
cell's norm is what keeps genuine curvature in the differentiated path. The stack is scanned over a
sequence and fine-tuned over support sequences. An ensemble of those rematerialized learners adapts
independently before its mean is evaluated across batched tasks. An outer trainer meta-learns every
starting value and learning rate on a freshly sampled batch of tasks each step, and a task is
observable only through its support trajectories, so adaptation is the only route to the query.

Each level is an ordinary Node transform. The inner, support and query gradients are nested exactly,
so smooth model operations contribute genuine third-order derivatives to the outer update.

The batch norm in each block earns nothing as modeling; it is here as a flex of composition. Its
running statistics are ordinary cyclic state threading stack, tie, finetune, ensemble and the outer
training scan, with the batch axis reaching it by name through every level, and that is the point
being made. It also marks the scaling limit of this construction: normalization's backward gain
grows as one over the activation scale, and inside a differentiated test-time-training loop those
gains compound. At this stream length the few crossings stay tame; pushed to stream length 128 the
finetune landscape grows cliffs (a mild gradient at the initialization, of order 1e9 one small
update away) and third-order training only stabilizes with the norm out of the differentiated path.

Run directly: ``python -m examples.third_order_learning``
"""

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
    split_aux,
    stack,
    taps,
    tie,
    train_step,
    trained,
)
from nodejax import nn
from nodejax.transforms.learning import learned_sgd
from nodejax.core.types import PyTree


FEATURES = 2
WIDTH = 3
DEPTH = 2
MEMBERS = 2
SUPPORT_SEQUENCES = 4
STREAM = 8
TASKS = 2
META_STEPS = 120
SPECTRAL_RADIUS = 0.9

TTT_RATE = 0.5
FINETUNE_RATE = 0.3
META_RATE = 0.02


@node
def Encoder(features: int, width: int) -> Node:
    """Map an observation into the recurrent width using one shared matrix."""
    def param(rng):
        scale = jnp.sqrt(features)
        return Struct(weight=jax.random.normal(rng.next(), (features, width)) / scale)

    def apply(param, input):
        return input @ param.weight

    return Leaf(apply, param=param)


@node
def Decoder(features: int, width: int) -> Node:
    """Map back with the transpose of the encoder matrix supplied by ``tie``."""
    def param(rng):
        scale = jnp.sqrt(features)
        return Struct(weight=jax.random.normal(rng.next(), (features, width)) / scale)

    def apply(param, input):
        return input @ param.weight.T

    return Leaf(apply, param=param)


def token_loss(output: PyTree, target: jax.Array) -> jax.Array:
    prediction, _ = split_aux(output)
    return jnp.mean((prediction - target) ** 2)


def sequence_loss(output: PyTree, target: jax.Array) -> jax.Array:
    prediction, _ = split_aux(output)
    return jnp.mean((prediction[..., 1:, :] - target[..., 1:, :]) ** 2)


def predictor() -> Node:
    block = residual(nn.MinGRU(WIDTH)) >> nn.BatchNorm(momentum=0.1) >> nn.tanh
    adaptive = next_step_ttt(
        train_step(
            block,
            token_loss,
            learned_sgd(TTT_RATE),
        )
    )
    recurrent = stack(adaptive, n=DEPTH)
    observed = taps(serial(recurrent=recurrent, activation=nn.tanh))
    pipe = serial(
        encoder=Encoder(FEATURES, WIDTH),
        recurrent=observed,
        decoder=Decoder(FEATURES, WIDTH),
    )
    return tie(pipe, 'encoder', 'decoder')


def third_order_learning() -> Node:
    online = scanned(predictor())
    adapted = finetune(
        train_step(online, sequence_loss, learned_sgd(FINETUNE_RATE)))
    committee = ensemble(remat(adapted), n=MEMBERS) >> reduce(jnp.mean)
    outer = train_step(
        batch(committee), sequence_loss, optax.adam(META_RATE))
    return trained(outer)


# Synthetic input data for the dynamical-system task. Nothing below builds Nodes.


def make_sequence(matrix: jax.Array, initial: jax.Array) -> jax.Array:
    """Generate one trajectory of ``state = tanh(matrix @ state)``."""
    def step(state, unused: None) -> tuple[jax.Array, jax.Array]:
        next_state = jnp.tanh(matrix @ state)
        return next_state, next_state

    _, following = jax.lax.scan(step, initial, None, length=STREAM - 1)
    return jnp.concatenate((initial[None], following), axis=0)


def make_tasks(rng: jax.Array) -> tuple[Struct, jax.Array]:
    """Sample one batch of dynamical-system tasks per meta step.

    A task is a random matrix rescaled to a fixed spectral radius. Its
    identity is observable only through the support trajectories."""
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
        bundle=Struct(input=episodes, target=target)
    ).parameterize(rng=jax.random.PRNGKey(0))
    final, aux = jax.jit(learner.apply)(input=episodes, target=target)
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
        if jax.tree_util.keystr(path).endswith('.recurrent.activation')
    ]
    activation_trace, = tapped
    print(result.initial.summary())
    print()

    quarter = META_STEPS // 4
    start = float(result.aux.loss[0])
    settled = float(jnp.mean(result.aux.loss[-quarter:]))

    print(f'meta loss over fresh tasks: {start:.6f} -> settled {settled:.6f}')
    print(f'trainable leaves: {len(leaves)}')
    print(f'tied decoder parameter leaves: {decoder_leaves}')
    print(f'tapped activation trace: {activation_trace.shape}')
    print(f'all finite: {bool(jnp.all(jnp.isfinite(result.aux.loss)))}')

    assert settled < 0.5 * start
    assert decoder_leaves == 0
    assert bool(jnp.all(jnp.isfinite(result.aux.loss)))


if __name__ == '__main__':
    main()
