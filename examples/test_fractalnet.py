"""FractalNet: depth grown by a recursion, joined by means, no residuals.

    f_1     = block
    f_(C+1) = join(block, f_C >> f_C)

(Larsson et al. 2017.) Gradient paths of every length from one block to
2^(C-1) coexist in one tree: deep credit assignment without residual
connections, regularized by local drop-path. The join is one custom
composite node: it feeds its input to a shallow and a deep member and
merges the mean of the live ones, muting at most one coin-chosen member
per example in training (a simpler local drop-path than the paper's
independent drops, and without its global column sampling). The
expansion rule takes a repeatable, shape-compatible base Node, so the
same recursion runs over conv blocks, dense blocks, attention layers,
or recurrent cells unchanged.

The trained net is test-time scalable by cutting the recursion: the
budget evaluator is ``fractalnet(levels)`` rebuilt smaller and bound
against the full trained record directly. The base case nests its block
under the same ``shallow`` name a join uses, so every path of a smaller
fractal exists verbatim in a bigger one's params and state, and binding
selects what the smaller definition names while the extra members fall
away. Every retained join survives the cut, so each smaller fractal is
a drop-path configuration visited during training and remains useful on
digits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import Composite, Node, batch, node, nn, serial, train_step, trained
from examples.test_conv_vit import accuracy, data, xent

WIDTH = 12
LEVELS = 3
MOMENTUM = 0.3
DROP_RATE = 0.15
BATCH, EPOCHS = 125, 25


@node
def Join(shallow: Node, deep: Node, drop_rate: float,
         train: bool = True) -> Node:
    """The fractal join: the drop-path-weighted mean of its members.

    Training mutes at most one member's output per example: with
    probability ``drop_rate`` a coin-chosen member is dropped and the
    survivor passes whole. Both members still execute and advance their
    state. Evaluation is the plain mean."""
    if not train:
        def merge(self, input):
            return (self.shallow(input) + self.deep(input)) / 2
    else:
        def merge(self, input, rng):
            drop = jax.random.bernoulli(rng.next(), drop_rate).astype(input.dtype)
            toss = jax.random.bernoulli(rng.next(), 0.5).astype(input.dtype)
            return ((1.0 - drop * toss) * self.shallow(input)
                    + (1.0 - drop * (1.0 - toss)) * self.deep(input)) / (2.0 - drop)

    return Composite(shallow=shallow, deep=deep)(merge)


def fractal(levels: int, block: Node, drop_rate: float) -> Node:
    """The expansion rule over a repeatable, shape-compatible base block:
    one structural placement of ``block`` joined with two half-depth
    fractals in series. Reusing the Node creates independent param and state
    slots at every member path. The block must accept its own output, and the
    shallow and deep results must have matching pytrees and shapes. The base
    case nests its block under the same ``shallow`` name a join gives its
    block, so every param and state path of a smaller fractal exists verbatim
    in a bigger one's records."""
    if levels == 1:
        return serial(shallow=block)
    deep = serial(
        first=fractal(levels - 1, block, drop_rate),
        second=fractal(levels - 1, block, drop_rate),
    )
    return Join(block, deep, drop_rate)


def fractalnet(levels: int = LEVELS, drop_rate: float = DROP_RATE) -> Node:
    """(64,) pixels -> (10,) logits: one fractal stage over the image."""
    block = nn.Conv(WIDTH) >> nn.BatchNorm(MOMENTUM, axis=('batch', 0, 1)) >> nn.relu
    return serial(
        image=nn.Reshape((8, 8, 1)),
        fractal=fractal(levels, block, drop_rate),
        flat=nn.flat,
        head=nn.Linear(10),
    )


def test_fractal_expansion_counts_blocks() -> None:
    """f_C holds 2^C - 1 blocks: the deepest path doubles per level while
    the shallow path stays a single block."""
    model = batch(fractalnet()).with_input(
        jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))
    kernels = [
        path
        for path, _ in jax.tree_util.tree_flatten_with_path(model.param)[0]
        if jax.tree_util.keystr(path).endswith('.kernel')
    ]
    assert len(kernels) == 2 ** LEVELS - 1


def test_fractal_is_generic_over_its_block() -> None:
    """The same recursion over dense, recurrent, and attention blocks:
    builds, counts, runs, and cuts to smaller fractals, with no conv
    anywhere."""
    vector = jnp.linspace(-1.0, 1.0, 6)
    sequence = jnp.arange(24.0).reshape(4, 6) / 24.0

    for label, block, marker, sample in (
            ('dense', nn.Linear(6) >> nn.gelu, '.w', vector),
            ('recurrent', nn.GRU(6), '.update_bias', vector),
            ('attention', nn.Attention(heads=2), '.wqkv', sequence)):
        model = fractal(LEVELS, block, DROP_RATE).with_input(
            sample).parameterize(rng=jax.random.PRNGKey(0))
        blocks = [
            path
            for path, _ in jax.tree_util.tree_flatten_with_path(model.param)[0]
            if jax.tree_util.keystr(path).endswith(marker)
        ]
        assert len(blocks) == 2 ** LEVELS - 1, label

        def evaluated(tree, weights) -> jax.Array:
            built = tree.specialize(**{'*.train': False}).with_input(
                sample).bind(weights)
            if built.cyclic:
                built = built.initialize(input=sample)
                _, output = built(sample)
                return output
            return built(sample)

        full = evaluated(model, model.param)
        cut = evaluated(fractal(2, block, DROP_RATE), model.param)
        single = evaluated(fractal(1, block, DROP_RATE), model.param)
        assert full.shape == cut.shape == single.shape == sample.shape
        assert not jnp.allclose(full, cut), label
        assert not jnp.allclose(full, single), label
        assert not jnp.allclose(cut, single), label


def test_fractalnet_trains_and_evaluates_subfractals() -> None:
    """Train on real digits through drop-path and running statistics, then
    evaluate at every depth by rebuilding the recursion smaller and binding
    the full trained record directly."""
    pipe = batch(fractalnet())
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, xent, optax.adam(1e-2))

    X_train, y_train, X_test, y_test = data()
    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)

    final, aux = trained(trainer).apply(input=X_train[batch_indices],
                                        target=y_train[batch_indices],
                                        rng=jax.random.PRNGKey(1))
    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.1 * aux.loss[0]

    # drop-path is alive in the train build: different keys, different masks
    _, logits_a = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(2))
    _, logits_b = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(3))
    assert not jnp.allclose(logits_a, logits_b)

    evaluator = nn.eval_mode(final)
    assert not evaluator.cyclic
    _, logits = evaluator(X_test)
    assert jnp.allclose(logits, evaluator(X_test)[1])
    scores = {'full': accuracy(logits, y_test)}

    for levels in range(1, LEVELS + 1):
        budget = batch(fractalnet(levels)).bind(
            final.param, state=final.state)
        _, budget_logits = nn.eval_mode(budget)(X_test)
        scores[levels] = accuracy(budget_logits, y_test)
        if levels == LEVELS:
            assert jnp.allclose(budget_logits, logits)

    assert scores['full'] > 0.9, scores
    assert all(scores[levels] > 0.7 for levels in range(1, LEVELS)), scores

    print(f"\n[fractalnet] blocks {2 ** LEVELS - 1} | "
          f"loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} | "
          f"acc full {scores['full']:.3f} | "
          + ' | '.join(f'level {levels}: {scores[levels]:.3f}'
                       for levels in range(1, LEVELS + 1)))
