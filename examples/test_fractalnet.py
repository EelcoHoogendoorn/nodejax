"""FractalNet depth expansion (Larsson et al., 2017) built by recursive mean joins.

    f_1     = block
    f_(C+1) = join(block, f_C >> f_C)

Gradient paths of every length from `1` to `2^(C-1)` blocks coexist in one `Composite` without
residual connections, regularized by local drop-path. `Join` feeds its input to `shallow` and
`deep` members and averages the active branches, muting at most one fair-coin-selected branch per
example during training. Because `fractal` accepts any shape-preserving `BaseNode`, the same
recursive expansion runs over convolutional, dense, recurrent, or attention blocks.

Cutting the recursion to `fractalnet(levels)` produces shallower evaluators that bind directly
against the full trained parameter and state records. Wrapping the `levels == 1` base case under
`shallow` aligns every member path of a shallower column with a prefix of a deeper one, so extra
members are ignored during binding while retained joins match sub-networks visited under drop-path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import BaseNode, Composite, Node, PNode, PyTree, batch, node, nn, serial, train_step, trained
from examples.test_conv_vit import data, xent

WIDTH = 12
LEVELS = 3
MOMENTUM = 0.3
DROP_RATE = 0.15
BATCH, STEPS = 32, 3


@node
def Join(shallow: BaseNode, deep: BaseNode, drop_rate: float,
         train: bool = True) -> Node | PNode:
    """Averages `shallow` and `deep` member outputs with local single-branch drop-path in training.

    When `train=True`, mutes one coin-selected branch with probability `drop_rate` while still
    executing both members to advance their state; when `train=False`, returns the unweighted mean.
    """
    if not train:
        def merge(self, input) -> jax.Array:
            return (self.shallow(input) + self.deep(input)) / 2
    else:
        def merge(self, input, rng) -> jax.Array:
            drop = jax.random.bernoulli(rng.next(), drop_rate).astype(input.dtype)
            toss = jax.random.bernoulli(rng.next(), 0.5).astype(input.dtype)
            return ((1.0 - drop * toss) * self.shallow(input)
                    + (1.0 - drop * (1.0 - toss)) * self.deep(input)) / (2.0 - drop)

    return Composite(shallow=shallow, deep=deep)(merge)


def fractal(levels: int, block: BaseNode, drop_rate: float) -> Node | PNode:
    """Expands `block` recursively into a `levels`-deep FractalNet column.

    Reusing `block` allocates independent parameter and state slots at each member path, and
    wrapping the `levels == 1` base case under `shallow` aligns shallower member paths with deeper
    records.
    """
    if levels == 1:
        return serial(shallow=block)
    deep = serial(
        first=fractal(levels - 1, block, drop_rate),
        second=fractal(levels - 1, block, drop_rate),
    )
    return Join(block, deep, drop_rate)


def fractalnet(levels: int = LEVELS, drop_rate: float = DROP_RATE) -> Node:
    """Builds a per-sample `levels`-deep convolutional FractalNet classifier for `8x8` digits."""
    block = nn.Conv(WIDTH) >> nn.BatchNorm(MOMENTUM, axis=('batch', 0, 1)) >> nn.relu
    return serial(
        image=nn.Reshape((8, 8, 1)),
        fractal=fractal(levels, block, drop_rate),
        flat=nn.flat,
        head=nn.Linear(10),
    )


def test_fractal_expansion_counts_blocks() -> None:
    """Verifies that a depth-`LEVELS` FractalNet column instantiates `2**LEVELS - 1` conv blocks."""
    model = batch(fractalnet()).with_input(
        jnp.zeros((BATCH, 64))).parameterize(rng=jax.random.PRNGKey(0))
    kernels = [
        path
        for path, _ in jax.tree_util.tree_flatten_with_path(model.param)[0]
        if jax.tree_util.keystr(path).endswith('.kernel')
    ]
    assert len(kernels) == 2 ** LEVELS - 1


def test_fractal_is_generic_over_its_block() -> None:
    """Verifies recursive FractalNet expansion and sub-depth slicing across dense, GRU, and attention blocks."""
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

        def evaluated(tree: BaseNode, weights: PyTree) -> jax.Array:
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
    """Trains FractalNet on digits and evaluates shallower sub-columns bound to the trained record."""
    pipe = batch(fractalnet())
    model = pipe.with_input(jnp.zeros((BATCH, 64))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, xent, optax.adam(1e-2))

    X_train, y_train, X_test, _ = data()
    shuffle = np.random.RandomState(1)
    batch_indices = shuffle.randint(0, len(X_train), size=(STEPS, BATCH))

    final, aux = trained(trainer).apply(input=X_train[batch_indices],
                                        target=y_train[batch_indices],
                                        rng=jax.random.PRNGKey(1))
    assert aux.loss.shape == (STEPS,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, model.param))

    # Distinct PRNG keys draw distinct drop-path masks on the training-mode `PSNode`.
    _, logits_a = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(2))
    _, logits_b = final(input=X_train[:BATCH], rng=jax.random.PRNGKey(3))
    assert not jnp.allclose(logits_a, logits_b)

    evaluator = nn.eval_mode(final)
    assert not evaluator.cyclic
    X_eval = X_test[:BATCH]
    _, logits = evaluator(X_eval)
    assert jnp.allclose(logits, evaluator(X_eval)[1])

    for levels in range(1, LEVELS + 1):
        budget = batch(fractalnet(levels)).bind(
            final.param, state=final.state)
        _, budget_logits = nn.eval_mode(budget)(X_eval)
        assert budget_logits.shape == (BATCH, 10)
        assert jnp.all(jnp.isfinite(budget_logits))
        if levels == LEVELS:
            assert jnp.allclose(budget_logits, logits)
        else:
            assert not jnp.allclose(budget_logits, logits)
