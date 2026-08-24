"""Sanity tests for tree_view, summary, and print utilities."""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    node, Leaf, Node, PNode, PSNode, Struct,
    stack, ensemble, train_step,
    summary, tree_view, print_tree, print_summary
)
from nodejax.examples.util import mse


@node
def SimpleLinear(features: int) -> Node:
    def param(node, rng):
        d = node.input.shape[-1]
        return Struct(
            w=jax.random.normal(rng.next(), (d, features)),
            b=jnp.zeros(features)
        )

    def apply(param, input):
        return input @ param.w + param.b

    return Leaf(apply, param=param)


@node
def StatefulCell(width: int) -> Node:
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (width, width)))

    def init():
        return jnp.zeros(width)

    def apply(param, state, input):
        h = jnp.tanh(state @ param.w + input)
        return h, h

    return Leaf(apply, param=param, init=init)


def test_tree_view_and_summary_on_generic_node():
    """tree_view and summary work on unbound generic nodes."""
    tower = SimpleLinear() >> stack(StatefulCell(), n=3)
    assert tower.generic

    tv = tree_view(tower)
    print("\n--- Generic Node Tree View ---\n", tv)
    assert isinstance(tv, str) and len(tv) > 0

    sm = summary(tower)
    print("\n--- Generic Node Summary ---\n", sm)
    assert isinstance(sm, str) and len(sm) > 0

    assert tower.tree_view() == tv
    assert tower.summary() == sm


def test_tree_view_and_summary_on_bound_pnode():
    """tree_view and summary work on parameterized PNodes."""
    model = (SimpleLinear(16) >> SimpleLinear(8)).with_input(jnp.zeros((32, 10))).parameterize(
        rng=jax.random.PRNGKey(0)
    )
    assert isinstance(model, PNode)

    tv = tree_view(model)
    print("\n--- PNode Tree View ---\n", tv)
    assert isinstance(tv, str) and len(tv) > 0

    sm = summary(model)
    print("\n--- PNode Summary ---\n", sm)
    assert isinstance(sm, str) and len(sm) > 0

    assert model.tree_view() == tv
    assert model.summary() == sm


def test_tree_view_and_summary_on_bound_psnode():
    """tree_view and summary work on state-bound PSNodes."""
    cell = StatefulCell(16).with_input(jnp.zeros(16)).parameterize(
        rng=jax.random.PRNGKey(1)
    ).initialize()
    assert isinstance(cell, PSNode)

    tv = tree_view(cell)
    print("\n--- PSNode Tree View ---\n", tv)
    assert isinstance(tv, str) and len(tv) > 0

    sm = summary(cell)
    print("\n--- PSNode Summary ---\n", sm)
    assert isinstance(sm, str) and len(sm) > 0

    assert cell.tree_view() == tv
    assert cell.summary() == sm


def test_summary_with_max_depth():
    """max_depth truncation parameter works without error."""
    deep_tree = ensemble(SimpleLinear(16) >> stack(StatefulCell(16), n=4), n=2)
    model = deep_tree.with_input(jnp.zeros((8, 16))).parameterize(
        rng=jax.random.PRNGKey(0)
    ).initialize()

    sm = summary(model, max_depth=1)
    print("\n--- Truncated Summary (depth=1) ---\n", sm)
    assert isinstance(sm, str) and len(sm) > 0

    tv = tree_view(model, max_depth=1)
    assert isinstance(tv, str) and len(tv) > 0


def test_summary_on_composite_trainer():
    """summary works on composite training pipelines."""
    model = SimpleLinear(4).with_input(jnp.zeros((2, 8))).parameterize(
        rng=jax.random.PRNGKey(0)
    )
    trainer = train_step(model, mse, optax.adam(0.01)).initialize()
    assert isinstance(trainer, PSNode)

    sm = summary(trainer)
    print("\n--- Trainer Summary ---\n", sm)
    assert isinstance(sm, str) and len(sm) > 0
