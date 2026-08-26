"""Param construction with rng: constructors are plain callables — the
original tree(param) form. A constructor declaring rng receives a
KeyStream (rng.next() per draw, no split bookkeeping), and one boundary
key routes down every path params flow: per pipe member, per ensemble
member / stack layer. Explicit values need no key at all: PID gains from
values, network weights from keys, one syntax.
"""

import jax
import pytest
import jax.numpy as jnp

from nodejax import Node, Leaf, ensemble, stack, KeyStream
from nodejax.struct import Struct
from nodejax.control import Gain


def Linear(n_in: int, n_out: int) -> Node:
    def param(rng, weight=None, bias=None):
        if weight is None:
            weight = jax.random.normal(rng.next(), (n_in, n_out))
        return Struct(weight=weight,
                      bias=jnp.zeros(n_out) if bias is None else bias)

    def apply(param, input):
        return input @ param.weight + param.bias

    return Leaf(apply, param=param, name='linear')


def test_keystream_yields_distinct_replayable_keys():
    keys = KeyStream(jax.random.PRNGKey(0))
    k1, k2 = keys.next(), keys.next()
    assert jnp.any(k1 != k2)
    again = KeyStream(jax.random.PRNGKey(0))
    assert jnp.all(again.next() == k1) and jnp.all(again.next() == k2)


def test_constructor_draws_from_keystream():
    lin = Linear(2, 3)
    a = lin.parameterize(rng=jax.random.PRNGKey(0))
    b = lin.parameterize(rng=jax.random.PRNGKey(0))
    c = lin.parameterize(rng=jax.random.PRNGKey(1))

    assert a.param.weight.shape == (2, 3)
    assert jnp.allclose(a.param.bias, 0.0)                  # constructor default
    assert jnp.allclose(a.param.weight, b.param.weight)     # same key, same draw
    assert not jnp.allclose(a.param.weight, c.param.weight)


def test_explicit_values_need_no_rng():
    """Finished values enter through bind, the contract door — no key in
    sight. The CONSTRUCTOR always takes its key (rng never has a default),
    so parameterize without one is loud."""
    node = Linear(2, 3).bind(Struct(weight=jnp.ones((2, 3)), bias=jnp.ones(3)))
    assert jnp.allclose(node.apply(jnp.ones(2)), 3.0)
    with pytest.raises(TypeError, match='parameterize requires rng'):
        Linear(2, 3).parameterize(weight=jnp.ones((2, 3)), bias=jnp.ones(3))


def test_overrides_mix_with_draws():
    node = Linear(2, 3).parameterize(rng=jax.random.PRNGKey(0), bias=jnp.ones(3))
    assert jnp.allclose(node.param.bias, 1.0)
    assert node.param.weight.shape == (2, 3)


def test_pipe_splits_param_rng_per_member():
    pipe = Linear(2, 2) >> Linear(2, 2)
    bound = pipe.parameterize(rng=jax.random.PRNGKey(0))

    assert not jnp.allclose(bound.param.linear.weight,
                            bound.param.linear_2.weight)    # split, not copied

    # nested overrides merge over the drawn defaults; other members' draws
    # are unaffected (keys split per member, in member order)
    tweaked = pipe.parameterize(rng=jax.random.PRNGKey(0),
                                linear_2=Struct(bias=jnp.ones(2)))
    assert jnp.allclose(tweaked.param.linear_2.bias, 1.0)
    assert jnp.allclose(tweaked.param.linear.weight, bound.param.linear.weight)


def test_pipe_mixes_plain_constructors():
    """Members whose constructors cannot consume a key are simply not
    handed one; the rest draw from the split."""
    pipe = Gain() >> Linear(2, 2)
    bound = pipe.parameterize(rng=jax.random.PRNGKey(0), gain=Struct(scale=2.0))

    assert jnp.allclose(bound.param.gain.scale, 2.0)
    assert 'rng' not in bound.param.gain.__keys__            # reserved, not packed
    assert bound.param.linear.weight.shape == (2, 2)


def test_ensemble_and_stack_draw_n_members():
    """Declared size + one key = n independent draws, stacked — the
    hand-built stacked-params boilerplate, deleted."""
    e = ensemble(Linear(2, 2), n=3).parameterize(rng=jax.random.PRNGKey(0))
    assert e.param.weight.shape == (3, 2, 2)
    assert not jnp.allclose(e.param.weight[0], e.param.weight[1])

    s = stack(Linear(2, 2), n=4).parameterize(rng=jax.random.PRNGKey(0))
    assert s.param.weight.shape == (4, 2, 2)

    # explicit stacked params remain a first-class path — through bind
    manual = ensemble(Linear(2, 2), n=3).bind(
        Struct(weight=jnp.ones((3, 2, 2)), bias=jnp.zeros((3, 2))))
    assert jnp.allclose(manual.apply(jnp.ones(2)), 2.0)
