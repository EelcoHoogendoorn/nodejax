"""repeat: weight-tied depth — one param, applied n times in sequence."""

import jax.numpy as jnp

from nodejax import repeat
from nodejax.examples import gain_def, integrator_def


def test_repeat():
    """One param applied n times — vs stack's per-layer params — and
    rebinding a bound Node (param meaning unchanged)."""
    r = repeat(gain_def(), n=3).parameterize(scale=jnp.array(2.0))
    assert jnp.allclose(r.apply(1.0), 8.0)
    assert r.param.scale.shape == ()          # no layer axis, unlike stack

    bound = gain_def().parameterize(scale=jnp.array(3.0))
    assert jnp.allclose(repeat(bound, n=2).apply(1.0), 9.0)


def test_repeat_cyclic():
    """Tied weights, untied state: each position keeps its own state slot."""
    r = repeat(integrator_def(), n=2).parameterize(gain=jnp.array(1.0))
    state = r.init()
    assert state.shape == (2,)
    state, out = r.apply(state, 1.0)
    state, out = r.apply(state, 1.0)
    # position 0 integrates the input, position 1 integrates position 0's
    # output: [1, 1] -> [2, 3]
    assert jnp.allclose(state, jnp.array([2.0, 3.0]))
    assert jnp.allclose(out, 3.0)
