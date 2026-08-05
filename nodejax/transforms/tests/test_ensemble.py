"""ensemble over a shape-resolved pipe.

Regression: the non-cyclic init path handed the pipe's init walk the
STACKED member params, which broadcast against an unstacked carry. The
(empty) state builds through one member's row, exactly as stack does.
"""
import jax
import jax.numpy as jnp

from nodejax import node_def, ensemble
from nodejax.struct import Struct


def linear_def(n_out):
    def param(ndef, rng):
        n_in = ndef.apply_input_spec.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)),
                      b=jnp.zeros(n_out))
    def apply(param, input):
        return input @ param.w + param.b
    return node_def(apply, param=param, name='linear')


def test_ensemble_of_resolved_pipe():
    relu = node_def(lambda input: jnp.maximum(input, 0.0), name='relu')
    X = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
    net = (linear_def(8) >> relu >> linear_def(1)).with_input(X)

    population = ensemble(net, n=4).parameterize(rng=jax.random.PRNGKey(1))
    assert population.param.linear.w.shape == (4, 4, 8)
    assert population.apply(X).shape == (4, 32, 1)
