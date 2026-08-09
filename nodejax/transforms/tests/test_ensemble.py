"""ensemble over a shape-resolved pipe.

Regression: the non-cyclic init path handed the pipe's init walk the
STACKED member params, which broadcast against an unstacked carry. The
(empty) state builds through one member's row, exactly as stack does.
"""
import jax

from nodejax import ensemble, nn


def test_ensemble_of_resolved_pipe():
    X = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
    net = (nn.Linear(8) >> nn.relu >> nn.Linear(1)).with_input(X)

    population = ensemble(net, n=4).parameterize(rng=jax.random.PRNGKey(1))
    assert population.param.linear.w.shape == (4, 4, 8)
    assert population.apply(X).shape == (4, 32, 1)
