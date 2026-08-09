"""Named axes and tags under batching and ensembling.

A def can declare local tags (e.g. tags={'population'}); batch() and ensemble()
bind vmap axes directly ('batch' / 'ensemble' by convention).
"""

import jax
import jax.numpy as jnp

from nodejax import node_def, batch, ensemble, residual, composite, nn
from nodejax.struct import Struct


def center():
    """The minimal consumer: subtract the batch mean."""
    def apply(input):
        return input - jax.lax.pmean(input, 'batch')
    return node_def(apply, name='center', tags={'single_batch_state'})


def test_tags_are_local_to_nodes():
    assert center().ndef.tags == frozenset({'single_batch_state'})

    b = batch(center().ndef)
    x = jnp.asarray([1.0, 2.0, 6.0])
    out = b.bind(()).apply(x)
    assert jnp.allclose(out, x - 3.0)          # the collective saw the batch


def test_composition_and_wrappers_preserve_node_identity():
    gain = node_def(lambda param, input: param.g * input,
                    param=lambda g=2.0: Struct(g=g), name='gain')
    pipe = gain >> center()

    def wapply(self, input):
        return self.inner(input)
    comp = composite(wapply, members=dict(inner=pipe), name='rig')

    node = batch(comp).parameterize()
    x = jnp.asarray([1.0, 2.0, 6.0])
    assert jnp.allclose(node.apply(x), 2.0 * x - 6.0)


def test_ensemble_binds_its_own_name():
    def Spread():
        def apply(param, input):
            y = param.w * input
            return y - jax.lax.pmean(y, 'ensemble')      # population-centered
        return node_def(apply, param=lambda rng: Struct(w=jax.random.normal(rng.next())),
                        name='spread')

    e = ensemble(Spread(), n=4)
    node = e.parameterize(rng=jax.random.PRNGKey(0))
    out = node.apply(jnp.asarray(1.0))
    assert out.shape == (4,)
    assert jnp.allclose(jnp.mean(out), 0.0, atol=1e-6)   # centered across members


def test_batch_norm_is_one_term_in_a_batch_agnostic_pipe():
    """nn.BatchNorm drops into a per-sample pipe and computes true cross-batch moments under batch()."""
    pipe = nn.Linear(4) >> nn.BatchNorm(momentum=1.0)   # momentum 1: state = batch moments
    assert 'single_batch_state' in pipe.members['bn'].tags

    x = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    model = batch(pipe).with_input(x).parameterize(rng=jax.random.PRNGKey(0))
    state = model.init()
    state, out = model.apply(state, x)

    # With population tag, state is a single shared 1D array (C,), not tiled (B, C)!
    assert state.bn.mean.shape == (4,)
    # Output has zero mean and unit variance per feature across the batch
    _, out2 = model.apply(state, x)
    assert jnp.allclose(jnp.mean(out2, axis=0), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.std(out2, axis=0), 1.0, atol=1e-2)
