"""Named axes and tags under batching and ensembling.

A def can declare local tags (e.g. tags={'population'}); batch() and ensemble()
bind vmap axes directly ('batch' / 'ensemble' by convention).
"""

import jax
import jax.numpy as jnp

from nodejax import node_def, batch, ensemble, stack, residual, composite, nn
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


def test_single_batch_state_descends_through_a_transform_wrapper():
    """A tagged member inside a TRANSFORM keeps one state copy, not one per
    sample. The tag is a per-leaf property, so the walk that reads it has to
    descend a transform exactly as it descends a pipe. A transform that
    answered for its whole subtree would hand the tagged member a batch axis
    and then read only element 0, discarding the rest of the batch.

    stack() and ensemble() are the two spellings of depth and population, and
    both are how a real model gets its batchnorm: `stack(Linear >> BatchNorm)`
    is the ordinary deep net, not an exotic construction."""
    x = jnp.arange(6.0).reshape(3, 2)                 # batch of 3, width 2

    for label, tower in [
        ('stack', batch(stack(nn.BatchNorm(0.1) >> nn.RNN(2), n=2))),
        ('ensemble', batch(ensemble(nn.BatchNorm(0.1) >> nn.RNN(2), n=2))),
    ]:
        model = tower.with_input(x).parameterize(rng=jax.random.PRNGKey(0))
        state = model.init()
        assert state.bn.mean.shape == (2, 2), (label, state.bn.mean.shape)
        assert state.rnn.shape == (3, 2, 2), (label, state.rnn.shape)

        # and it holds THROUGH an apply, not only at init: out_axes mirrors
        # the in_axes, so the shared slot comes back shared
        new_state, out = model.apply(state, x)
        assert new_state.bn.mean.shape == (2, 2), label
        assert new_state.rnn.shape == (3, 2, 2), label
        assert out.shape == (3, 2, 2) or out.shape == (3, 2), (label, out.shape)


def test_a_shared_slot_that_is_not_really_shared_is_loud():
    """The tag asserts the member's state is the same for every element.
    jax enforces it: a node that declares a shared state but computes a
    per-element one fails at the vmap, rather than silently keeping one
    element's copy."""
    def liar():
        def init(ndef):
            return jnp.zeros_like(ndef.input)

        def apply(state, input):
            return input, input                # per-element, despite the tag

        return node_def(apply, init=init, name='liar', tags={'single_batch_state'})

    x = jnp.arange(6.0).reshape(3, 2)
    model = batch(liar()).with_input(x).parameterize()
    import pytest
    with pytest.raises(ValueError, match='out_axes is None'):
        model.apply(model.init(), x)
