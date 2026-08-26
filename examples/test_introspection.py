"""Statics, generics, and the introspection utils, on one real tower.

The tree under inspection is four levels of composition with its one
size argument left unbound: batch(scan(stack(Cell()))). Everything this
file demonstrates follows from two facts. The @node record keeps every
factory call, so the whole argument graph is data (.statics_by_path()), a
deferred build prints as the record it is (.describe()), and specialize
re-runs it with any unambiguous spelling of the address. And bindings
are separate trees, so the same describe that writes a def out writes
a bound model's param and state out beside it.

The record paths read as the composition reads: batch's member is the
per-sample program, scan's is the step, stack's is the layer, so the
unbound width lives at sample.step.layer.width, and fills route to it
from any shorter spelling that stays unambiguous.
"""

import jax
import jax.numpy as jnp

from nodejax import (
    REQUIRED, Leaf, Node, Struct, batch, node, scan, stack, train_step,
)
from examples.util import mse


@node
def Cell(width: int) -> Node:
    """One recurrent unit: its size the ONE static, its weights drawn at
    parameterize, its carry starting at rest."""
    def param(rng):
        return Struct(w=0.3 * jax.random.normal(rng.next(), (width, width)))

    def init(node):
        return jnp.zeros(width)

    def apply(param, state, input):
        new = jnp.tanh(state @ param.w + input)
        return new, new

    return Leaf(apply, param=param, init=init)


def committee_tower() -> Node:
    """The tree, in its own scope: a stack of cells makes the column,
    scan runs the column over time, batch runs four sequences side by
    side. The cell's width stays unbound on purpose: the whole tower is
    a generic until that one number arrives."""
    column = stack(Cell(), n=3)
    over_time = scan(column)
    return batch(over_time, n=4)


def test_the_record_is_readable_and_addressable():
    """statics_by_path lists the whole argument graph at full addresses;
    describe prints the deferred record with the unbound argument
    marked; specialize fills it at its listed address."""
    tower = committee_tower()
    assert tower.generic

    listed = tower.statics_by_path()
    assert listed['sample.step.layer.width'] is REQUIRED
    assert listed['sample.step.n'] == 3           # stack's own depth, at home

    written = tower.describe()
    print(written)
    assert 'width=<unbound>' in written
    assert 'generic]' in written                  # flagged generic, every level

    filled = tower.specialize(**{'sample.step.layer.width': 8})
    assert not filled.generic
    assert filled.statics_by_path()['sample.step.layer.width'] == 8


def test_describe_writes_the_bound_model_out():
    """The same entry writes a bound model: the member hierarchy above,
    the param and state trees below, shapes and a leaf tally. The param
    tree carries the layer axis stack added; the carry is per layer and,
    once batched, per sequence."""
    model = committee_tower().specialize(**{'sample.step.layer.width': 8}).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    written = model.describe()
    print()
    print(written)
    assert 'param:' in written and 'state:' in written
    assert 'w = float32(3, 8, 8)' in written      # one weight per layer
    assert model.state.shape == (4, 3, 8)         # per sequence, per layer

    sequences = jnp.ones((4, 20, 8))
    advanced, outputs = model(sequences)
    assert outputs.shape == (4, 20, 8)
    assert not jnp.allclose(advanced.state, model.state)


def test_the_trainer_prints_its_two_members():
    """train_step of the bound model is a two-member composite, and
    describe says so: opt beside model, the weights under both readings
    (param.model is what training starts from, state.opt.params where
    it has got to)."""
    import optax
    model = committee_tower().specialize(**{'sample.step.layer.width': 8}).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, mse, optax.sgd(0.1))
    written = trainer.describe()
    print(written)
    assert 'opt:' in written and 'model:' in written
    assert 'model.w = float32(3, 8, 8)' in written
    assert 'opt.params.w = float32(3, 8, 8)' in written


def test_configuring_by_path():
    """One address vocabulary, all three construction doors, and the
    rules are strict on purpose: a bare key binds in the node's own
    record or refuses (never routing deeper on its own), reaching a
    depth means spelling the path (dot-joined or nested, one call
    either way), and '*.field' is the one broadcast, string form only.
    parameterize and initialize take the same dot-joined spelling for
    their bundles."""
    tower = committee_tower()
    by_path = tower.specialize(**{'sample.step.layer.width': 8})
    nested = tower.specialize(sample={'step': {'layer': {'width': 8}}})
    assert by_path.statics_by_path() == nested.statics_by_path()

    try:
        tower.specialize(width=8)                  # bare: nothing routes
        raise AssertionError('a bare key must not reach a depth')
    except TypeError as refusal:
        assert 'width' in str(refusal)

    wider = by_path.specialize(**{'*.width': 16})
    assert wider.statics_by_path()['sample.step.layer.width'] == 16

    from nodejax.control import Gain
    pipe = Gain() >> Gain()
    flat = pipe.parameterize(**{'gain.scale': 2.0, 'gain_2.scale': 3.0})
    nested = pipe.parameterize(gain=Struct(scale=2.0),
                               gain_2=Struct(scale=3.0))
    assert flat.param.gain.scale == nested.param.gain.scale == 2.0
    assert flat.param.gain_2.scale == nested.param.gain_2.scale == 3.0
