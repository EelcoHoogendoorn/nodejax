"""Construction-record inspection, `describe()` formatting, and path-addressed specialization.

Leaving `width` unbound in `batch(scan(stack(Cell(), n=3)), n=4)` defers instantiation and records
the full static argument graph as a `Generic` blueprint. `statics_by_path()` lists every static
argument under its dotted member address (`sample.step.layer.width`), `describe()` formats both
deferred `Generic` blueprints and bound `PSNode` hierarchies alongside parameter and state shapes,
and `specialize`, `parameterize`, and `initialize` resolve explicit dotted paths, nested mappings,
or `'*.field'` wildcards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import (
    Generic, REQUIRED, Leaf, Node, Struct, batch, node, scan, stack, train_step,
)
from examples.util import mse


@node
def Cell(width: int) -> Node:
    """Steps a `width`-unit `tanh` recurrent cell from a zero initial state vector."""
    def param(rng) -> Struct:
        return Struct(w=0.3 * jax.random.normal(rng.next(), (width, width)))

    def init(node) -> jax.Array:
        return jnp.zeros(width)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        new = jnp.tanh(state @ param.w + input)
        return new, new

    return Leaf(apply, param=param, init=init)


def committee_tower() -> Generic:
    """Builds a deferred `batch(scan(stack(Cell(), n=3)), n=4)` blueprint with `width` unbound."""
    column = stack(Cell(), n=3)
    over_time = scan(column)
    return batch(over_time, n=4)


def test_the_record_is_readable_and_addressable() -> None:
    """Verifies that `statics_by_path`, `describe`, and `specialize` inspect and bind `width`."""
    tower = committee_tower()
    assert tower.generic

    listed = tower.statics_by_path()
    assert listed['sample.step.layer.width'] is REQUIRED
    assert listed['sample.step.n'] == 3

    written = tower.describe()
    print(written)
    assert 'width=<unbound>' in written
    assert 'generic]' in written

    filled = tower.specialize(**{'sample.step.layer.width': 8})
    assert not filled.generic
    assert filled.statics_by_path()['sample.step.layer.width'] == 8


def test_describe_writes_the_bound_model_out() -> None:
    """Verifies that `describe` formats member hierarchy, parameter shapes, and batched per-layer state."""
    model = committee_tower().specialize(**{'sample.step.layer.width': 8}).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    written = model.describe()
    print()
    print(written)
    assert 'param:' in written and 'state:' in written
    assert 'w = float32(3, 8, 8)' in written
    assert model.state.shape == (4, 3, 8)

    sequences = jnp.ones((4, 20, 8))
    advanced, outputs = model(sequences)
    assert outputs.shape == (4, 20, 8)
    assert not jnp.allclose(advanced.state, model.state)


def test_the_trainer_prints_its_two_members() -> None:
    """Verifies that `describe` on `train_step` displays both `objective` and `opt` parameter paths."""
    import optax
    model = committee_tower().specialize(**{'sample.step.layer.width': 8}).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, mse, optax.sgd(0.1))
    written = trainer.describe()
    print(written)
    assert 'objective:' in written and 'opt:' in written
    assert 'objective.model.w = float32(3, 8, 8)' in written
    assert 'opt.params.model.w = float32(3, 8, 8)' in written


def test_configuring_by_path() -> None:
    """Verifies strict path addressing across `specialize` and `parameterize` with dotted keys, nested mappings, and wildcards."""
    tower = committee_tower()
    by_path = tower.specialize(**{'sample.step.layer.width': 8})
    nested = tower.specialize(sample={'step': {'layer': {'width': 8}}})
    assert by_path.statics_by_path() == nested.statics_by_path()

    try:
        # Bare keys bind only on the outer record rather than routing implicitly into nested members.
        tower.specialize(width=8)
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
