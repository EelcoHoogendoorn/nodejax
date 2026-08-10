"""train_step: internalize optimization — params become trainer state."""

import jax.numpy as jnp
import optax

from nodejax import Node, train_step, serial, nn, map_members, tree_detach
from nodejax.struct import Struct
from nodejax.control import Gain


def test_train_step_convergence():
    gain = Gain()
    trainer = train_step(gain, lambda pred, target: (pred - target) ** 2, optax.sgd(0.01))
    assert isinstance(trainer, Node) and trainer.cyclic

    state = trainer.init(model=gain.parameterize(scale=jnp.array(1.0)).param)
    inputs = Struct(input=jnp.full(500, 2.0), target=jnp.full(500, 6.0))
    final, losses = trainer.scan(state, inputs)

    assert jnp.allclose(final.model.scale, 3.0, atol=0.01)
    assert losses[-1] < 1e-3


def test_train_step_wrapper_rebuild():
    """Verify train_step is a Wrapper and rebuilds properly under map_members and tree_detach."""
    l1 = nn.Linear(4)
    l2 = nn.Linear(4)
    pipe = serial(l1=l1, l2=l2)
    trainer = train_step(pipe, lambda pred, target: jnp.sum((pred - target) ** 2), optax.sgd(0.01))

    # Replace l2 member inside trainer
    new_l2 = nn.Linear(4)
    rebuilt = map_members(trainer, lambda m: new_l2 if m is l2 else m)
    assert rebuilt.ndef.inner.members['l2'] is new_l2

    # tree_detach inside trainer by key
    detached = tree_detach(trainer, 'l2')
    assert detached.ndef.inner.members['l2'].name == 'detach(linear)'
