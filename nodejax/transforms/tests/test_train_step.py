"""train_step: internalize optimization — params become trainer state."""

import jax.numpy as jnp
import optax

from nodejax import Node, train_step
from nodejax.struct import Struct
from nodejax.examples import gain_def


def test_train_step_convergence():
    Gain = gain_def()
    trainer = train_step(Gain, lambda pred, target: (pred - target) ** 2, optax.sgd(0.01))
    assert isinstance(trainer, Node) and trainer.cyclic

    state = trainer.init(model=Gain.parameterize(scale=jnp.array(1.0)).param)
    inputs = Struct(input=jnp.full(500, 2.0), target=jnp.full(500, 6.0))
    final, losses = trainer.scan(state, inputs)

    assert jnp.allclose(final.model.scale, 3.0, atol=0.01)
    assert losses[-1] < 1e-3
