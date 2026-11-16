"""Verifies named PyTree key paths on `Struct` and `PNode` alongside `replace_by_path`.

Because `Struct` and `PNode` register named PyTree paths, `jax.tree_util` and
Optax path masks operate directly on bound Nodes and compose with `train_step`.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import trained, scan, train_step
from nodejax.struct import Struct
from nodejax.control import Gain


def test_param_paths_are_named():
    """Verifies that PyTree paths carry member and field names across both `.param` and `PNode`."""
    bound = (Gain() >> Gain()).parameterize(
        gain=Struct(scale=jnp.asarray(1.0)), gain_2=Struct(scale=jnp.asarray(2.0)))

    leaves = jax.tree_util.tree_flatten_with_path(bound.param)[0]
    assert [jax.tree_util.keystr(path) for path, _ in leaves] == \
        ['.gain.scale', '.gain_2.scale']

    leaves = jax.tree_util.tree_flatten_with_path(bound)[0]
    assert [jax.tree_util.keystr(path) for path, _ in leaves] == \
        ['.gain.scale', '.gain_2.scale']


def test_freeze_by_path_with_plain_optax():
    """Verifies that an Optax path mask freezes a selected model member inside `train_step`."""
    pipe = Gain() >> Gain()
    model = pipe.parameterize(gain=Struct(scale=jnp.asarray(1.0)),
                              gain_2=Struct(scale=jnp.asarray(1.0)))

    def label(path, leaf):
        return 'frozen' if 'gain_2' in jax.tree_util.keystr(path) else 'train'

    model_labels = jax.tree_util.tree_map_with_path(label, model.param)
    labels = Struct(model=model_labels)
    optimizer = optax.multi_transform(
        {'train': optax.adam(0.1), 'frozen': optax.set_to_zero()}, labels)

    trainer = train_step(model.initialize(),
                         lambda pred, target: (pred - target) ** 2, optimizer)
    final, aux = trained(trainer).apply(input=jnp.full(500, 2.0), target=jnp.full(500, 12.0))

    assert final.param.gain_2.scale == 1.0
    assert jnp.allclose(final.param.gain.scale, 6.0, atol=0.05)
    assert aux.loss[-1] < 1e-3


def test_replace_by_path():
    """Verifies `replace_by_path` with direct values, relative callables, and invalid paths."""
    import pytest
    from nodejax import replace_by_path
    from nodejax import nn
    node = nn.Linear(2).with_input(jnp.zeros(2)).bind(
        Struct(w=jnp.eye(2), b=jnp.ones(2)))
    edited = replace_by_path(node, {
        '.b': jnp.zeros(2),
        '.w': lambda w: 3.0 * w,
    })
    assert jnp.allclose(edited.param.w, 3.0 * jnp.eye(2))
    assert jnp.allclose(edited.param.b, 0.0)
    assert jnp.allclose(node.param.w, jnp.eye(2))

    with pytest.raises(KeyError):
        replace_by_path(node, {'.typo': 1.0})
