"""Named-path parameter surgery.

Struct and PNode register KEYED pytree paths, so jax's path-aware tooling —
and therefore optax's — works on nodes with zero framework helpers:
freeze-by-path, per-subtree optimizers, decay masks are all plain
jax.tree_util + optax, composing with train_step unchanged.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import trained, scan, train_step
from nodejax.struct import Struct
from nodejax.control import Gain


def test_param_paths_are_named():
    """Pytree paths carry member and field names, not anonymous indices."""
    bound = (Gain() >> Gain()).parameterize(
        gain=Struct(scale=jnp.asarray(1.0)), gain_2=Struct(scale=jnp.asarray(2.0)))

    leaves = jax.tree_util.tree_flatten_with_path(bound.param)[0]
    assert [jax.tree_util.keystr(path) for path, _ in leaves] == \
        ['.gain.scale', '.gain_2.scale']

    # and through the PNode itself
    leaves = jax.tree_util.tree_flatten_with_path(bound)[0]
    assert [jax.tree_util.keystr(path) for path, _ in leaves] == \
        ['.gain.scale', '.gain_2.scale']


def test_freeze_by_path_with_plain_optax():
    """Freeze a member by name via optax.multi_transform — no framework
    surgery API, and train_step consumes the composed optimizer as-is."""
    pipe = Gain() >> Gain()
    model = pipe.parameterize(gain=Struct(scale=jnp.asarray(1.0)),
                              gain_2=Struct(scale=jnp.asarray(1.0)))

    def label(path, leaf):
        return 'frozen' if 'gain_2' in jax.tree_util.keystr(path) else 'train'

    labels = jax.tree_util.tree_map_with_path(label, model.param)
    optimizer = optax.multi_transform(
        {'train': optax.adam(0.1), 'frozen': optax.set_to_zero()}, labels)

    trainer = train_step(model.initialize(),
                         lambda pred, target: (pred - target) ** 2, optimizer)
    final, aux = trained(trainer).apply(input=jnp.full(500, 2.0), target=jnp.full(500, 12.0))

    assert final.param.gain_2.scale == 1.0                     # frozen: bit-exact
    assert jnp.allclose(final.param.gain.scale, 6.0, atol=0.05)  # carried the load
    assert aux.loss[-1] < 1e-3


def test_replace_by_path():
    """The promoted surgery primitive: absolute values, relative
    callables, loud failure on unknown addresses."""
    import pytest
    from nodejax import replace_by_path
    from nodejax import nn
    node = nn.Linear(2).with_input(jnp.zeros(2)).bind(
        Struct(w=jnp.eye(2), b=jnp.ones(2)))
    edited = replace_by_path(node, {
        '.b': jnp.zeros(2),                       # absolute
        '.w': lambda w: 3.0 * w,                  # relative
    })
    assert jnp.allclose(edited.param.w, 3.0 * jnp.eye(2))
    assert jnp.allclose(edited.param.b, 0.0)
    assert jnp.allclose(node.param.w, jnp.eye(2))  # original untouched

    with pytest.raises(KeyError):
        replace_by_path(node, {'.typo': 1.0})
