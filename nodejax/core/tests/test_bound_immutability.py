"""Functional replacement and reconstruction of bound nodes."""

import jax
import jax.numpy as jnp

from nodejax import PNode, PSNode
from nodejax.control import Gain, Integrator


def test_pnode_replacement_and_tree_map_construct_new_bindings():
    bound = Gain().parameterize(scale=jnp.asarray(2.0))

    mapped = jax.tree.map(lambda value: value + 1, bound)
    replaced = bound.bind(param=mapped.param)
    assert type(mapped) is PNode
    assert mapped._def is bound._def
    assert replaced is not bound
    assert jnp.allclose(bound.param.scale, 2.0)
    assert jnp.allclose(mapped.param.scale, 3.0)
    assert jnp.allclose(replaced.param.scale, 3.0)


def test_psnode_replacement_and_apply_construct_successors():
    bound = Integrator().parameterize().bind(state=jnp.asarray(0.0))

    leaves, structure = jax.tree.flatten(bound)
    rebuilt = jax.tree.unflatten(structure, leaves)
    successor, output = rebuilt(jnp.asarray(1.0))
    replaced = bound.bind(state=jnp.asarray(4.0))

    assert type(rebuilt) is PSNode
    assert rebuilt._def is bound._def
    assert type(successor) is PSNode
    assert successor._def is bound._def
    assert replaced is not bound
    assert jnp.allclose(bound.state, 0.0)
    assert jnp.allclose(replaced.state, 4.0)
    assert jnp.allclose(successor.state, 1.0)
    assert jnp.allclose(output, 1.0)
