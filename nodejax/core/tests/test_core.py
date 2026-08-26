"""Core contract: authoring, views, pytree identity, grad-through-node."""

import jax
import jax.numpy as jnp
import nodejax

from nodejax import scan, PNode, Node, Leaf
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator
from nodejax import nn


def test_every_declared_package_export_exists():
    missing = [name for name in nodejax.__all__ if name not in vars(nodejax)]
    assert missing == []


def test_plain_node():
    double = Leaf(lambda input: input * 2.0, name='double')
    assert isinstance(double, PNode) and not double.cyclic
    assert double.apply(3.0) == 6.0
    assert double(3.0) == 6.0


def test_parametric_node():
    gain = Gain()
    assert isinstance(gain, Node) and gain.parametric
    g = gain.parameterize(scale=2.0)
    assert isinstance(g, PNode)
    assert g.apply(3.0) == 6.0


def test_cyclic_node():
    integrator = Integrator()
    node = integrator.parameterize()
    state = node.init()
    state, out = node.apply(state, 2.0)
    assert out == 2.0
    state, out = node.apply(state, 3.0)
    assert out == 5.0

    # the scanned node's state IS the inner node's, so its init serves
    final, outs = scan(node)(node.init(), jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(outs, jnp.array([1.0, 3.0, 6.0]))


def test_generic_is_a_closure():
    """The matmul case that breaks discovery tracing: no dummies, no problem."""
    linear = nn.Linear(3).with_input(jnp.zeros(4)).bind(
        Struct(w=jnp.ones((4, 3)), b=jnp.zeros(3)))
    out = linear.apply(jnp.arange(4.0))
    assert out.shape == (3,)
    assert jnp.allclose(out, 6.0)


def test_treedef_stable_across_bindings():
    gain = Gain()
    a = gain.parameterize(scale=jnp.array(1.0))
    b = gain.parameterize(scale=jnp.array(2.0))
    assert type(a) is type(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)


def test_grad_wrt_node():
    """The pytree is the object: grad w.r.t. the bound node itself."""
    g = Gain().parameterize(scale=jnp.array(2.0))
    grads = jax.grad(lambda n: n.apply(3.0))(g)
    assert isinstance(grads, PNode)
    assert jnp.allclose(grads.param.scale, 3.0)


def test_jit_and_treedef_reuse():
    """One class per PNode means jit caches hit across rebindings."""
    gain = Gain()
    a = gain.parameterize(scale=jnp.array(2.0))
    b = gain.parameterize(scale=jnp.array(5.0))

    @jax.jit
    def run(node, x):
        return node.apply(x)

    assert run(a, 3.0) == 6.0
    assert run(b, 3.0) == 15.0  # same treedef: cache hit, not a retrace error


def test_attribute_access_is_destructuring_only():
    """Attribute access on a bound node destructures the TREE: methods
    bind, members slice, and VALUES are spelled through .param
    explicitly. A field read off the node itself refuses, pointing at
    the spelling (recalibrated when 'param plays self' forwarding was
    removed: no field ever shadows a member or a method again)."""
    import pytest

    inner = nn.Linear(2).with_input(jnp.zeros(2)).bind(
        Struct(w=jnp.eye(2), b=jnp.ones(2)))
    with pytest.raises(AttributeError, match='values live under .param'):
        inner.w
    assert jnp.allclose(inner.param.w, jnp.eye(2))         # the one spelling

    def param(block, gain=2.0):
        return Struct(block=block, gain=jnp.asarray(gain))

    def apply(param, input):
        return param.gain * input

    def gain(param):
        return param.gain

    comp = Leaf(apply, param=param, name='comp',
                    methods=dict(gain=gain)).parameterize(block=inner)
    # param.block IS a bound node riding the tree; its values sit one
    # more explicit hop down, same rule at every level
    assert jnp.allclose(comp.param.block.param.w, jnp.eye(2))
    assert callable(comp.gain) and comp.gain() == 2.0      # methods bind
    assert comp.name == 'comp'                             # real attributes win

    with pytest.raises(AttributeError, match='values live under .param'):
        comp.nonexistent
