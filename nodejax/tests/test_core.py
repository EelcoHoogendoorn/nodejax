"""Core contract: authoring, views, pytree identity, grad-through-node."""

import jax
import jax.numpy as jnp

from nodejax import Node, NodeDef, node_def
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator
from nodejax import nn


def test_plain_node():
    double = node_def(lambda input: input * 2.0, name='double')
    assert isinstance(double, Node) and not double.cyclic
    assert double.apply(3.0) == 6.0
    assert double(3.0) == 6.0


def test_parametric_node():
    gain = Gain()
    assert isinstance(gain, NodeDef) and gain.parametric
    g = gain.parameterize(scale=2.0)
    assert isinstance(g, Node)
    assert g.apply(3.0) == 6.0


def test_cyclic_node():
    integrator = Integrator()
    node = integrator.parameterize(gain=jnp.array(1.0))
    state = node.init()
    state, out = node.apply(state, 2.0)
    assert out == 2.0
    state, out = node.apply(state, 3.0)
    assert out == 5.0

    final, outs = node.scan(None, jnp.array([1.0, 2.0, 3.0]))
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
    assert isinstance(grads, Node)
    assert jnp.allclose(grads.param.scale, 3.0)


def test_jit_and_treedef_reuse():
    """One class per Node means jit caches hit across rebindings."""
    gain = Gain()
    a = gain.parameterize(scale=jnp.array(2.0))
    b = gain.parameterize(scale=jnp.array(5.0))

    @jax.jit
    def run(node, x):
        return node.apply(x)

    assert run(a, 3.0) == 6.0
    assert run(b, 3.0) == 15.0  # same treedef: cache hit, not a retrace error


def test_param_field_forwarding():
    """'Param plays self', readable from outside: node.field forwards to
    param fields, chains through nested nodes, and loses to methods and
    real Node attributes."""
    import pytest
    from functools import partial as _p

    inner = nn.Linear(2).with_input(jnp.zeros(2)).bind(
        Struct(w=jnp.eye(2), b=jnp.ones(2)))
    assert jnp.allclose(inner.w, jnp.eye(2))               # field forwarding

    def param(block, gain=2.0):
        return Struct(block=block, gain=jnp.asarray(gain))

    def apply(param, input):
        return param.gain * input

    def gain(param):                                       # method named like a field
        return param.gain

    comp = node_def(apply, param=param, name='comp',
                    methods=dict(gain=gain)).parameterize(block=inner)
    assert jnp.allclose(comp.block.w, jnp.eye(2))     # chains through the Node
    assert callable(comp.gain) and comp.gain() == 2.0      # methods beat fields
    assert comp.name == 'comp'                             # real attributes beat fields
    assert comp.param.gain == 2.0                          # the unambiguous spelling

    with pytest.raises(AttributeError, match='param fields'):
        comp.nonexistent
