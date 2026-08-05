"""The static stage: composable generics.

A pipe of generics exposes member statics as one nested tree at a single
point of use; defaults merge; transforms commute with specialization.
"""

import jax.numpy as jnp

from nodejax.struct import Struct

from nodejax import NodeDef, GenericDef, node_def, generic, ensemble
from nodejax.examples import linear


def test_generic_leaf():
    assert isinstance(linear, GenericDef)
    nd = linear.specialize(in_features=3, out_features=2)
    assert isinstance(nd, NodeDef) and nd.parametric
    node = linear(3, 2).parameterize(weight=jnp.ones((3, 2)), bias=jnp.zeros(2))
    assert jnp.allclose(node.apply(jnp.ones(3)), 3.0)


def test_generic_static_composition():
    """The point of the static stage: a pipe of generics exposes member
    statics as ONE nested tree at a single point of use — no threading
    statics through intermediate constructors."""
    mlp = linear >> linear
    assert isinstance(mlp, GenericDef)

    nd = mlp.specialize(
        linear={'in_features': 4, 'out_features': 8},
        linear_2={'in_features': 8, 'out_features': 2},
    )
    bound = nd.parameterize(
        linear=Struct(weight=jnp.ones((4, 8)), bias=jnp.zeros(8)),
        linear_2=Struct(weight=jnp.ones((8, 2)), bias=jnp.zeros(2)),
    )
    assert jnp.allclose(bound.apply(jnp.ones(4)), 32.0)  # 4 * 8 fan-in


def test_generic_pipe_with_plain_members():
    """AlexNet shape: generics mixed with plain (param-less, static-less)
    nodes; those members simply don't appear in the static or param trees."""
    relu = node_def(lambda input: jnp.maximum(input, 0.0), name='relu')
    mlp = linear >> relu >> linear

    nd = mlp.specialize(
        linear={'in_features': 2, 'out_features': 2},
        linear_2={'in_features': 2, 'out_features': 1},
    )
    bound = nd.parameterize(
        linear=Struct(weight=jnp.eye(2), bias=jnp.zeros(2)),
        linear_2=Struct(weight=jnp.ones((2, 1)), bias=jnp.zeros(1)),
    )
    # relu clips the negative channel: [1, -2] -> [1, 0] -> 1.0
    assert jnp.allclose(bound.apply(jnp.array([1.0, -2.0])), 1.0)


def test_generic_defaults_merge():
    """Authored generics pre-configure members; supplied statics merge over
    defaults leaf-wise."""
    head = generic(linear.specialize_fn, name='head', out_features=2)
    nd = head.specialize(in_features=3)                    # default out_features
    assert isinstance(nd, NodeDef)
    node = nd.parameterize(weight=jnp.ones((3, 2)), bias=jnp.zeros(2))
    assert node.apply(jnp.ones(3)).shape == (2,)

    nd5 = head.specialize(in_features=3, out_features=5)   # override wins
    node5 = nd5.parameterize(weight=jnp.ones((3, 5)), bias=jnp.zeros(5))
    assert node5.apply(jnp.ones(3)).shape == (5,)


def test_generic_derived_statics_are_closures():
    """Both natures coexist: DERIVED statics are ordinary closure logic
    (hidden wired between layers); FREE statics surface through the args."""
    @generic
    def mlp(in_features, hidden, out_features):
        return (linear(in_features, hidden) >> linear(hidden, out_features))

    bound = mlp(4, 8, 2).parameterize(
        linear=Struct(weight=jnp.ones((4, 8)), bias=jnp.zeros(8)),
        linear_2=Struct(weight=jnp.ones((8, 2)), bias=jnp.zeros(2)),
    )
    assert jnp.allclose(bound.apply(jnp.ones(4)), 32.0)


def test_transform_of_generic_commutes():
    """Transforms lift over the static stage by deferral:
    ensemble(G).specialize(s) == ensemble(G.specialize(s))."""
    eG = ensemble(linear)
    assert isinstance(eG, GenericDef)

    params = Struct(weight=jnp.ones((3, 2, 2)), bias=jnp.zeros((3, 2)))
    via_generic = eG.specialize(in_features=2, out_features=2).parameterize(params)
    via_def = ensemble(linear.specialize(in_features=2, out_features=2)).parameterize(params)

    x = jnp.array([1.0, 2.0])
    assert via_generic.apply(x).shape == (3, 2)
    assert jnp.allclose(via_generic.apply(x), via_def.apply(x))


def test_wildcard_statics_broadcast():
    """AMBIENT STATICS: '*.name' reaches every member that DECLARES the
    static — no threading through constructors; explicit member statics
    win over the broadcast; non-declaring and plain members are untouched."""
    @generic
    def amp(gain=1.0):
        return node_def(lambda input: gain * input, name='amp')

    @generic
    def off(bias=0.0):
        return node_def(lambda input: input + bias, name='off')

    relu = node_def(lambda input: jnp.maximum(input, 0.0), name='relu')
    pipe = amp >> off >> amp >> relu               # amp, off, amp_2, relu

    nd = pipe.specialize(**{'*.gain': 2.0}, amp_2={'gain': 5.0})
    node = nd.bind(nd.build_param())            # all-plain pipe: () per member
    # x=1: amp 2.0 -> off +0 -> amp_2 5x -> relu = 10
    assert jnp.allclose(node.apply(jnp.asarray(1.0)), 10.0)


def test_wildcard_statics_nested_depth():
    """The broadcast recurses through composed generics: one entry at the
    top reaches leaves two levels down, and a deeper explicit still wins."""
    from nodejax.compose import serial_generic

    @generic
    def amp(gain=1.0):
        return node_def(lambda input: gain * input, name='amp')

    inner = serial_generic(a=amp, b=amp)
    outer = serial_generic(body=inner, tail=amp)

    from nodejax import Node
    nd = outer.specialize(**{'*.gain': 3.0}, body={'b': {'gain': 1.0}})
    node = nd if isinstance(nd, Node) else nd.bind(nd.build_param())
    out = node.apply(jnp.asarray(1.0))
    assert jnp.allclose(out, 9.0)                  # 3 (a) * 1 (b explicit) * 3 (tail)


def test_wildcard_statics_through_transform():
    """Deferred transforms forward the broadcast: the wildcard survives
    batch(G) and resolves when the inner generic finally specializes."""
    from nodejax import batch

    @generic
    def amp(gain=1.0):
        return node_def(lambda input: gain * input, name='amp')

    from nodejax import Node
    nd = batch(amp >> amp).specialize(**{'*.gain': 2.0})
    node = nd if isinstance(nd, Node) else nd.bind(nd.build_param())
    out = node.apply(jnp.ones(3))
    assert jnp.allclose(out, 4.0)


def test_wildcard_mode_flag():
    """The motivating case: a train/eval flag as ONE ambient entry —
    mode-sensitive members switch together, params transfer by bind."""
    @generic
    def noisy_scale(train=True):
        # a stand-in for dropout: behavior differs by mode, params do not
        def param(rng):
            return __import__('jax').random.normal(rng.next(), ())
        def apply(param, input):
            return input * param * (0.5 if train else 1.0)
        return node_def(apply, param=param, name='ns')

    g = noisy_scale >> noisy_scale
    import jax
    train_nd = g.specialize(**{'*.train': True})
    eval_nd = g.specialize(**{'*.train': False})
    model = train_nd.parameterize(rng=jax.random.PRNGKey(0))

    train_out = model.apply(jnp.asarray(1.0))
    eval_out = eval_nd.apply(model.param, jnp.asarray(1.0))        # same weights
    assert jnp.allclose(eval_out, train_out * 4.0)                 # 0.5^2 lifted
