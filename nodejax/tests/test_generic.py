"""The static stage: composable generics.

A pipe of generics exposes member statics as one nested tree at a single
point of use; defaults merge; transforms commute with specialization.
"""

import jax.numpy as jnp

from nodejax.struct import Struct

from nodejax import nn, NodeDef, GenericDef, node_def, generic, ensemble


@generic(name='linear')
def LinearGeneric(in_features, out_features):
    """The same layer as a GenericDef: free statics, exposed for composition
    (pipes of generics take nested statics at a single point of use)."""
    def param(weight, bias):
        return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))
    def apply(param, input):
        return input @ param.weight + param.bias
    return node_def(apply, param=param, name='linear')


def test_generic_leaf():
    assert isinstance(LinearGeneric, GenericDef)
    nd = LinearGeneric.specialize(in_features=3, out_features=2)
    assert isinstance(nd, NodeDef) and nd.parametric
    node = LinearGeneric(3, 2).parameterize(weight=jnp.ones((3, 2)), bias=jnp.zeros(2))
    assert jnp.allclose(node.apply(jnp.ones(3)), 3.0)


def test_generic_static_composition():
    """The point of the static stage: a pipe of generics exposes member
    statics as ONE nested tree at a single point of use — no threading
    statics through intermediate constructors."""
    mlp = LinearGeneric >> LinearGeneric
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
    mlp = LinearGeneric >> nn.relu >> LinearGeneric

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
    head = generic(LinearGeneric.specialize_fn, name='head', out_features=2)
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
        return (LinearGeneric(in_features, hidden) >> LinearGeneric(hidden, out_features))

    bound = mlp(4, 8, 2).parameterize(
        linear=Struct(weight=jnp.ones((4, 8)), bias=jnp.zeros(8)),
        linear_2=Struct(weight=jnp.ones((8, 2)), bias=jnp.zeros(2)),
    )
    assert jnp.allclose(bound.apply(jnp.ones(4)), 32.0)


def test_transform_of_generic_commutes():
    """Transforms lift over the static stage by deferral:
    ensemble(G).specialize(s) == ensemble(G.specialize(s))."""
    eG = ensemble(LinearGeneric)
    assert isinstance(eG, GenericDef)

    params = Struct(weight=jnp.ones((3, 2, 2)), bias=jnp.zeros((3, 2)))
    via_generic = eG.specialize(in_features=2, out_features=2).parameterize(params)
    via_nodedef = ensemble(LinearGeneric.specialize(in_features=2, out_features=2)).parameterize(params)

    x = jnp.array([1.0, 2.0])
    assert via_generic.apply(x).shape == (3, 2)
    assert jnp.allclose(via_generic.apply(x), via_nodedef.apply(x))


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

    pipe = amp >> off >> amp >> nn.relu               # amp, off, amp_2, relu

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


def test_static_input_spec_and_dot_paths():
    """Verify static_input_spec reflects required/default statics and dot-path specialization works."""
    from nodejax.core import REQUIRED
    from nodejax.struct import Struct

    spec = LinearGeneric.static_input_spec
    assert spec.in_features is REQUIRED and spec.out_features is REQUIRED

    mlp = LinearGeneric >> LinearGeneric
    mlp_spec = mlp.static_input_spec
    assert mlp_spec.linear.in_features is REQUIRED
    assert mlp_spec.linear_2.out_features is REQUIRED

    # Dot-path specialization
    nd = mlp.specialize(**{"linear.in_features": 4, "linear.out_features": 8,
                           "linear_2.in_features": 8, "linear_2.out_features": 2})
    assert isinstance(nd, NodeDef)


def test_partial_static_binding():
    """Verify partial static binding returns a refined GenericDef awaiting remaining statics."""
    partial_g = LinearGeneric(in_features=10)
    assert isinstance(partial_g, GenericDef)
    assert partial_g.defaults['in_features'] == 10

    full_nd = partial_g(out_features=5)
    assert isinstance(full_nd, NodeDef)


def test_generic_in_composite_and_parallel():
    """Passing GenericDefs into composite() or parallel() yields a GenericDef composite/parallel block."""
    from nodejax import composite, parallel

    comp_g = composite(lambda self, x: self.fc2(self.fc1(x)),
                       members=dict(fc1=LinearGeneric, fc2=LinearGeneric))
    assert isinstance(comp_g, GenericDef)
    comp_nd = comp_g.specialize(fc1={'in_features': 4, 'out_features': 8},
                                fc2={'in_features': 8, 'out_features': 2})
    assert isinstance(comp_nd, NodeDef)

    par_g = parallel(a=LinearGeneric, b=LinearGeneric)
    assert isinstance(par_g, GenericDef)
    par_nd = par_g.specialize(a={'in_features': 4, 'out_features': 8},
                              b={'in_features': 8, 'out_features': 2})
    assert isinstance(par_nd, NodeDef)
