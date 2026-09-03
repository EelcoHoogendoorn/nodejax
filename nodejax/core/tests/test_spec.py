"""Layer 2: declare the input, derive the rest.

- the input slot: init receives input=<example> and rng=<key> from
  services (initialize, composite init, scan, batch), takes what it declares
- composite init threads the value member-to-member: input-shaped state and
  rng keys reach every pipe member, and shape mismatches surface at init
  time with the member named
- the friction this dissolves, now asserted: state-from-input (tree-EMA),
  pd/plant without shape statics, batch init without n
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import (Node, Leaf, Composite, batch, scan, scanned,
                           spec, spec_of, materialize)
from nodejax.struct import Struct
from nodejax.control import Integrator, PD, Walker


def Plant(dt: float=0.01, spring_k: float=1.0, damping_c: float=0.1) -> Node:
    def init(node):
        return Struct(pos=jnp.zeros_like(node.input), vel=jnp.zeros_like(node.input))
    def apply(state, input):
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos
    return Leaf(apply, init=init, name='plant')


def TreeEMA() -> Node:
    """The rewrite.md tree-EMA case: state whose STRUCTURE copies the input,
    derived from the init input value — no declaration possible upfront."""
    def param(alpha):
        return Struct(alpha=jnp.asarray(alpha))

    def init(node, param):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(param, state, input):
        new = jax.tree.map(lambda s, x: (1 - param.alpha) * s + param.alpha * x, state, input)
        return new, new

    return Leaf(apply, param=param, init=init, name='tema')


def test_state_structure_from_input():
    """State structure copies the input structure (tree-EMA): the bound
    spec shapes what init builds."""
    ema = TreeEMA().parameterize(alpha=0.5)

    state = ema.with_input(spec(3)).bind(ema.param).init()
    assert state.shape == (3,)

    signal = Struct(a=spec((2,)), b=spec(()))
    state = ema.with_input(signal).bind(ema.param).init()
    assert state.a.shape == (2,) and state.b.shape == ()

    # a concrete example works identically (and would prime value-dependent state)
    state = ema.with_input(
        Struct(a=jnp.ones(2), b=jnp.asarray(1.0))).bind(ema.param).init()
    assert jnp.allclose(state.a, 0.0)


def test_pipe_init_propagates_input():
    """Composite init hands each member ITS OWN input, threaded through the
    preceding members — pd and plant carry no shape statics at all."""
    pipe = PD(0.1) >> Plant(0.1, 1.0, 0.3).node
    bound = pipe.parameterize(pd=Struct(kp=jnp.array(1.0), kd=jnp.array(0.0)))

    state = bound.with_input(spec(2)).bind(bound.param).init()
    assert state.pd.shape == (2,)          # previous error, shaped by the signal
    assert state.plant.pos.shape == (2,)   # shaped by pd's OUTPUT, one hop downstream


def DeclaredLinear(n_in: int, n_out: int) -> Node:
    """A layer whose fan-in is DECLARED rather than read from the resolved input spec,
    so a pipe can be wired with members that disagree. nn.Linear sizes
    itself from what it is handed and therefore cannot be mismatched."""
    def param(weight, bias):
        return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))

    def apply(param, input):
        return input @ param.weight + param.bias

    return Leaf(apply, param=param, name=f'linear{n_in}x{n_out}')


def test_pipe_shape_error_names_member():
    """Shape mismatches between members surface where shapes are actually
    walked: resolving a parameter-bound pipe names the offender. The init
    of an all-acyclic pipe builds only empty slots and probes nothing, so
    it is no longer that place (recalibrated when the init walk learned to
    skip acyclic members)."""
    pipe = DeclaredLinear(4, 3) >> DeclaredLinear(5, 2)  # 3 -> 5: incompatible
    with pytest.raises(TypeError, match='linear5x2'):
        pipe.with_input(spec(4)).parameterize(
            linear4x3=Struct(weight=jnp.ones((4, 3)), bias=jnp.zeros(3)),
            linear5x2=Struct(weight=jnp.ones((5, 2)), bias=jnp.zeros(2)),
        )


def test_pipe_init_routes_rng():
    """Composite init splits rng per member: stochastic members mid-pipe get
    independent keys — the gap explicit-state rng couldn't cover."""
    pipe = Walker() >> Walker()
    bound = pipe.parameterize(walker=Struct(sigma=jnp.asarray(1.0)),
                              walker_2=Struct(sigma=jnp.asarray(1.0)))

    state = bound.init(rng=jax.random.PRNGKey(0))
    assert 'rng' in state.walker and 'rng' in state.walker_2
    assert jnp.any(state.walker.rng != state.walker_2.rng)  # split, not copied

    # the closed pipe steps: both walkers advance independently
    state, out = bound.apply(state, 0.0)
    state, out2 = bound.apply(state, 0.0)
    assert not jnp.allclose(out, out2)


def test_batch_init_from_input():
    """batch(...).init derives n and the per-element input from a batched
    spec or example — no explicit n."""
    b = batch(Integrator()).parameterize()
    state = b.with_input(spec(5)).bind(b.param).init()
    assert state.shape == (5,)

    # stochastic case: n and independent keys, from the input value and boundary key
    bw = batch(Walker()).parameterize(sigma=jnp.asarray(1.0))
    state = bw.with_input(spec(3)).bind(bw.param).init(
        rng=jax.random.PRNGKey(0))
    state, out = bw.apply(state, jnp.zeros(3))
    assert jnp.unique(out).size == 3


def test_scan_resolves_from_first_element():
    """scan internalizes init with the first sequence element as its input, so
    input-shaped state derives from the sequence itself."""
    seq = scanned(TreeEMA()).parameterize(alpha=jnp.asarray(0.5))
    outs = seq.apply(jnp.ones((10, 3)))
    assert outs.shape == (10, 3)
    assert jnp.allclose(outs[0], 0.5)  # first step: ema from zeros toward ones


def test_materialize_and_spec_of_roundtrip():
    s = Struct(a=spec((2, 3)), b=spec(()))
    x = materialize(s)
    assert x.a.shape == (2, 3) and jnp.allclose(x.a, 0.0)
    r = spec_of(x)
    assert r.a.shape == (2, 3) and r.b.shape == ()


def test_declared_input_starts_init():
    """Leaf(input=...) doubles as the init's fallback input: when no
    caller passes a value, the declared spec materializes instead — the
    derivation lives on the node, and the init cannot tell which it
    received. A real value always wins."""
    def init(node):
        return jnp.zeros_like(node.input)

    def apply(state, input):
        return state + input, state

    node = Leaf(apply, init=init, name='shaped', apply_input_spec=jnp.zeros(3))
    assert node.init().shape == (3,)                          # shaped by declaration
    resolved = node.with_input(jnp.zeros((2, 3))).bind(node.param)
    assert resolved.init().shape == (2, 3)


def test_resolved_bundle_spec_requires_every_sequence_field():
    from nodejax import scan

    def acc():
        def param(gain=1.0):
            return Struct(gain=gain)

        def init():
            return jnp.asarray(0.0)

        def apply(param, state, input):
            s = state + input * param.gain
            return s, s

        return Leaf(apply, param=param, init=init, name='acc')

    def rig():
        members = Composite(acc=acc())

        def apply(self, x, bias):
            return self.acc(x + bias)

        return members(apply, name='rig')

    sc = scanned(rig()).with_input(
        Struct(x=jnp.zeros(1), bias=jnp.zeros(1))).parameterize()
    ys = sc.apply(x=jnp.ones(3), bias=jnp.zeros(3))
    assert jnp.allclose(ys, jnp.array([1.0, 2.0, 3.0]))

    with pytest.raises(TypeError, match='missing required input fields'):
        sc.apply(x=jnp.ones(3))

    with pytest.raises(TypeError, match='unknown input fields'):
        sc.apply(x=jnp.ones(3), bias=jnp.zeros(3), extra=jnp.ones(3))

    with pytest.raises(TypeError):
        sc.apply(x=jnp.ones((3, 2)), bias=jnp.zeros(3))
