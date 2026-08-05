"""Layer 2: declare the input, derive the rest.

- meta() derives param/state/output specs by eval_shape — exact through
  matmuls, computed from the functions that actually run
- the input channel: init receives input=<example> and rng=<key> from
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

from nodejax import (node_def, batch, scan,
                           spec, spec_of, materialize, initialize, meta)
from nodejax.struct import Struct
from nodejax.examples import integrator_def, Linear, pd_def, plant_node, walker_def


def tree_ema_def():
    """The rewrite.md tree-EMA case: state whose STRUCTURE copies the input,
    derived from the init input value — no declaration possible upfront."""
    def param(alpha):
        return Struct(alpha=jnp.asarray(alpha))

    def init(param, ndef):
        return jax.tree.map(jnp.zeros_like, ndef.input)

    def apply(param, state, input):
        new = jax.tree.map(lambda s, x: (1 - param.alpha) * s + param.alpha * x, state, input)
        return new, new

    return node_def(apply, param=param, init=init, name='tema')


def test_declared_input_spec_derives_meta():
    """Declare only the input; param/state/output derive by eval_shape —
    including through shape-dependent ops (matmul)."""
    def param(weight, bias):
        return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))
    def apply(param, input):
        return input @ param.weight + param.bias

    lin = node_def(apply, param=param, name='lin', apply_input_spec=spec(4))
    node = lin.parameterize(weight=jnp.ones((4, 3)), bias=jnp.zeros(3))

    m = meta(node)
    assert m.apply_input_spec.shape == (4,)
    assert m.param_spec.weight.shape == (4, 3)
    assert m.state_spec == ()
    assert m.output_spec.shape == (3,)


def test_meta_of_cyclic_node():
    node = integrator_def().parameterize(gain=jnp.array(1.0))
    m = meta(node, input=spec(()))
    assert m.cyclic
    assert m.state_spec.shape == ()
    assert m.output_spec.shape == ()


def test_state_structure_from_input():
    """State structure copies the input structure (tree-EMA): initialize
    materializes the spec as init's input value."""
    ema = tree_ema_def().parameterize(alpha=0.5)

    state = initialize(ema, input=spec(3))
    assert state.shape == (3,)

    signal = Struct(a=spec((2,)), b=spec(()))
    state = initialize(ema, input=signal)
    assert state.a.shape == (2,) and state.b.shape == ()

    # a concrete example works identically (and would prime value-dependent state)
    state = initialize(ema, input=Struct(a=jnp.ones(2), b=jnp.asarray(1.0)))
    assert jnp.allclose(state.a, 0.0)


def test_pipe_init_propagates_input():
    """Composite init hands each member ITS OWN input, threaded through the
    preceding members — pd and plant carry no shape statics at all."""
    pipe = pd_def(0.1) >> plant_node(0.1, 1.0, 0.3).ndef
    bound = pipe.parameterize(pd=Struct(kp=jnp.array(1.0), kd=jnp.array(0.0)))

    state = bound.with_input(spec(2)).init()
    assert state.pd.shape == (2,)          # previous error, shaped by the signal
    assert state.plant.pos.shape == (2,)   # shaped by pd's OUTPUT, one hop downstream


def test_pipe_init_shape_error_names_member():
    """Shape mismatches between members surface at init time, named — not
    at first apply."""
    pipe = Linear(4, 3) >> Linear(5, 2)  # 3 -> 5: incompatible
    bound = pipe.parameterize(
        linear4x3=Struct(weight=jnp.ones((4, 3)), bias=jnp.zeros(3)),
        linear5x2=Struct(weight=jnp.ones((5, 2)), bias=jnp.zeros(2)),
    )
    with pytest.raises(TypeError, match='linear5x2'):
        bound.with_input(spec(4)).init()


def test_pipe_init_routes_rng():
    """Composite init splits rng per member: stochastic members mid-pipe get
    independent keys — the gap explicit-state rng couldn't cover."""
    pipe = walker_def() >> walker_def()
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
    b = batch(integrator_def()).parameterize(gain=jnp.array(1.0))
    state = b.with_input(spec(5)).init()
    assert state.shape == (5,)

    # stochastic case: n and independent keys, from the input value and boundary key
    bw = batch(walker_def()).parameterize(sigma=jnp.asarray(1.0))
    state = bw.with_input(spec(3)).init(rng=jax.random.PRNGKey(0))
    state, out = bw.apply(state, jnp.zeros(3))
    assert jnp.unique(out).size == 3


def test_scan_resolves_from_first_element():
    """scan internalizes init with the first sequence element as its input, so
    input-shaped state derives from the sequence itself."""
    seq = scan(tree_ema_def()).parameterize(alpha=jnp.asarray(0.5))
    outs = seq.apply(jnp.ones((10, 3)))
    assert outs.shape == (10, 3)
    assert jnp.allclose(outs[0], 0.5)  # first step: ema from zeros toward ones


def test_materialize_and_spec_of_roundtrip():
    s = Struct(a=spec((2, 3)), b=spec(()))
    x = materialize(s)
    assert x.a.shape == (2, 3) and jnp.allclose(x.a, 0.0)
    r = spec_of(x)
    assert r.a.shape == (2, 3) and r.b.shape == ()


def test_declared_input_seeds_init():
    """node_def(input=...) doubles as the init's fallback input: when no
    caller passes a value, the declared spec materializes instead — the
    derivation seed lives on the def, and the init cannot tell which it
    received. A real value always wins."""
    def init(ndef):
        return jnp.zeros_like(ndef.input)

    def apply(state, input):
        return state + input, state

    node = node_def(apply, init=init, name='seeded', apply_input_spec=jnp.zeros(3))
    assert node.init().shape == (3,)                          # seeded by declaration
    assert node.with_input(jnp.zeros((2, 3))).init().shape == (2, 3)  # with_input wins
