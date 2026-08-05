"""The aux channel: one tuple convention for secondary outputs.

The output doctrine makes it unambiguous: a node's output is an array or a
Struct (positional pairs as DATA are named Structs, in the same
named-bundle style as param/state/input; lists remain the positional
escape hatch). The 2-tuple is thereby contract syntax:

    return output          # no aux
    return output, aux     # sown losses, taps, metrics

Composites divert member aux under member names and re-emit
(carry, collection) — same pair shape, so the channel nests with no
wrapper fields and no unwrap heuristics; destructure at the top. And
because aux rides ordinary outputs, scan stacks it over time, ensemble
over members, batch over elements: the axes just appear.
"""

import jax.numpy as jnp
import optax
import pytest

from nodejax import node_def, serial, scan, train_step, split_aux
from nodejax.struct import Struct
from nodejax.examples import gain_def, mse, tile


def watched_def():
    """A node that sows: emits its activity alongside its output."""
    def param(w):
        return Struct(w=jnp.asarray(w))
    def apply(param, input):
        y = param.w * input
        return y, Struct(activity=y ** 2)
    return node_def(apply, param=param, name='watched')


def test_aux_diverts_and_carry_flows_clean():
    """Downstream members see the raw signal; the aux surfaces at the top,
    keyed by member name."""
    pipe = gain_def() >> watched_def() >> gain_def()
    bound = pipe.parameterize(gain=Struct(scale=jnp.asarray(2.0)),
                              watched=Struct(w=jnp.asarray(3.0)),
                              gain_2=Struct(scale=jnp.asarray(10.0)))
    out, aux = bound.apply(1.0)
    assert out == 60.0                   # 1 * 2 * 3 * 10 — aux never touched the wire
    assert aux.watched.activity == 36.0  # (2*3)^2, sown from mid-pipe


def test_aux_nests_through_composition():
    inner = gain_def() >> watched_def()
    outer = serial(core=inner, post=gain_def())
    bound = outer.parameterize(core=Struct(gain=Struct(scale=2.0), watched=Struct(w=3.0)),
                               post=Struct(scale=10.0))
    out, aux = bound.apply(1.0)
    assert out == 60.0
    assert aux.core.watched.activity == 36.0   # member path, nested


def test_aux_stacks_under_scan():
    """Time axis appears on the aux automatically: it is just output."""
    def watched_integrator():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def init(param):
            return jnp.asarray(0.0)
        def apply(param, state, input):
            new = state + param.gain * input
            return new, (new, Struct(mag=new ** 2))   # (state, (output, aux))
        return node_def(apply, param=param, init=init, name='wint')

    pipe = watched_integrator() >> gain_def()
    seq = scan(pipe).parameterize(wint=Struct(gain=1.0), gain=Struct(scale=10.0))
    ys, aux = seq.apply(jnp.array([1.0, 1.0, 1.0]))

    assert jnp.allclose(ys, jnp.array([10.0, 20.0, 30.0]))
    assert jnp.allclose(aux.wint.mag, jnp.array([1.0, 4.0, 9.0]))  # (T,)


def test_aux_loss_in_training():
    """The sow-a-regularizer use case: the loss destructures the pair, and
    the trained weight lands at the analytic compromise."""
    model = watched_def()   # y = w*x, aux.activity = y^2
    lam = 0.125

    def loss(output, target):
        y, aux = output
        return mse(y, target) + lam * aux.activity

    trainer = train_step(model, loss, optax.adam(0.05))
    state = trainer.init(model=model.parameterize(w=jnp.asarray(0.0)).param)
    steps = 600
    inputs = Struct(input=tile(jnp.asarray(2.0), steps),
                    target=tile(jnp.asarray(6.0), steps))
    final, losses = trainer.scan(state, inputs)

    # d/dw [(2w-6)^2 + lam*(2w)^2] = 0  ->  w = 24/(8 + 8*lam) = 8/3
    assert jnp.allclose(final.model.w, 24.0 / (8.0 + 8.0 * lam), atol=0.01)


def test_split_aux_is_the_whole_convention():
    """The channel is one public function; custom composites reuse it."""
    out, aux = split_aux((1.0, Struct(a=2.0)))
    assert out == 1.0 and aux.a == 2.0            # pair = (output, aux)
    out, aux = split_aux(Struct(y=1.0, z=2.0))
    assert out.y == 1.0 and aux is None           # Structs are plain outputs
    out, aux = split_aux(jnp.asarray(3.0))
    assert aux is None                            # arrays too
    with pytest.raises(TypeError, match='2-tuple'):
        split_aux((1.0, 2.0, 3.0))                # loud, not misread