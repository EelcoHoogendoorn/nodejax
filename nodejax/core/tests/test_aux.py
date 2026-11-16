"""Verifies `Aux` collection across `Leaf`, `Composite`, `serial`, `scan`, and `train_step`."""

import jax.numpy as jnp
import optax
import pytest

from nodejax import Node, Leaf, serial, scan, scanned, train_step, split_aux, Aux
from nodejax.struct import Struct
from nodejax.control import Gain
from nodejax import tile


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


def Watched() -> Node:
    """Builds a `Leaf` that emits squared output activity as `Aux`."""
    def param(w):
        return Struct(w=jnp.asarray(w))
    def apply(param, input):
        y = param.w * input
        return y, Aux(activity=y ** 2)
    return Leaf(apply, param=param, name='watched')


def test_aux_diverts_and_carry_flows_clean():
    """Downstream members see the raw signal; the aux surfaces at the top,
    keyed by member name."""
    pipe = Gain() >> Watched() >> Gain()
    bound = pipe.parameterize(gain=Struct(scale=jnp.asarray(2.0)),
                              watched=Struct(w=jnp.asarray(3.0)),
                              gain_2=Struct(scale=jnp.asarray(10.0)))
    out, aux = bound.apply(1.0)
    assert out == 60.0                   # Primary wire evaluates 1 * 2 * 3 * 10.
    assert aux.watched.activity == 36.0  # Mid-pipe `Aux` records (2 * 3)^2.


def test_aux_nests_through_composition():
    inner = Gain() >> Watched()
    outer = serial(core=inner, post=Gain())
    bound = outer.parameterize(core=Struct(gain=Struct(scale=2.0), watched=Struct(w=3.0)),
                               post=Struct(scale=10.0))
    out, aux = bound.apply(1.0)
    assert out == 60.0
    assert aux.core.watched.activity == 36.0


def test_aux_stacks_under_scan():
    """Verifies `scan` stacks `Aux` leaves along the leading time axis."""
    def watched_integrator():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def init(param):
            return jnp.asarray(0.0)
        def apply(param, state, input):
            new = state + param.gain * input
            return new, (new, Aux(mag=new ** 2))
        return Leaf(apply, param=param, init=init, name='wint')

    pipe = watched_integrator() >> Gain()
    seq = scanned(pipe).parameterize(wint=Struct(gain=1.0), gain=Struct(scale=10.0))
    ys, aux = seq.apply(jnp.array([1.0, 1.0, 1.0]))

    assert jnp.allclose(ys, jnp.array([10.0, 20.0, 30.0]))
    assert jnp.allclose(aux.wint.mag, jnp.array([1.0, 4.0, 9.0]))


def test_aux_loss_in_training():
    """An aux-aware loss reaches the expected analytic compromise."""
    model = Watched()
    lam = 0.125

    def loss(output, target, *, aux):
        return mse(output, target) + lam * aux.activity

    trainer = train_step(model.parameterize(w=jnp.asarray(0.0)).initialize(),
                         loss, optax.adam(0.05))
    steps = 600
    final, (_, aux) = trainer.scan(input=tile(jnp.asarray(2.0), steps),
                                   target=tile(jnp.asarray(6.0), steps))

    # Stationary point of (2w - 6)^2 + lam * (2w)^2 is w = 24 / (8 + 8 * lam) = 8 / 3.
    assert jnp.allclose(
        final.state.opt.params.model.w,
        24.0 / (8.0 + 8.0 * lam),
        atol=0.01,
    )


def test_split_aux_is_the_whole_convention():
    """Verifies `split_aux` distinguishes `(output, Aux)` pairs from `Struct`, array, and plain tuple outputs."""
    out, aux = split_aux((1.0, Aux(a=2.0)))
    assert out == 1.0 and aux.a == 2.0
    out, aux = split_aux(Struct(y=1.0, z=2.0))
    assert out.y == 1.0 and aux is None
    out, aux = split_aux(jnp.asarray(3.0))
    assert aux is None
    out, aux = split_aux((1.0, 2.0))
    assert out == (1.0, 2.0) and aux is None


def test_self_sow_sugar():
    """Verifies `self.sow(**kwargs)` emits auxiliary fields from an authored `Composite`."""
    from nodejax import Composite

    def Block():
        def apply(self, input):
            self.sow(activity=input ** 2)
            return input * 2.0
        return Composite(**{})(apply, name='sower')

    node = Block()
    out, aux = node.apply(3.0)
    assert out == 6.0
    assert aux.activity == 9.0


def test_leaf_returns_explicit_aux():
    """A Leaf returns its auxiliary values alongside its primary output."""
    def WatchedLeaf():
        def param(w):
            return Struct(w=jnp.asarray(w))
        def apply(param, input):
            output = param.w * input
            return output, Aux(activity=output ** 2)
        return Leaf(apply, param=param, name='watched_leaf')

    node = WatchedLeaf().parameterize(w=3.0)
    out, aux = node.apply(2.0)
    assert out == 6.0
    assert aux.activity == 36.0
