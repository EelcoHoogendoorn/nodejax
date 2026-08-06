"""Example nodes and user-land transforms, shared by the test suite.

feedback and meta_sgd are ordinary node_def-based code written against
the public contract, as downstream code would write it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import Node, node_def, derive, generic, train_step
from nodejax import hoist_rng


# --- loss and data helpers ---

def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def tile(tree, n):
    """Tile a pytree along a new leading axis (for scanning a fixed batch)."""
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)


# --- basic example nodes ---

def gain_def():
    """PN analog: output = scale * x."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))
    def apply(param, input):
        return param.scale * input
    return node_def(apply, param=param, name='gain')


def integrator_def():
    """PCN analog: state accumulates gain * x."""
    def param(gain):
        return Struct(gain=jnp.asarray(gain))
    def init(param):
        return 0.0
    def apply(param, state, input):
        new = state + param.gain * input
        return new, new
    return node_def(apply, param=param, init=init, name='integrator')


def Linear(in_features, out_features):
    """Generic node as a plain closure; specialize is function application."""
    def param(weight, bias):
        return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))
    def apply(param, input):
        return input @ param.weight + param.bias
    return node_def(apply, param=param, name=f'linear{in_features}x{out_features}')


@generic
def linear(in_features, out_features):
    """The same layer as a GenericDef: free statics, exposed for composition
    (pipes of generics take nested statics at a single point of use)."""
    def param(weight, bias):
        return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))
    def apply(param, input):
        return input @ param.weight + param.bias
    return node_def(apply, param=param, name='linear')


def Ema(momentum):
    """Stateful normalizer analog: running mean as cyclic state."""
    def param(gamma, beta):
        return Struct(gamma=jnp.asarray(gamma), beta=jnp.asarray(beta))
    def init(param):
        return 0.0
    def apply(param, state, input):
        new_mean = (1 - momentum) * state + momentum * jnp.mean(input)
        return new_mean, (input - new_mean) * param.gamma + param.beta
    return node_def(apply, param=param, init=init, name='ema')


def Norm(momentum=0.1, eps=1e-5, axis=0, shape=(1,)):
    """Batchnorm as a generic parametric cyclic node.

    Statics: momentum/eps/axis/shape (plain closure captures).
    Param:   gamma, beta.
    State:   running mean/var, updated by EMA every apply — no train/eval
             mode flags; evaluation is simply reusing a frozen state.
    """
    def param(gamma, beta):
        return Struct(gamma=jnp.asarray(gamma), beta=jnp.asarray(beta))

    def init(param):
        return Struct(running_mean=jnp.zeros(shape), running_var=jnp.ones(shape))

    def apply(param, state, input):
        batch_mean = jnp.mean(input, axis=axis, keepdims=True)
        batch_var = jnp.var(input, axis=axis, keepdims=True)
        new_state = Struct(
            running_mean=(1 - momentum) * state.running_mean + momentum * batch_mean,
            running_var=(1 - momentum) * state.running_var + momentum * batch_var,
        )
        normalized = (input - state.running_mean) / jnp.sqrt(state.running_var + eps)
        return new_state, param.gamma * normalized + param.beta

    return node_def(apply, param=param, init=init, name='norm')


# --- control: closing the loop ---

def feedback(pipe, last=0.0):
    """Close the loop — a user-land control transform: the wrapped node maps
    tracking error -> output, and feedback supplies error = reference - the
    previous output. Cyclic -> cyclic; param meaning unchanged.

    When init receives an input value (by scan, spec.initialize, or an enclosing
    pipe's composite init), the previous-output seed and the inner pipe's
    state derive from it — no shape arguments anywhere. The explicit last=
    remains for manual seeding without an input value. The loop only closes
    if the pipe's output has the input's structure.

    The seed spec is the boundary hoist over the one wrapped slot: the
    pipe's seeds nest under inner=, rng rides the boundary exactly when
    the pipe's init requires it — declared to derive(), since no single
    authored signature states a spec computed from the wrapped def."""
    def init(ndef, param, state=Struct()):
        seed = state.inner if 'inner' in state else Struct()
        if 'rng' in state:
            seed = seed.replace(rng=state.rng)
        if ndef.apply_input_spec is None:
            return Struct(inner=pipe.build_state(param, seed),
                          last=jax.tree.map(jnp.asarray, last))
        return Struct(inner=pipe.build_state(param, seed, input=ndef.input),
                      last=jax.tree.map(jnp.zeros_like, ndef.input))

    def apply(param, state, input):
        error = jax.tree.map(jnp.subtract, input, state.last)
        new_inner, out = pipe.apply_fn(param, state.inner, error)
        return Struct(inner=new_inner, last=out), out

    # same param meaning as the pipe: derive inherits its constructor and
    # specs, lifting only the new apply/init
    seed_spec = hoist_rng(dict(inner=pipe.state_input_spec if pipe.cyclic else Struct()))
    return derive(pipe, apply=apply, init=init, state_input_spec=seed_spec,
                  name=f'feedback({pipe.name})')


def pd_def(dt):
    """PD controller: parametric cyclic node. Its state (the previous error,
    for the D term) is shaped by the init input value — no shape static."""
    def param(kp, kd):
        return Struct(kp=jnp.asarray(kp), kd=jnp.asarray(kd))
    def init(ndef, param):
        return jnp.zeros_like(ndef.input)
    def apply(param, state, input):
        force = param.kp * input + param.kd * (input - state) / dt
        return input, force
    return node_def(apply, param=param, init=init, name='pd')


def plant_node(dt, spring_k, damping_c):
    """Spring-mass-damper plant, semi-implicit Euler; non-parametric cyclic
    node. Elementwise, so a vector input gives independent axes; state
    (position/velocity) is shaped by the init input value."""
    def init(ndef):
        return Struct(pos=jnp.zeros_like(ndef.input), vel=jnp.zeros_like(ndef.input))
    def apply(state, input):
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos
    return node_def(apply, init=init, name='plant')


def walker_def():
    """Random-walk node: stochastic cyclic state via the reserved rng field.
    The apply reads state.rng and never splits or threads it."""
    def param(sigma):
        return Struct(sigma=jnp.asarray(sigma))

    def init(param, rng):
        return Struct(x=jnp.asarray(0.0), rng=rng)

    def apply(param, state, input):
        new_x = state.x + input + param.sigma * jax.random.normal(state.rng)
        return state.replace(x=new_x), new_x

    return node_def(apply, param=param, init=init, name='walker')


# the IMU components live with their framework comparisons
from nodejax.examples.comparisons.imu_nodejax import (derivative_node, noise_def,
                                                     drift_def, quantizer)


# --- meta-learning: promoting statics to params ---

def meta_sgd(ndef, loss_fn):
    """Meta-learn optimizer hyperparameters — a user-land variant of finetune
    needing no new library machinery: the inner learning rate is promoted from
    a static capture to a component of param, so the outer trainer learns it
    alongside the initial weights (Meta-SGD; a per-leaf lr pytree would work
    the same way).

    param = Struct(init=<model params>, lr=<inner sgd learning rate>).
    """
    def param(init, lr):
        init = init.param if isinstance(init, Node) else init
        return Struct(init=init, lr=jnp.asarray(lr))

    def apply(param, input):
        inner = train_step(ndef, loss_fn, optax.sgd(param.lr))
        tuned, _ = inner.scan(inner.init(model=param.init), input.support)
        # the contract fn, not the mirror: its THREE slots are uniform over
        # cyclicity, and this wrapper serves cyclic and plain models alike
        _, output = ndef.apply_fn(tuned.model, tuned.inner, input.query)
        return output

    return node_def(apply, param=param, name=f'meta_sgd({ndef.name})')
