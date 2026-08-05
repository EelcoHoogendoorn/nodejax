# The nodejax cookbook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/nodejax/blob/main/docs/cookbook.ipynb)

This cookbook starts from a minimal node and builds it out, explaining as it goes. Every snippet runs as written, and each section uses what the previous one defined.

## 1. A first node

```python
import jax
import jax.numpy as jnp
import optax
from nodejax import node_def, Struct, serial, train_step, ensemble, composite

def gain_def():
    def param(scale):
        return Struct(scale=jnp.asarray(scale))
    def apply(param, input):
        return param.scale * input
    return node_def(apply, param=param, name='gain')

node = gain_def().parameterize(scale=2.0)
assert node.apply(3.0) == 6.0
```

A node is at most three pure functions; this one needs two. `param` is the constructor: its signature declares what a caller must supply (`scale`, required because it has no default), and `parameterize` runs it and binds the result. `apply` computes. The names in the signatures are the specification; there is no other registration step.

## 2. State

```python
def integrator_def():
    def param(gain):
        return Struct(gain=jnp.asarray(gain))
    def init(param):
        return jnp.asarray(0.0)
    def apply(param, state, input):
        new = state + param.gain * input
        return new, new
    return node_def(apply, param=param, init=init, name='integrator')

node = integrator_def().parameterize(gain=1.0)
state = node.init()
state, y = node.apply(state, 5.0)          # cyclic: (state, input) -> (state, output)
final, trajectory = node.scan(state, jnp.ones(10))
assert trajectory[-1] == 15.0
```

The third function, `init`, builds the starting state, and its presence makes the node CYCLIC: output feeding back into the next step. A cyclic node applies as `(state, input) -> (state, output)`, which is exactly a `lax.scan` step, so `scan` over a time axis comes built in. State is explicit data you hold, never something the node hides.

## 3. Composition

```python
pipe = gain_def() >> integrator_def()
node = pipe.parameterize(gain=Struct(scale=2.0), integrator=Struct(gain=1.0))

state = node.init()
final, trajectory = node.scan(state, jnp.ones(10))
assert trajectory[-1] == 20.0
assert node.param.gain.scale == 2.0        # params: a tree named by member
assert final.integrator == 20.0            # state: same shape, same names
```

`>>` chains nodes into a node. The composite's params and state are trees keyed by member name, and both remain plain pytrees: read them, `jax.tree.map` them, `jax.grad` through them. The state threading between members that you would otherwise write by hand does not exist as user code.

## 4. Shapes and keys

```python
def linear_def(n_out):
    def param(ndef, rng):
        n_in = ndef.apply_input_spec.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))
    def apply(param, input):
        return input @ param.w + param.b
    return node_def(apply, param=param, name='linear')

relu = node_def(lambda input: jnp.maximum(input, 0.0), name='relu')

X = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
net = linear_def(8) >> relu >> linear_def(1)
model = net.with_input(X).parameterize(rng=jax.random.PRNGKey(1))
assert model.param.linear.w.shape == (4, 8)
assert model.apply(X).shape == (32, 1)
```

Two reserved names appear. `ndef` hands the constructor its own definition, resolved, so it can read the input shape instead of being told; `with_input(X)` is where that shape comes from, and the pipe walks it member by member so the second linear sees the first one's output width. `rng` declares a dependency on randomness, and what arrives is a scope-local stream of keys: every `rng.next()` yields a fresh key, as many as the body needs, with no line anywhere dedicated to splitting or naming keys. The flow of keys stays explicit at the function boundary, since the signature says a key is owed and the caller must pass one, while the plumbing below the boundary disappears: the pipe splits the one key so every member's stream is independent.

## 5. Training

```python
def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

y = X @ jnp.ones((4, 1))
trainer = train_step(net.with_input(X), mse, optax.adam(1e-2))
tstate = trainer.init(model=model.param)

steps = 300
stream = Struct(input=jnp.broadcast_to(X, (steps, *X.shape)),
                target=jnp.broadcast_to(y, (steps, *y.shape)))
tstate, losses = jax.jit(trainer.scan)(tstate, stream)
assert losses[-1] < 0.01
assert tstate.model.linear.w.shape == (4, 8)   # the weights, readable mid-training
```

`train_step` turns the model into another cyclic node whose state holds the weights, the optimizer moments, and the model's own state, so a training run is the same `scan` as section 2. The loop around it is your own Python: chunk it, log anything (the state is data), stop when you like, resume by passing the state back in.

## 6. Randomness at apply time

```python
def jitter_def():
    def param(sigma):
        return Struct(sigma=jnp.asarray(sigma))
    def apply(param, x, rng):
        return x + param.sigma * jax.random.normal(rng.next())
    return node_def(apply, param=param, name='jitter')

noisy = jitter_def().parameterize(sigma=0.1)
a = noisy.apply(x=1.0, rng=jax.random.PRNGKey(0))
b = noisy.apply(x=1.0, rng=jax.random.PRNGKey(0))
assert a == b                              # same key, same draw: still pure
```

An apply that draws noise declares `rng` too, as an input field, and receives the same scope-local stream: draw with `rng.next()` as often as the step needs. Purity is kept: the key is data, the same key reproduces the same draws. When such a node sits inside a pipe, the requirement bubbles to the boundary, and the composite splits one key toward every member that declared the need.

## 7. Wiring by hand

```python
def residual_def(body):
    def apply(self, input):
        return input + self.body(input)
    return composite(apply, members=dict(body=body), name='residual')

res = residual_def(linear_def(4) >> relu >> linear_def(4))
model = res.with_input(X).parameterize(rng=jax.random.PRNGKey(2))
assert model.apply(X).shape == (32, 4)
```

When `>>` is not the shape of your dataflow, write the step as a function of `self`: a scope-local, mutable, object-like view of the node, bound to the live params and state. Calling a member (`self.body(input)`) runs it and advances its state slice in place; reads see the current values; you write ordinary imperative wiring. Like the key stream, it is purely a scope-local abstraction: the sugar transforms the function into an ordinary pure `apply`, and to anything outside, only the node contract is visible, which is why the residual composes onward like any node.

## 8. Transforms

```python
mlp = linear_def(8) >> relu >> linear_def(1)
population = ensemble(mlp.with_input(X), n=4).parameterize(rng=jax.random.PRNGKey(3))
assert population.param.linear.w.shape == (4, 4, 8)   # a member axis, stacked
assert population.apply(X).shape == (4, 32, 1)        # four models, one call
```

Because every node declares which tree is params and which is state, transforms know which axes to act on: `ensemble` maps over params, `batch` over data, `scan` over time, `train_step` over optimization. Each returns a node, so they nest; the README's opening block is nothing but this, stacked deep.

## Where next

The [README](../README.md) for the view from the top, [`docs/handbook.md`](handbook.md) for the patient reference, and the test suite for working examples of everything: [`nodejax/examples/`](../nodejax/examples/) and [`nodejax/tests/`](../nodejax/tests/) are written to be read.
