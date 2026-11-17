# The NodeJAX Cookbook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/nodejax/blob/main/docs/cookbook.ipynb)

Progressive runnable examples building from a minimal `Leaf` to pipelines, stochastic layers, training, axis transforms, open `Generic` templates, and closed-loop control. Every code block runs sequentially in a single Python environment.

## 1. Defining a parametric `Leaf`

```python
import jax
import jax.numpy as jnp
import optax
from nodejax import (
    Aux, Composite, Leaf, Node, Struct, Wrapper, batch, closed_loop,
    ensemble, nn, node, residual, scanned, serial, split_aux, stack,
    train_step, trained,
)

@node
def Gain() -> Node:
    def param(scale: float) -> Struct:
        return Struct(scale=scale)

    def apply(param: Struct, input: float) -> float:
        return param.scale * input

    return Leaf(apply, param=param)

gain = Gain().parameterize(scale=2.0)
assert gain(3.0) == 6.0
```

A `Leaf` declares up to three pure functions (`param`, `init`, and `apply`). `@node` records compile-time static factory arguments (changing them recompiles under `jax.jit`) and derives the default snake_case name `gain`, whereas `param` constructs post-compilation PyTree values such as `scale` that can vary across calls, `vmap` sweeps, or optimization steps without recompiling.

## 2. Cyclic Nodes

```python
@node
def Integrator() -> Node:
    def param(gain: float = 1.0) -> Struct:
        return Struct(gain=gain)

    def init() -> jax.Array:
        return jnp.asarray(0.0)

    def apply(param: Struct, state: jax.Array, input: float) -> tuple[jax.Array, jax.Array]:
        next_state = state + param.gain * input
        return next_state, next_state

    return Leaf(apply, param=param, init=init)

integrator = Integrator().parameterize(gain=1.0).initialize()

integrator, y = integrator(5.0)
assert y == 5.0

final_integrator, trajectory = integrator.scan(jnp.ones(10))
assert trajectory[-1] == 15.0
```

Adding `init` and `state` makes a Node cyclic. Calling `.initialize()` binds the initial state and returns an immutable `PSNode`. Calling a `PSNode` steps once and returns `(next_psnode, output)` without mutating the previous view, while `.scan(...)` unrolls the transition across a sequence via `jax.lax.scan`.

## 3. Sequential pipelines with `>>` and `serial`

```python
pipe = Gain().parameterize(scale=2.0) >> Integrator().parameterize(gain=1.0)
system = pipe.initialize()

final_system, trajectory = system.scan(jnp.ones(10))
assert trajectory[-1] == 20.0
assert system.param.gain.scale == 2.0
assert final_system.state.integrator == 20.0
```

The `>>` operator chains stages so each output feeds the next input, while grouping member parameters and states into immutable `Struct` trees keyed by member name (`param.gain`, `state.integrator`). `serial(gain=..., integrator=...)` builds the same pipeline with explicit member names.

## 4. Input-resolved shapes and PRNG streams

```python
@node
def Linear(out_features: int) -> Node:
    def param(node, rng) -> Struct:
        in_features = node.input.shape[-1]
        w = jax.random.normal(rng.next(), (in_features, out_features)) / jnp.sqrt(in_features)
        b = jnp.zeros(out_features)
        return Struct(w=w, b=b)

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)

relu = Leaf(lambda input: jnp.maximum(input, 0.0), name='relu')

x_batch = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
net = Linear(8) >> relu >> Linear(1)

model = net.with_input(x_batch).parameterize(rng=jax.random.PRNGKey(1))
assert model.param.linear.w.shape == (4, 8)
assert model.param.linear_2.w.shape == (8, 1)
assert model(x_batch).shape == (32, 1)
```

Declaring `node` in `param` or `init` provides resolved input metadata (`node.input.shape`), propagated downstream from `.with_input(x_batch)`. Declaring `rng` provides a scope-local `KeyStream` whose `rng.next()` draws independent keys from a single `rng=key` passed at `.parameterize(...)`.

## 5. Optimization as a Node (`train_step` and `trained`)

```python
def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)

y_batch = x_batch @ jnp.ones((4, 1))
steps = 300
train_x = jnp.broadcast_to(x_batch, (steps, *x_batch.shape))
train_y = jnp.broadcast_to(y_batch, (steps, *y_batch.shape))

trainer = train_step(model, mse, optax.adam(1e-2))
fitted_model, aux = split_aux(trained(trainer)(input=train_x, target=train_y))

assert aux.loss[-1] < 0.01
_, preds = fitted_model(x_batch)
assert preds.shape == (32, 1)
```

`train_step(model, loss_fn, optimizer, trainable=...)` turns a model into a cyclic Node whose `param` holds non-optimized parameters alongside initial values of `trainable` parameters and whose `state` holds active `trainable` parameters, any model state, and Optax moments. `trained(trainer)` scans updates across the leading step axis and returns the fitted model as a `PSNode` (`fitted_model.pnode` drops the empty state on acyclic models) alongside per-step `aux.loss`.

## 6. Runtime stochasticity and `Aux` telemetry

```python
@node
def Jitter() -> Node:
    def param(sigma: float = 0.1) -> Struct:
        return Struct(sigma=sigma)

    def apply(param: Struct, input: jax.Array, rng) -> jax.Array:
        return input + param.sigma * jax.random.normal(rng.next(), input.shape)

    return Leaf(apply, param=param)

@node
def Probe() -> Node:
    def apply(input: jax.Array) -> tuple[jax.Array, Aux]:
        return input, Aux(norm=jnp.linalg.norm(input))

    return Leaf(apply)

wired = Linear(8) >> Jitter() >> Probe() >> relu >> Linear(1)
stochastic_model = wired.with_input(x_batch).parameterize(
    rng=jax.random.PRNGKey(1),
    jitter=Struct(sigma=0.1),
)

out, aux = split_aux(stochastic_model.apply(input=x_batch, rng=jax.random.PRNGKey(4)))
assert out.shape == (32, 1)
assert aux.probe.norm.shape == ()
```

Declaring `rng` in `apply` marks forward execution as stochastic so the caller supplies `rng=key` at runtime. Returning `(output, Aux(...))` emits sideband telemetry collected under the member's path (`aux.probe.norm`) without changing the tensor passed to downstream stages.

## 7. Branching dataflow with `Composite` and `Wrapper`

```python
@node
def Highway(width: int) -> Node:
    members = Composite(
        gate=Linear(width),
        body=Linear(width) >> relu >> Linear(width),
    )

    def apply(self, input: jax.Array) -> jax.Array:
        mix = jax.nn.sigmoid(self.gate(input))
        return mix * self.body(input) + (1.0 - mix) * input

    return members(apply)

@node
def UnitNormalized(inner: Node) -> Node:
    def apply(self, input: jax.Array) -> jax.Array:
        y = self.inner(input)
        return y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-6)

    return Wrapper(inner=inner)(apply)

highway = UnitNormalized(Highway(4)).with_input(x_batch).parameterize(rng=jax.random.PRNGKey(2))
assert set(highway.param.__keys__) == {'gate', 'body'}
assert highway(x_batch).shape == (32, 4)
```

`Composite(**members)` wires multiple named child Nodes through `self.<member>(...)`. `Wrapper(inner=...)` adapts a single child Node with custom `apply` logic while keeping `param`, `state`, and `Aux` flat (`highway.param` exposes `gate` and `body` directly without an extra `inner` nesting level).

## 8. Axis transforms (`batch`, `ensemble`, `stack`)

```python
# 1. batch: shared parameters, per-sample mapped inputs, collective named batch axis
per_sample = Linear(4) >> nn.BatchNorm(0.9)
batched = batch(per_sample).with_input(x_batch).parameterize(rng=jax.random.PRNGKey(5)).initialize()
assert batched.param.linear.w.shape == (4, 4)
assert batched.state.batch_norm.mean.shape == (4,)
batched, out = batched(x_batch)
assert out.shape == (32, 4)

# 2. ensemble: independent parameter sets mapped across n members
population = ensemble(wired.with_input(x_batch), n=4).parameterize(
    rng=jax.random.PRNGKey(3),
    jitter=Struct(sigma=0.1),
)
assert population.param.linear.w.shape == (4, 4, 8)
pop_out, pop_aux = split_aux(population.apply(input=x_batch, rng=jax.random.PRNGKey(4)))
assert pop_out.shape == (4, 32, 1)
assert pop_aux.probe.norm.shape == (4,)

# 3. stack: sequential depth scan over n layers with per-layer parameters
deep = stack(Linear(8) >> relu, n=3).with_input(jnp.zeros(8)).parameterize(rng=jax.random.PRNGKey(6))
assert deep.param.linear.w.shape == (3, 8, 8)
assert deep(jnp.ones(8)).shape == (8,)
```

`batch`, `ensemble`, and `stack` map data, population, and depth axes across `param`, `state`, `input`, `Aux`, and `rng` according to each Node's contract roles.

## 9. Open `Generic` templates and `specialize`

```python
wider_net = net.specialize(**{'linear.out_features': 16})
retrained = wider_net.with_input(x_batch).parameterize(rng=jax.random.PRNGKey(8))
assert retrained.param.linear.w.shape == (4, 16)
assert retrained.param.linear_2.w.shape == (16, 1)

unit = Linear()
assert unit.generic
generic_tower = stack(unit, n=3)

concrete_tower = generic_tower.specialize(**{'layer.out_features': 8})
tower_model = concrete_tower.with_input(jnp.zeros(8)).parameterize(rng=jax.random.PRNGKey(9))
assert tower_model.param.w.shape == (3, 8, 8)
```

`.specialize(**overrides)` replays `@node` construction with updated static arguments and recomputes downstream input shapes. Omitting required static arguments on a `@node` call returns a `Generic` blueprint that composes through pipes and transforms before dimensions are supplied.

## 10. Closed-loop control and `tree_view`

```python
@node
def FirstOrderPlant(dt: float = 0.05) -> Node:
    def init(node) -> jax.Array:
        return jnp.zeros_like(node.input)

    def apply(state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_state = state + dt * (2.0 * jnp.tanh(input) - 0.5 * state)
        return next_state, next_state

    return Leaf(apply, init=init)

controller = nn.Linear(8) >> stack(residual(nn.GRU(8)), n=2) >> nn.Linear(1)
rollout = batch(scanned(closed_loop(
    serial(controller=controller, plant=FirstOrderPlant()),
    output_spec=jnp.zeros((1,)),
)))

refs = 0.5 * jnp.ones((8, 40, 1))
ctrl_trainer = train_step(
    rollout.with_input(refs).parameterize(rng=jax.random.PRNGKey(10)),
    mse,
    optax.adam(2e-2),
)
trained_rollout, ctrl_aux = split_aux(trained(ctrl_trainer)(
    input=jnp.broadcast_to(refs, (60, *refs.shape)),
    target=jnp.broadcast_to(refs, (60, *refs.shape)),
))
assert ctrl_aux.loss[-1] < ctrl_aux.loss[0] * 0.2
print(rollout.with_input(refs).tree_view())
```

Physical plants, feedback delays (`closed_loop`), recurrent neural controllers, time rollouts (`scanned`), setpoint batches (`batch`), and BPTT optimization (`train_step`) compose in one expression and expose their full member hierarchy via `.tree_view()`.

## Further reading

- The [authoring guide](authoring.md) covers Node boundary placement and decomposition patterns across control, neural architectures, training, and RL.
- The [handbook](handbook.md) documents every Node view, authoring hook, combinator, transform, `nn` layer, and `control` block.
- The [philosophy](philosophy.md) document explains why JAX transformations lead to single-`apply` FOOP.
- The [framework comparison](comparison.md) evaluates side-by-side implementations across JAX and PyTorch libraries.
