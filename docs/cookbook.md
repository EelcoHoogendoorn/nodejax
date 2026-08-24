# The NodeJAX Cookbook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/nodejax/blob/main/docs/cookbook.ipynb)

This cookbook builds up from a minimal node to deep architectures, worked training loops, and vectorized transforms. Every snippet runs as written, and each section builds on the previous one.

---

## 1. Defining a First Node

```python
import jax
import jax.numpy as jnp
import optax
from nodejax import node, Leaf, Struct, Aux, train_step, trained, Composite
from nodejax import batch, ensemble, stack, nn

@node
def Gain():
    def param(scale: float) -> Struct:
        return Struct(scale=scale)

    def apply(param: Struct, input: float) -> float:
        return param.scale * input

    return Leaf(apply, param=param)

model = Gain().parameterize(scale=2.0)
assert model.apply(3.0) == 6.0
```

A node is defined by up to three pure functions. `Gain` uses two:
* `param`: The parameter constructor. Its signature declares what the caller must supply (`scale`).
* `apply`: Computes the output given the parameter PyTree and input.

The `@node` decorator records the construction arguments (allowing future re-specialization) and automatically assigns a snake_case name matching the factory (`gain`).

---

## 2. Cyclic and stateful Nodes

```python
@node
def Integrator():
    def param(gain: float = 1.0) -> Struct:
        return Struct(gain=gain)

    def init(param: Struct) -> jax.Array:
        return jnp.asarray(0.0)

    def apply(param: Struct, state: jax.Array, input: float) -> tuple[jax.Array, float]:
        new_state = state + param.gain * input
        return new_state, new_state

    return Leaf(apply, param=param, init=init)

model = Integrator().parameterize(gain=1.0).initialize()

# Each step returns a new immutable successor binding the updated state:
model, y = model(5.0)
assert y == 5.0

# .scan runs over an entire sequence, continuing from the current state:
final_model, trajectory = model.scan(jnp.ones(10))
assert trajectory[-1] == 15.0
```

Adding `init` makes a node **cyclic**:
* `initialize()` constructs and binds initial state (a **state-full** or S-Node).
* Bound nodes are immutable. Calling a stateful node returns a **successor** `(new_model, output)` binding the updated state, without mutating the original object.
* `model.scan(...)` compiles with `jax.lax.scan` to run the model over a sequence.

---

## 3. Composing Pipelines with `>>`

```python
pipe = Gain().parameterize(scale=2.0) >> Integrator().parameterize(gain=1.0)
model = pipe.initialize()

final_model, trajectory = model.scan(jnp.ones(10))
assert trajectory[-1] == 20.0

# Parameters and state are structured PyTrees keyed by member name:
assert model.param.gain.scale == 2.0
assert final_model.state.integrator == 20.0
```

The `>>` operator chains nodes sequentially:
* Member outputs flow into downstream inputs.
* Parameters and states are composed into `Struct`s keyed by member name.
* Intermediate state threading is derived automatically from the pipeline structure.

---

## 4. Dynamic Shapes and PRNG Keys

```python
@node
def Linear(out_features: int):
    def param(node, rng) -> Struct:
        in_features = node.input.shape[-1]
        w = jax.random.normal(rng.next(), (in_features, out_features)) / jnp.sqrt(in_features)
        b = jnp.zeros(out_features)
        return Struct(w=w, b=b)

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)

relu = Leaf(lambda input: jnp.maximum(input, 0.0), name='relu')

X = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
net = Linear(8) >> relu >> Linear(1)

# with_input binds the input shape spec; parameterize provides the root PRNG key:
model = net.with_input(X).parameterize(rng=jax.random.PRNGKey(1))
assert model.param.linear.w.shape == (4, 8)
assert model.param.linear_2.w.shape == (8, 1)
assert model.apply(X).shape == (32, 1)
```

Two reserved authoring parameters:

* `node`: Grants access to resolved definition metadata, such as `node.input.shape`. `with_input(X)` binds the outer node's input shape; composition projects that evidence to members when parameterization or initialization walks the graph.
* `rng`: Declares that this function needs randomness. It receives a
  scope-local `KeyStream`; call `rng.next()` whenever a new key is needed. The
  public caller supplies one `rng=key` for the Node.

---

## 5. Training Inside the Algebra

```python
def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)

y = X @ jnp.ones((4, 1))
steps = 300
train_x = jnp.broadcast_to(X, (steps, *X.shape))
train_y = jnp.broadcast_to(y, (steps, *y.shape))

# 1. train_step turns a model into an optimization node:
trainer = train_step(model.initialize(), mse, optax.adam(1e-2))

# 2. trained() runs optimization to completion over batches:
fitted_model, aux = trained(trainer).apply(input=train_x, target=train_y)
assert aux.loss[-1] < 0.01

# The returned object is the trained model, ready to execute:
_, preds = fitted_model(X)
assert preds.shape == (32, 1)
```

`train_step` creates a standard node whose parameter is the starting weights and whose state holds the active weights and optimizer moments:
* `trained(trainer)` runs the optimization scan and returns the final trained model directly, with loss traces in `aux`.
* For custom training loops, `trainer.scan(...)` can be stepped in chunks with host-side logging and early stopping.

---

## 6. Stochasticity and Auxiliary Telemetry (`Aux`)

```python
@node
def Jitter():
    def param(sigma: float = 0.1) -> Struct:
        return Struct(sigma=sigma)

    def apply(param: Struct, input: jax.Array, rng) -> jax.Array:
        return input + param.sigma * jax.random.normal(rng.next(), input.shape)

    return Leaf(apply, param=param)

@node
def Probe():
    def apply(input: jax.Array) -> tuple[jax.Array, Aux]:
        return input, Aux(norm=jnp.linalg.norm(input))

    return Leaf(apply)

wired = Linear(8) >> Jitter() >> Probe() >> relu >> Linear(1)
model = wired.with_input(X).parameterize(rng=jax.random.PRNGKey(1),
                                         jitter=Struct(sigma=0.1))

# Forward pass with runtime PRNG key:
out, aux = model.apply(input=X, rng=jax.random.PRNGKey(4))
assert out.shape == (32, 1)
assert aux.probe.norm.shape == ()
```

* An authored `rng` parameter makes the call stochastic. The caller supplies a
  raw `rng=key`, and the function receives a scope-local `KeyStream`. RNG is
  separate from the input bundle.
* Returning `output, Aux(...)` emits auxiliary metrics without disrupting the primary dataflow wire. Auxiliary outputs collect into a member-keyed `Aux` PyTree.

---

## 7. Custom data flow with `Composite`

When dataflow is not a serial passage (e.g. residual bypasses or gating branches), use `Composite`:

```python
@node
def Highway(width: int):
    members = Composite(
        gate=Linear(width),
        body=Linear(width) >> relu >> Linear(width)
    )

    def apply(self, input: jax.Array) -> jax.Array:
        mix = jax.nn.sigmoid(self.gate(input))
        return mix * self.body(input) + (1.0 - mix) * input

    return members(apply)

model = Highway(4).with_input(X).parameterize(rng=jax.random.PRNGKey(2))
assert model.apply(X).shape == (32, 4)
```

`Composite(**members)` registers child sub-nodes. Inside `apply(self, input)`, calling `self.member_name(...)` steps the sub-node and manages its parameter and state slices automatically.

---

## 8. The Axis Transforms: `batch`, `ensemble`, `stack`

Transforms map dimensions onto parameters and state based on their declared contract roles:

```python
# 1. batch: Vectorizes data (vmap over inputs; parameters shared, state mapped)
per_sample = Linear(4) >> nn.BatchNorm(0.9)
batched = batch(per_sample).with_input(X).parameterize(rng=jax.random.PRNGKey(5)).initialize()
assert batched.param.linear.w.shape == (4, 4)       # Parameters shared
assert batched.state.batch_norm.mean.shape == (4,)  # Collective running stats
batched, out = batched(X)
assert out.shape == (32, 4)

# 2. ensemble: Vectorizes parameters (vmap over independent model instances)
population = ensemble(wired.with_input(X), n=4).parameterize(
    rng=jax.random.PRNGKey(3), jitter=Struct(sigma=0.1))
assert population.param.linear.w.shape == (4, 4, 8)  # 4 independent model parameter sets
out, aux = population.apply(input=X, rng=jax.random.PRNGKey(4))
assert out.shape == (4, 32, 1)                       # 4 distinct outputs
assert aux.probe.norm.shape == (4,)                  # Auxiliary outputs stacked per member

# 3. stack: Vectorizes depth (scan over layers; layer k feeds layer k+1)
deep = stack(Linear(8) >> relu, n=3).with_input(jnp.zeros(8)).parameterize(rng=jax.random.PRNGKey(6))
assert deep.param.linear.w.shape == (3, 8, 8)       # 3 stacked layers
assert deep.apply(jnp.ones(8)).shape == (8,)        # Sequentially evaluated depth
```

---

## 9. Statics, Reconfiguration (`specialize`), and Generics

```python
# Reconfigure an existing architecture deterministically:
wider_net = net.specialize(**{'linear.out_features': 16})
retrained = wider_net.with_input(X).parameterize(rng=jax.random.PRNGKey(8))
assert retrained.param.linear.w.shape == (4, 16)
assert retrained.param.linear_2.w.shape == (16, 1)  # Downstream layer shape re-derives automatically
```

### Composable Generics
Calling a node factory without specifying all static arguments yields a **`Generic`**:

```python
# Unbound layer (out_features open):
unit = Linear()
assert unit.generic

# Compose generic blueprints through transforms:
generic_tower = stack(unit, n=3)

# Specialize to resolve open static arguments:
concrete_tower = generic_tower.specialize(**{'layer.out_features': 8})
model = concrete_tower.with_input(jnp.zeros(8)).parameterize(rng=jax.random.PRNGKey(9))
assert model.param.w.shape == (3, 8, 8)
```

Generics allow neural architectures to be authored as open templates and composed into larger systems before concrete dimensions are fixed.

---

## Where Next

* [`docs/philosophy.md`](philosophy.md): The core architectural design doctrine, Python OOP vs. FOOP comparison, and JAX primitive alignment.
* [`docs/handbook.md`](handbook.md): The comprehensive technical reference manual for all contracts and transforms.
* [`docs/comparison.md`](comparison.md): Empirical benchmarks and architectural comparisons with Equinox, Flax, Haiku, and PyTorch.
