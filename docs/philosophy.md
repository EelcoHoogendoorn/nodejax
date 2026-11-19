# nodejax: Nodes That Compose Like JAX Functions

## Closure under composition and transformation

JAX closes pure functions under composition and under transformations such as `jit`, `vmap`, `lax.scan`, and `grad`, where transforming a function yields another function that composes and transforms again. `nodejax` lifts that closure from bare functions to computation-graph `Node`s.

Because JAX traces Python programs into computation graphs for XLA, `nodejax` calls its unit of organization a ***Node***. A `Node` is a vertex centered on one privileged computation (`apply`), bundled with the configuration (`statics`, `param`) and feedback state (`init`, `state`) that `apply` consumes. Rather than another neural network library, `nodejax` defines a general functional object model for JAX programs, serving equally as the foundation for neural architectures (`nodejax.nn`), physical plants, feedback controllers (`nodejax.control`), state estimators, KV caches, replay buffers, and optimizers. Composition connects `Node`s, transforms reinterpret `Node`s, and both return ordinary `Node`s.

| | Functions (`jax`) | `Node`s (`nodejax`) |
| :--- | :--- | :--- |
| Compose | `lambda x: jnp.sum(jax.nn.sigmoid(x))` | `nn.RNN(64) >> nn.relu` |
| Transform | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

## JAX primitives shape the Node contract

An authored `Node` declares up to three pure functions (`param`, `init`, and `apply`) inside a `@node` factory:

```python
@node
def LowPass(width: int, dt: float = 0.01) -> Node:
    def param(rate: float = 1.0, gain: float = 1.0) -> Struct:
        return Struct(rate=rate, gain=gain)

    def init() -> jax.Array:
        return jnp.zeros(width)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        next_state = state + dt * param.rate * (param.gain * input - state)
        return next_state, next_state

    return Leaf(apply, param=param, init=init)
```

Each role in that contract matches a boundary imposed by compilation, recurrence, or vectorization:

| Primitive | Boundary in JAX | Declaration in `LowPass` |
| :--- | :--- | :--- |
| `jax.jit` | Trace-time graph structure vs. post-compilation XLA buffers | `width` and `dt` are `statics` (recompiling if changed), while `param`, `state`, and `input` cross `jit` as traced PyTrees |
| `jax.lax.scan` | Rollout-invariant configuration vs. step carry and per-step signal | `rate` and `gain` stay fixed in `param`, while `state` and `input` occupy `(carry, input) -> (next_carry, output)` |
| `jax.vmap` | Distinct mapped and broadcast axes per role | `param`, `state`, and `input` remain separate PyTrees (`batch` broadcasts `param`; `ensemble` maps `param` and `state`) |

`param` is defined by compilation and temporal recurrence rather than by gradient descent. Unlike `statics`, `param` values vary after compilation without recompiling; unlike `state` and `input`, they stay invariant across the steps of a rollout. A linear layer's weights, a filter's `rate` and `gain`, and a motor plant's physical constants all belong in `param` whether swept with `vmap`, tuned by evolution, or differentiated on a selected subset with `jax.grad`.

## Transform products are ordinary Nodes

Closure holds because every transform produces a full `Node` ready to enter further pipelines and transforms without caller-supplied adapters. A scalar affine projection maps one vector to one scalar, reading its input width from upstream dataflow:

```python
@node
def Projection() -> Node:
    def param(node, rng) -> Struct:
        width = node.input.shape[-1]
        return Struct(
            w=jax.random.normal(rng.next(), (width,)) / jnp.sqrt(width),
            b=jnp.zeros(()),
        )

    def apply(param, input) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)
```

A multi-output linear layer is an ensemble of independent scalar projections:

```python
linear = ensemble(Projection(), n=3)
```

`ensemble` forwards the resolved input shape, adds the population axis to `param` and `state`, splits initialization keys across the three rows, and publishes the stacked output shape downstream. The resulting `linear` `Node` pipes through `>>`, maps under `batch`, and trains under `train_step` identically to a hand-written matrix layer.

## Functional objects and cyclic state

`nodejax` organizes programs through ***Functional Object-Oriented Programming*** (FOOP). Conventional Python module libraries populate statics, parameters, running statistics, and child modules into one instance inside eager `__init__`, then recover pure functions and PyTrees afterward through split/merge passes, ambient contexts, or runtime filtering. Python `class` syntax also centers on vertical inheritance, whereas computation graphs grow horizontally through sequential piping, feedback loops, axis mapping, and static specialization.

In FOOP, a `Node` is pure where it is authored; `apply` is the computation, while `param` and `init` build its immutable PyTree inputs. Binding `param` returns an immutable `PNode` PyTree, binding `state` returns an immutable `PSNode` PyTree, and stepping a `PSNode` returns a successor `PSNode` alongside its output without mutating the caller's value (`advanced, output = model(input_data)`).

In a computation graph, a feed-forward `Node` (`input -> output`) is acyclic, whereas a `Node` whose `apply` feeds `next_state` back into its own `state` input on the following step forms a directed feedback cycle. `nodejax` therefore calls a state-threading `Node` ***cyclic*** (`node.cyclic`) rather than "stateful"; neither a `Node` definition nor a `PSNode` snapshot holds mutable storage, and `cyclic` names the recurrent edge unrolled by `lax.scan`. An RNN hidden vector, `BatchNorm` moments, a motor's currents and velocity, a replay buffer, and Adam momentum all share this one `cyclic` transition, so composing controllers, plants, estimators, and optimizers yields one structured `state` PyTree with zero translation glue.

## State lifetimes stay attached to the owner

Sharing one `cyclic` contract leaves each carry free to reset on its own schedule. During chunked sequence training, an RNN hidden state resets per episode while running normalization statistics persist across episodes; in adaptive control, a plant restarts per trial while an online estimator keeps its belief.

`nodejax` attaches reset rules to the `Node` that owns the state (`state_reinit(nn.RNN(64), boundary='episode')` or a leaf `boundary={'episode': ...}` hook). An enclosing `scan(model, boundary='episode')` triggers matching boundaries across its subtree without inspecting member layouts, so the `cyclic` `Node` carries its reset policy unchanged across new rollouts and trainers.

## Generic Nodes decouple architecture from sizing

Fixing every dimension inside eager constructors forces outer builders to mirror and forward nested hyperparameters down constructor chains. Calling a `@node` factory with unbound static arguments returns a `Generic` blueprint that already pipes and transforms before sizes are chosen:

```python
block = residual(nn.LayerNorm() >> nn.Attention()) >> residual(nn.LayerNorm() >> nn.MLP())
tower = stack(block, n=12)

model = tower.specialize(**{'*.width': 768, '*.heads': 12, '*.ratio': 4})
```

Dataflow-implied dimensions such as input fan-in resolve later through `with_input(...)`, while open architectural statics resolve by path through `specialize(...)`, letting reusable architectural blocks live as ordinary values rather than constructor parameter plumbing.

## Optimization lives inside the same algebra

Training is a `Node` transform rather than a separate runtime outside the model abstraction:

```python
trainer = train_step(system, loss_fn, optax.adam(1e-3), trainable='controller')
```

When `system` couples a controller to a physical plant, `trainable` partitions `system.param` by path segment so non-optimized plant parameters and initial controller weights stay in `trainer.param`, while active controller weights, `system.state`, and optimizer state occupy `trainer.state`. Gradient-free population optimizers inhabit the same `cyclic` contract without calling `jax.grad`, and when updates are smooth, differentiating with respect to `trainer.param` differentiates through optimization itself. Separating `scan(step)` as a reusable `Node` transform lets inner adaptation, physical rollouts, ensembles, and outer meta-optimizers compose in one algebra.

## Further reading

- The [cookbook](cookbook.md) builds runnable programs from a minimal leaf through composition, transforms, and training.
- The [authoring guide](authoring.md) covers `Node` boundary placement and decomposition patterns across control, physics, neural architectures, and reinforcement learning.
- The [handbook](handbook.md) is the complete technical reference for views, authoring primitives, transforms, `nodejax.nn`, and `nodejax.control`.
- The [design document](design.md) covers internal `Def` lowering and T3 transform authoring.
- The [framework comparison](comparison.md) evaluates side-by-side implementations in Flax, Equinox, Haiku, PyTorch, Keras, and Brax.
