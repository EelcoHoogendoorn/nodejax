# NodeJAX: Nodes That Compose Like JAX Functions

## Closure under composition and transformation

NodeJAX builds functional objects in JAX. JAX closes pure functions under composition and under transformations such as `jit`, `vmap`, `lax.scan`, and `grad`, where transforming a function yields another function that can be composed and transformed again. NodeJAX lifts that closure from bare functions to structured computation-graph units.

Because JAX traces Python programs into computation graphs for XLA, NodeJAX calls its unit of organization a ***Node***. A `Node` is first and foremost a unit of computation (its privileged `apply` function), bundled with the configuration (`statics`, `param`) and feedback state (`init`, `state`) that `apply` consumes. Rather than another neural network library, NodeJAX defines a general functional object model for JAX programs. That model makes a strong foundation for neural architectures (`nodejax.nn`), while applying identically to physical plants, industrial feedback controllers (`nodejax.control`), state estimators, KV caches, replay buffers, and optimizers. Composition connects Nodes, transformations reinterpret Nodes, and both return ordinary Nodes.

| | JAX functions | NodeJAX Nodes |
| :--- | :--- | :--- |
| Closure under composition | `f = lambda x: sum(sigmoid(x))` | `node = nn.RNN(64) >> nn.relu` |
| Closure under transforms | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

## JAX primitives dictate the Node contract

An authored Node declares up to three functions (`param`, `init`, and `apply`) inside a `@node` factory:

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

Each slot in that contract records a distinction that compilation, recurrence, or vectorization requires:

| Compiler / transform axis | What JAX requires | What `LowPass` declares |
| :--- | :--- | :--- |
| Compilation (`jax.jit`) | Trace-time graph structure separated from post-compilation XLA values | `width` and `dt` are statics (recompiling if changed; an unbound `width` stays open in a `Generic`), while `param`, `state`, and `input` cross `jit` as traced PyTrees |
| Recurrence (`jax.lax.scan`) | Rollout-invariant configuration separated from step-to-step carry and per-step signal | `rate` and `gain` stay fixed in `param`, while `state` and `input` occupy `(carry, input) -> (next_carry, output)` in `apply` |
| Vectorization (`jax.vmap`) | Mapped and broadcast axes chosen independently per role | `param`, `state`, and `input` remain distinct PyTrees (`batch` broadcasts `param`; `ensemble` maps `param` and `state`) |

`param` is defined by its role in compilation and temporal recurrence rather than by gradient descent. Neural network frameworks routinely equate parameters with differentiable weights, yet a physical plant's winding resistance or `LowPass`'s `rate` and `gain` belong in `param` even when never differentiated. Unlike `statics`, `param` values can change after compilation without recompiling; unlike `state` and `input`, they stay invariant across the steps of a rollout. Sweeping `param` with `vmap`, randomizing plant physics across domains, tuning gains by evolutionary search, or taking gradients with `jax.grad` are downstream choices made after the contract is defined.

Because the Node already separates statics, parameters, feedback state, and call inputs, transforms act on declared roles without reconstructing meaning from array shapes or object attributes.

## Transform products are ordinary Nodes

A transform preserves closure only when its result keeps the full Node contract, ready to enter further pipelines and transforms without caller-supplied adapters.

A scalar affine projection maps one vector to one scalar, reading its input width from the resolved input specification:

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

`ensemble` forwards the resolved input shape to each projection, adds the population axis to parameter construction, splits initialization keys across the three rows, maps any state along the same axis, and publishes the stacked output specification to downstream Nodes. The resulting `linear` Node pipes through `>>`, maps over data under `batch`, and trains under `train_step` identically to a hand-written matrix layer.

## Functional Object-Oriented Programming (FOOP)

NodeJAX organizes programs through ***Functional Object-Oriented Programming*** (FOOP), building hierarchical composition, attribute inspection, and reusable definitions around a single privileged computation (`apply`) over immutable PyTree values.

Conventional object-oriented libraries treat an object as a mutable noun with an instance dictionary and arbitrary peer methods (`reset`, `step`, `forward`, `update`), populated inside one eager `__init__`. Architectural statics, post-compilation parameters, running statistics, child modules, and caches all land in `self.__dict__` because Python provides a single construction hook, even though those values have distinct compilation lifetimes and axis rules. Before JAX can compile, vectorize, or scan the object, the library has to partition that dictionary back into pure functions and PyTree collections through split/merge passes, external state tables, or ambient contexts.

Python class syntax also reserves its primary header for vertical class inheritance, whereas computation graphs combine Nodes horizontally through sequential piping, feedback loops, axis mapping, and static specialization. NodeJAX provides `derive` for extending leaves, while dedicating `@node` factory headers to architectural statics and collaborator Nodes.

In FOOP, a `Node` is pure at the boundary where it is authored; `apply` is the computation, while `param` and `init` construct the PyTree inputs that `apply` expects. Binding parameters returns an immutable `PNode` PyTree, binding state returns an immutable `PSNode` PyTree, and stepping a state-bound Node returns a new `PSNode` holding the successor state:

```python
advanced, output = model(input_data)
```

## Cyclic Nodes and one state model across domains

In a computation graph, a feed-forward Node (`input -> output`) is an acyclic vertex, whereas a Node whose `apply` transition feeds `next_state` back into its own `state` input on the next step (`apply(param, state, input) -> (next_state, output)`) forms a directed feedback cycle. NodeJAX therefore calls a state-threading Node ***cyclic*** (`node.cyclic`) rather than "stateful." Imperative OOP calls an object stateful because it mutates hidden storage inside itself across calls; in NodeJAX, neither an unbound `Node` nor a `PSNode` `(Def, param, state)` snapshot holds mutable storage, and calling a `PSNode` leaves the original snapshot untouched. `cyclic` names the recurrent feedback edge in the compute graph that `lax.scan` unrolls across time.

An RNN hidden vector, BatchNorm running moments, Adam momentum tensors, a motor's rotor velocity and winding currents, a replay buffer, and an autoregressive KV cache all share this cyclic transition. Composing a neural controller, a state estimator, a physical plant, and an optimizer produces one structured state PyTree with no translation between module buffers, scan carries, optimizer stores, and simulator contexts.

## State lifetimes stay attached to the state

Sharing one transition contract does not force every state slot to share one reset schedule. During chunked sequence training, an RNN hidden state resets at the start of each episode while normalization statistics persist across episodes; in adaptive control, a plant restarts per trial while an online estimator carries its belief forward.

NodeJAX attaches reset behavior to the Node that owns the state. `state_reinit(RNN(64), boundary='episode')` or a leaf `boundary={'episode': ...}` hook declares how that state responds when a named boundary occurs. An enclosing `scan(model, boundary='episode')` announces the boundary at the start of each call without inspecting member layouts, and the cyclic Node carries its lifetime policy unchanged when composed into new rollouts or trainers.

## Generic Nodes decouple architecture from sizing

Fixing every dimension and flag at construction time forces enclosing composites to mirror and forward every nested hyperparameter. In NodeJAX, calling a `@node` factory with unbound static arguments returns a `Generic` construction record that can already be piped and transformed before concrete sizes exist:

```python
block = residual(nn.LayerNorm() >> nn.Attention()) >> residual(nn.LayerNorm() >> nn.MLP())
tower = stack(block, n=12)

model = tower.specialize(**{'*.width': 768, '*.heads': 12, '*.ratio': 4})
```

Dataflow-implied dimensions such as input fan-in resolve later from `with_input(...)`, while open architectural statics resolve by path pattern through `specialize(...)`. Libraries can publish complete architectural patterns without locking them to one problem's dimensions or burying them in configuration plumbing.

## Optimization lives inside the same algebra

Optimization is one downstream application of the `cyclic` contract rather than a special outer runtime:

```python
trainer = train_step(system, loss_fn, optax.adam(1e-3), trainable='controller.*')
```

When `system` combines a controller with a physical plant, not every `param` is optimized. `trainable` partitions `system`'s `param` PyTree by path so non-optimized plant parameters and initial controller parameters remain in `trainer.param`, while the evolving controller parameters, `system`'s own state, and the optimizer's moments occupy `trainer.state`. Gradient-free optimizers such as differential evolution inhabit the exact same `cyclic` contract by carrying candidate populations in `state` without calling `jax.grad`, and when the update step is differentiable, differentiating with respect to `trainer.param` differentiates through the optimization trajectory itself.

JAX's `lax.scan(step, initial, sequence)` couples declaring a recurrence with immediately executing it. Because a Node contract already distinguishes `state` from `input`, NodeJAX separates `scan(step)` as a reusable Node transform and supplies initial state and sequence inputs later. Inner task adaptation, population ensembles, physical rollouts, and outer meta-optimizers all compose as ordinary Nodes.

## Explicit randomness without key plumbing

Stochastic functions in NodeJAX declare `rng` in their signature and draw independent keys from a scope-local `KeyStream`:

```python
def param(node, rng) -> Struct:
    shape = node.input.shape
    w = jax.random.normal(rng.next(), shape)
    b = jax.random.normal(rng.next(), shape)
    return Struct(w=w, b=b)
```

At the public boundary, the caller supplies a single `rng=key` when a role consumes randomness, and deterministic calls reject stray keys. Composites route child streams to stochastic members and axis transforms split streams across mapped elements automatically, keeping entropy explicit at call boundaries without manual key counting inside model code.

## Further reading

- The [cookbook](cookbook.md) builds runnable programs from a minimal leaf through composition, transforms, and training.
- The [authoring guide](authoring.md) explains where to place Node boundaries and how programs decompose across control, neural architectures, training, and reinforcement learning.
- The [handbook](handbook.md) is the complete technical reference for contracts, views, authoring primitives, transforms, `nn`, and `control`.
- The [design document](design.md) covers internal `Def` lowering and T3 transform authoring.
- The [framework comparison](comparison.md) evaluates side-by-side implementations in Flax, Equinox, Haiku, PyTorch, Keras, and Brax.
