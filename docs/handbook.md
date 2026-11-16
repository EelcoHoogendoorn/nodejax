# NodeJAX Handbook

Technical reference for NodeJAX views, authoring primitives, combinators, transforms, domain modules (`nodejax.nn` and `nodejax.control`), and tree utilities. For boundary placement and decomposition patterns across domains, see the [authoring guide](authoring.md). For internal `Def` lowering and T3 transform construction, see the [design document](design.md).

## 1. Views and binding stages

NodeJAX separates architectural construction, shape resolution, parameter creation, and state initialization into distinct immutable values:

```text
Generic  --(specialize)-->  Node  --(with_input)-->  Node  --(parameterize)-->  PNode  --(initialize)-->  PSNode
```

`Node`, `PNode`, and `PSNode` inherit shared inspection and configuration operations from `BaseNode`. `Generic` represents an unspecialized `@node` call with one or more unbound static arguments (`is_generic(x)` tests for a `Generic`, and `REQUIRED` marks mandatory unbound fields internally).

| View | Bound data | JAX PyTree leaves | Calling `view(...)` |
| :--- | :--- | :--- | :--- |
| `Generic` | Open factory + partial statics | Not a JAX PyTree | Raises until `specialize(...)` supplies remaining `unbound()` statics |
| `Node` | Definition only (`param` unbound) | Empty static PyTree | `node.apply(param, ...)` runs with explicit `param` |
| `PNode` | `param` bound (stateless or uninitialized) | `param` leaves | Stateless `pnode(input)` returns `output`; cyclic `PNode` requires `.initialize(...)` first |
| `PSNode` | `param` and `state` bound | `(param, state)` leaves | `psnode(input)` steps once and returns `(next_psnode, output)` |

### Shared `BaseNode` properties and methods

| Attribute / method | Behavior |
| :--- | :--- |
| `.name`, `.tags`, `.members` | Node identifier string, `frozenset` of tag strings, and `Struct` of child member views |
| `.parametric`, `.cyclic` | Booleans indicating whether the Node declares `param` or `state` |
| `.bound`, `.state_bound` | Booleans indicating parameter binding (`PNode` / `PSNode`) and state binding (`PSNode`) |
| `.node`, `.generic` | Underlying unbound `Node` view and reconstructing `Generic` (if built by `@node`) |
| `.with_input(input_spec, *, bundle=...)` | Resolves input-dependent shapes from an example array, `Struct`, or `spec(...)` without binding runtime data |
| `.specialize(**overrides)` | Reruns construction with updated static arguments matched by exact path or glob (`'*.train'`, `'layer.width'`), returning a fresh unbound `Node` |
| `.parameterize(param_input=..., **fields)` | Constructs parameters (accepting `rng=key` when stochastic) and returns a `PNode` (or `PSNode` for a stateless Node that needs no state) |
| `.describe()`, `.tree_view(...)`, `.summary(...)`, `.statics_by_path()` | Structural and parameter/state inspection helpers (see Section 7) |

### Stage-specific view methods

| View method | Behavior |
| :--- | :--- |
| `Node.bind(param=(), *, state=...)` | Attaches pre-built `param` (and optional `state`), returning a `PNode` or `PSNode` |
| `Node.init(param, state_input=..., *, input=..., **fields)` | Computes initial `state` for `param` without wrapping in `PSNode` |
| `PNode.init(...)` / `PNode.initialize(...)` | `.init(...)` returns the raw `state` PyTree; `.initialize(...)` returns a bound `PSNode` |
| `PNode.bind(param=..., *, state=...)` | Replaces bound `param` or attaches `state` |
| `PSNode.reset(*args, **fields)` | Recomputes initial state from bound `param` and returns a fresh `PSNode` |
| `PSNode.scan(*args, **fields)` | Unrolls the `PSNode` across the leading sequence axis of its inputs and returns `(final_psnode, stacked_output)` |
| `PSNode.pnode` | Returns the underlying `PNode` with `state` dropped |

```python
import jax
import jax.numpy as jnp
from nodejax import nn

net = nn.Linear(16) >> nn.gelu >> nn.Linear(4)
model = net.with_input(jnp.zeros(8)).parameterize(rng=jax.random.PRNGKey(0))
output = model(jnp.ones(8))

eval_net = model.specialize(**{'*.bias': True}).bind(model.param)
```

## 2. Records, specs, randomness, and auxiliary outputs

### `Struct`, `Aux`, and `spec`

`Struct(**fields)` is an immutable keyword-addressed JAX PyTree container supporting attribute access (`s.x`), dictionary-style membership (`'x' in s`), shallow functional updates (`s.replace(x=new_x)`), and leading-axis indexing (`s[0]`).

`Aux(**fields)` is a `Struct` subtype that carries telemetry and auxiliary losses alongside primary outputs without altering downstream input shapes. `split_aux(output)` unpacks any call result into `(primary_output, aux)` (returning an empty `Aux()` when none was emitted), and `drop_aux` either strips `Aux` from a concrete output value (`drop_aux(out)`) or wraps a `Node` so its `Aux` stream is discarded (`drop_aux(node)`).

`spec(shape=(), dtype=jnp.float32)` builds a `jax.ShapeDtypeStruct` placeholder for `.with_input(...)` when no concrete array is needed. `spec_of(tree)` converts an array PyTree into `ShapeDtypeStruct` leaves, and `materialize(tree)` allocates zero arrays matching a spec PyTree.

### Explicit PRNG streams (`KeyStream`)

Any authored `param`, `init`, or `apply` function that draws random numbers declares `rng` in its signature. Inside that function, `rng` is a scope-local `KeyStream` whose `rng.next()` yields independent JAX PRNG keys. At the public call boundary (`parameterize`, `initialize`, `apply`, or `PSNode(...)`), the caller passes a single `rng=jax.random.PRNGKey(seed)` keyword argument. Deterministic calls reject unexpected `rng` arguments, and stochastic calls reject missing ones.

### Exported type aliases

`nodejax` exports `PyTree`, `Param`, `State`, `Input`, `Output`, `ParamFn`, `InitFn`, `ApplyFn`, `LossFn`, and `StaticTree` for type annotations on authored factories, transforms, and loss functions.

## 3. Authoring Nodes

### `@node` and `ambient`

`@node` (or `@node(name='custom')`) decorates a factory function whose parameters represent compile-time statics or collaborator Nodes. Calling a `@node` factory with all required arguments returns a `Node`; omitting required static arguments (without defaults) returns a `Generic` that can be composed and later completed via `.specialize(...)` or an `ambient(**statics)` context block:

```python
from nodejax import Leaf, Node, Struct, ambient, node

@node
def Affine(width: int) -> Node:
    def param(node, rng) -> Struct:
        in_dim = node.input.shape[-1]
        return Struct(
            w=jax.random.normal(rng.next(), (in_dim, width)) / jnp.sqrt(in_dim),
            b=jnp.zeros(width),
        )

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return input @ param.w + param.b

    return Leaf(apply, param=param)

with ambient(width=32):
    layer = Affine()
```

### `Leaf`

`Leaf(apply, *, param=None, init=None, name=None, apply_input_spec=None, methods={}, tags=(), boundary={})` constructs an atomic Node with no child members.

Role functions declare only the reserved names they read:

| Reserved argument | Available in | Value |
| :--- | :--- | :--- |
| `param` | `init`, `apply`, `methods` | Bound parameter PyTree |
| `state` | `apply`, `methods` | Current state PyTree; `apply` returns `(next_state, output)` (or `(next_state, (output, Aux(...)))`) |
| `input` | `init`, `apply` | Primary runtime input (in `init`, marks an input-primed initializer) |
| `node` | `param`, `init` | Resolved `Node` metadata (such as `node.input.shape` and `node.input.dtype`) |
| `rng` | `param`, `init`, `apply`, `methods` | Scope-local `KeyStream` |

Any non-reserved parameter name declared on `param`, `init`, or `apply` becomes a named constructor or call input field (`def apply(param, observation, command): ...`).

`methods` maps method names to functions over `(param, state, rng, *args, **kwargs)` that bind onto `PNode` and `PSNode` views as read-only operations. `boundary` maps boundary names (such as `'episode'`) to reset hooks `(carried, init, decided) -> state` triggered when an enclosing `scan(..., boundary='episode')` starts a call.

### `derive`

`derive(parent, *, apply=None, init=None, param=None, name=None, apply_input_spec=None, methods={}, tags=None)` produces a new leaf `Node` from an unbound leaf `parent`:

- Omitted `param`, `init`, or `apply` hooks are inherited from `parent`.
- When both `parent` and `derive` supply `param` or `init`, their returned `Struct` keys must be disjoint and merge into one flat `Struct`.
- Inherited transitions update their own state fields while preserving added state fields; an explicit replacement `apply` over merged state returns the complete state `Struct`.
- `methods` and `tags` merge by union, with child methods overriding matching names.

### `Composite` and `Wrapper`

`Composite(**members)(apply, *, param=None, init=None, apply_input_spec=None, name=None, methods={})` wires multiple named child Nodes. Its `param`, `state`, and `Aux` PyTrees nest under the declared member names (`param.encoder`, `param.decoder`).

`Wrapper(member=inner)(apply, *, init=None, name=None, tags=None, boundary={}, methods={}, rng_from=None, input_spec=None)` wraps a single named child Node while keeping `param`, `state`, and `Aux` flat (delegating directly to `inner` without adding an extra member-name level).

Inside `Composite` or `Wrapper` `apply(self, ...)`:

- `self.<member>` returns a live bound view of that member at its current state (`self.<member>.param`, `self.<member>.state`, plus any custom methods on that member).
- `self.<member>(x)` executes the member, threads child `rng` and `Aux` automatically, writes any updated member state back into `self`, and returns the primary output.
- `self.<member>.bind(state=s)` or `self.<member>.reset(...)` updates the member's state slot in `self` and returns the rebound member view.
- `self.sow(key=value, ...)` records composite-level `Aux` fields.

An authored `init(self, ...)` initializes member states imperatively through `self.<member>.bind(state=...)` and `self.<member>.reset(...)` when one member's initial state depends on another member or on runtime start data; untouched cyclic members initialize from their own defaults.

```python
from nodejax import Composite, Wrapper, node

@node
def GatedResidual(transform: Node, gate: Node) -> Node:
    members = Composite(transform=transform, gate=gate)

    def apply(self, input: jax.Array) -> jax.Array:
        update = self.transform(input)
        weight = jax.nn.sigmoid(self.gate(input))
        self.sow(gate_mean=jnp.mean(weight))
        return input + weight * update

    return members(apply)
```

## 4. Combinators and wiring transforms

| Function / operator | Signature | Behavior |
| :--- | :--- | :--- |
| `a >> b` | `BaseNode \| Generic \| Callable` | Chains stages sequentially; plain callables lift to stateless `Leaf` Nodes, and `>>` flattens chained anonymous pipes |
| `serial` | `serial(**members)` | Sequential pipeline with explicit member names (`serial(embed=..., core=..., head=...)`) |
| `parallel` | `parallel(**members)` | Broadcasts the same input to every named member and returns `Struct(**outputs)` |
| `sum_junction` | `sum_junction(**members)` | Feeds `input.<name>` to each named member and sums their outputs tree-wise (`jax.tree.map`) |
| `residual` | `residual(body)` | Evaluates `input + body(input)` tree-wise around a shape-preserving `body` |
| `tie` | `tie(pipe, source, *aliases)` | Removes `*aliases` from `pipe`'s `param` tree and routes `param.<source>` into every aliased member at runtime |
| `externalize` | `externalize(inner, member=None, at_init=None, *, field=None)` | Moves `member`'s (or the whole Node's when `member=None`) parameter tree out of `param` into a runtime call input named `field` (defaulting to `member` or `'param'`), using `at_init` if needed during shape/init probing |
| `taps` | `taps(inner)` | Wraps a composite `inner` so its output becomes `Struct(output=..., taps=Struct(**per_member_outputs))` |
| `at` | `at(inner, field)` | Applies `inner` to ` getattr(input, field)` of a `Struct` input and writes the result back via `input.replace({field: out})` |

## 5. Axis, iteration, learning, and structure transforms

### Axis and execution transforms

| Transform | Signature | `param` axis | `state` axis | `input` / `output` axis |
| :--- | :--- | :--- | :--- | :--- |
| `batch` | `batch(sample, n=None, axis='batch')` | Shared | Mapped (`vmap`) | Mapped along leading axis (enforcing length `n` when given); provides named `axis` for cross-sample ops like `BatchNorm` |
| `unbatched` | `unbatched(inner, axis='batch')` | Unchanged | Unchanged | Evaluates `inner` under a size-1 `axis_name` axis and squeezes it out so single-sample calls satisfy `lax.pmean` consumers |
| `ensemble` | `ensemble(member, n, axis='ensemble')` | Mapped (`n`) | Mapped (`n`) | Broadcasts input across `n` independent members and stacks outputs on axis 0 |
| `reduce` | `reduce(fn)` | Stateless `PNode` | Stateless | Applies `fn` (such as `jnp.mean` or `jnp.sum`) across axis 0 of every output leaf and emits `Aux(population=input)` |
| `stack` | `stack(layer, n)` | Stacked (`n`) | Stacked (`n`) | Chains `n` identically structured layers sequentially via `lax.scan` over depth |
| `remat` | `remat(inner, *, policy=None, prevent_cse=True)` | Unchanged | Unchanged | Wraps `inner`'s execution in `jax.checkpoint` for activation rematerialization |

### Iteration and state transforms

| Transform | Signature | State lifetime and output |
| :--- | :--- | :--- |
| `scan` | `scan(step, record=False, boundary=None, n=None)` | Cyclic sequence Node: carries `step`'s state across leading time axis (length `n` if given), running `boundary` hooks at call start; emits per-step states on `Aux(state=...)` when `record=True` |
| `scanned` | `scanned(step, record=False, n=None)` | Stateless-interface rollout: initializes `step`'s state fresh per call, scans over the sequence, and discards final carry; emits per-step states on `Aux(state=...)` when `record=True` |
| `carried` | `carried(step, n=None)` | Initializes `step` fresh, scans over the sequence, and returns `step` bound as a `PSNode` to its final state |
| `repeat` | `repeat(layer, n)` | Feeds output back as input `n` times with shared parameters, carrying state and returning the final output |
| `repeated` | `repeated(step, n)` | Initializes state internally, feeds output back `n` times, and stacks all `n` outputs into a trajectory |
| `iterated` | `iterated(step, n)` | Applies cyclic `step` `n` times to the *same* input while threading state, returning `(final_state, last_output)` |
| `cyclic` | `cyclic(step)` | Turns a stateless transition `apply(state_like_first_arg, ...)` whose output is `next_state` (or `(next_state, output)`) into a cyclic `Node` carrying that first input as `state` |

### Learning and test-time adaptation

| Transform / helper | Signature | Behavior |
| :--- | :--- | :--- |
| `train_step` | `train_step(model, loss_fn, tx, *, trainable=None)` | Cyclic trainer Node: `param` holds initial weights; `state` holds active weights, model state, and optimizer state (`tx` is an Optax transformation or optimizer Node). `loss_fn(output, target, *, aux=...)` (or a loss `Node`) produces scalar loss. `trainable` restricts updates by member name or path predicate. Exposes `.params()` and `.trained()` view methods |
| `trained` | `trained(step)` | Scans a `train_step` Node over a sequence of training batches and returns `step.trained()` bound to final learned weights and state |
| `finetune` | `finetune(step)` | Episode learner over `train_step`: adapts from `param` on `support` batches via `trained(step)` and evaluates the adapted model on `query` |
| `supervised_ttt` | `supervised_ttt(step)` | Adapts `train_step` online per call on `(input, target)` pairs while emitting pre-update predictions |
| `next_step_ttt` | `next_step_ttt(step)` | Self-supervised online learner over `train_step` where consecutive sequence elements form `(x[:-1], x[1:])` pairs |
| `reconstruction_ttt` | `reconstruction_ttt(step)` | Self-supervised online learner over `train_step` where each input serves as its own reconstruction target |
| `next_step`, `reconstruction` | `next_step()`, `reconstruction()` | Stateless `PNode` stream formatters producing `Struct(input=..., target=...)` pairs for sequence prediction or autoencoding |

Additional optimizer building blocks live in `nodejax.transforms.learning`: `optimizer(tx, name='optimizer')` wraps an Optax transformation as a cyclic `PNode`, `learned_sgd(lr0)` creates SGD with per-leaf learnable step sizes, `map_loss_target(loss_fn, target_fn)` adapts loss targets while preserving `aux` inspection, `opt_reinit(inner, boundary)` resets optimizer moments on boundaries while keeping learned weights, and `ttt(step)` wraps a `train_step` for single-sample online prediction-then-update.

### State policy, gradient policy, and structure surgery

| Transform | Signature | Behavior |
| :--- | :--- | :--- |
| `state_reinit` | `state_reinit(inner, boundary='episode')` | Attaches a boundary hook that resets `inner`'s entire state subtree to `init` whenever an enclosing `scan` triggers `boundary` |
| `freeze` | `freeze(inner, state)` | Fixes `inner`'s state to `state` on every call and removes `state` from the external contract |
| `tree_freeze` | `tree_freeze(node, frozen=None, tag=None)` | Applies `freeze` selectively across `node` using a sparse `Struct` of state subtrees (`frozen`) or member tag (`tag`) |
| `detach` | `detach(inner)` | Applies `lax.stop_gradient` to `inner`'s parameters before `apply` |
| `tree_detach` | `tree_detach(node, name)` | Applies `detach` to members matching `name` (string or predicate) |
| `tree_filter` | `tree_filter(state, name)` | Extracts a sparse `Struct` containing only state subtrees under members matching `name` |
| `map_members` | `map_members(node, fn)` | Rewrites a `Node` tree bottom-up, applying `fn(path, member)` at every composite level |

## 6. Domain modules

### `nodejax.nn`

Import via `from nodejax import nn` or `import nodejax.nn as nn`.

| Category | Exports |
| :--- | :--- |
| Projections & shapes | `Linear(n_out, bias=True, *, weight_init=..., bias_init=...)`, `Projection(*, weight_init=..., bias_init=...)`, `Embed(vocab, hidden)`, `Unembed(vocab, hidden)`, `OneHot(classes)`, `flat`, `Reshape(shape)` |
| Activations | `relu`, `gelu`, `silu`, `elu`, `leaky_relu`, `softplus`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `identity` |
| Normalization | `LayerNorm(eps=1e-5)`, `RMSNorm(eps=1e-6)`, `L2Norm(eps=1e-8)`, `BatchNorm(momentum, eps=1e-5, axis='batch', train=True)`, `Whiten(momentum=0.1, eps=1e-2, axis='batch', train=True)` |
| Spatial / 2D | `Conv(features, kernel=3, stride=1)`, `ConvTranspose(features, kernel=3, stride=2)`, `MaxPool(window=2, stride=None, padding='VALID')`, `AvgPool(window=2, stride=None, padding='VALID')`, `GlobalAvgPool()`, `Upsample(scale=2, method='nearest')`, `Downsample(scale=2, method='nearest')` |
| Attention & FFN | `Attention(heads, causal=False)`, `TransformerBlock(width, heads, ratio, causal=False)`, `tokens()`, `PosEmbed()`, `MLP(width, ratio)`, `SwiGLU(width, ratio)`, `MoE(hidden, experts)` |
| Recurrent cells | `RNN(hidden)`, `GRU(hidden)`, `MinGRU(hidden)`, `LSTM(hidden)` |
| Regularization & mode | `Dropout(rate, train=True)`, `DropPath(rate, train=True)`, `GaussianNoise(std, train=True)`, `EMA(tau, warm=False)`, `eval_mode(model)` |

`eval_mode(psnode)` specializes `train=False` across all training-mode layers (such as `BatchNorm` and `Dropout`), freezes running state subtrees (such as `BatchNorm` moments), and binds the learned parameters as a stateless inference `PNode`.

### `nodejax.control`

`control`, `closed_loop`, `observed_loop`, and `sum_junction` are exported directly from `nodejax`; the full block set lives in `nodejax.control`.

| Category | Exports |
| :--- | :--- |
| Feedback loops | `closed_loop(pipe, output_spec=0.0)`, `feedback(pipe, output_spec=0.0)`, `observed_loop(pipe, belief_spec, output_spec=0.0)`, `sum_junction(**members)` |
| Controllers | `PID(dt, dwrap=None)`, `PD(dt)` |
| Signal & nonlinear blocks | `Gain()`, `Integrator()`, `Diff(dt)`, `Delay(warm=False)`, `EMA(dt, warm=False)`, `MovingAverage(window, warm=False)`, `Blend(dt)`, `Clamp()`, `ClampNorm()`, `Deadband()`, `Quantize(resolution)`, `RateLimit(dt)` |
| Plants | `FirstOrder(dt, warm=False)`, `StateSpace()`, `Walker()` |

The two `EMA` factories parameterize retention differently: `nn.EMA(tau, warm=False)` takes a static dimensionless retention factor `tau` (`tau * state + (1 - tau) * x`), whereas `control.EMA(dt, warm=False)` declares `tau` as a differentiable continuous-time constant in `param` (`alpha = 1 / (param.tau / dt + 1)`).

## 7. Introspection, paths, and PyTree utilities

### Structural inspection

| Function | Signature | Output |
| :--- | :--- | :--- |
| `describe` | `describe(value) -> str` | Compact one-line summary of view stage, name, unbound statics, and `[P]` / `[C]` flags |
| `tree_view` / `print_tree` | `tree_view(value, *, max_depth=None, unicode=True) -> str` | Hierarchical member tree with `[P]` (parametric) and `[C]` (cyclic) annotations |
| `summary` / `print_summary` | `summary(value, *, max_depth=None, print_fn=None) -> str` | Tabular breakdown of member paths, parameter counts/shapes, and state shapes |
| `statics_by_path` | `statics_by_path(value) -> frozendict` | Flat mapping from dotted member paths to recorded `@node` static `Struct` values |

### Path and PyTree operations

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `set_by_path` / `replace_by_path` | `set_by_path(tree, updates)` | Functional path-keyed updates (`{'encoder.w': new_w}`) over `Struct` and mapping PyTrees |
| `tile` | `tile(tree, n)` | Broadcasts every leaf with a new leading axis of length `n` (`tree_broadcast_axis(tree, n, axis=0)`) |
| `tree_broadcast_axis` | `tree_broadcast_axis(tree, count, axis)` | Inserts a broadcast axis of size `count` at `axis` across all leaves |
| `tree_first`, `tree_last`, `tree_tail` | `(tree, axis=0)` | Indexes element `0`, element `-1`, or slice `1:` along `axis` on every leaf |
| `tree_len` | `tree_len(tree) -> int` | Leading axis length (`shape[0]`) of the first leaf in `tree` |
| `tree_reshape` | `tree_reshape(tree, shape, axes=1)` | Replaces the first `axes` dimensions of every leaf with `shape` |
| `tree_swap_axes` | `tree_swap_axes(tree, axis_a, axis_b)` | Swaps `axis_a` and `axis_b` on every leaf (`jnp.swapaxes`) |
| `tree_take` | `tree_take(tree, indices)` | Gathers `leaf[indices]` along axis 0 on every leaf |
| `tree_stop_gradient` | `tree_stop_gradient(tree)` | Applies `jax.lax.stop_gradient` to all array leaves |
