# NodeJAX Handbook

A technical reference guide to NodeJAX's core concepts, authoring system, transforms, and composition mechanics. For the overarching architectural design principles and comparison with standard Python OOP, see [`docs/philosophy.md`](philosophy.md).

---

# Part I: Core Architecture

## 1. The Core Contract

In NodeJAX, every node is defined by up to three pure functions adhering to a uniform contract:

```
node.contract.param(param_input, rng)               -> param
node.contract.init(param, state_input, rng)         -> state
node.contract.prime(param, state_input, input, rng) -> state
node.contract.apply(param, state, input, rng)       -> (state, output)
```

At this compiled boundary, `rng` is a `MaybeKeyStream`: keyed when the call
requires randomness and empty when it does not. This keeps the contract
signature uniform without exposing the transform-oriented type to authored
leaf functions.

`init` and `prime` are the two initialization calls: a definition that
builds its state from a real runtime value has `prime` and no `init`, and
every other definition has `init` and no `prime`. Which one applies is a
fact about the definition, never about whether an argument was passed.

### The Three Contract Slots
* **`param`**: Static weights and learnable variables created at parameterization.
* **`state`**: Cyclic variables that evolve sequentially across calls (RNN hidden state, running moments, KV cache, simulator variables).
* **`input` / `output`**: Transient data flowing through the node during forward execution.

Nodes compose into larger nodes via combinators (`>>`, `serial`, `parallel`, `Composite`), and higher-order transforms (`batch`, `ensemble`, `scan`, `train_step`) apply to any node by transforming these three underlying contract functions.

---

## 2. Functional State

State in NodeJAX is pure and functional:
* A stateful node is **cyclic**: its apply takes `(param, state, input)` and returns `(next_state, output)`.
* State is an immutable value held and passed explicitly by the caller or runtime; nodes do not hold hidden mutable instance state.
* In compositions (`>>`, `Composite`), member states are combined into an immutable `Struct` keyed by member name. State threading through intermediate members is handled automatically by the combinator.

---

## 3. Composition

### Leaf and Composite Construction

`Leaf` and `Composite` are two ways to build the same definition value. They
are not runtime subclasses.

* **Leaf construction (`Leaf`)**:
  * Implement raw array math (`apply`), parameter initializers (`param`), and state constructors (`init`).
  * Receive authored signatures with access to `self.param`, `self.state`, an explicit `rng` argument (`KeyStream`), and `self.sow(...)`.
  * Contain no child sub-nodes.

* **Composite construction (`Composite`, `serial`, `parallel`, `>>`)**:
  * Contain **no raw math of their own**; behavior is 100% structural composition over registered `self.members`.
  * Compose member hierarchies across all structural spaces:
    1. **`param`**: Member parameter PyTrees (`param.member_a`, `param.member_b`).
    2. **`state`**: Member state PyTrees (`state.member_a`, `state.member_b`).
    3. **`aux`**: Member auxiliary outputs (`aux.member_a`, `aux.member_b`).
    4. **`statics`**: Member configuration metadata.

> **The Structural Registration Rule**: Every sub-component relationship must be registered as a member (`Composite(**members)`, `serial`, or a wrapper's inner). Closing over sub-nodes in private Python variables prevents parameter collection, state composition, RNG routing, and tree surgery.

Both forms finish as a `Def`, normally exposed to users through `Node`. Member
structure and wrapper transparency are definition data, not Python class
identity.

---

## 4. Higher-Order Transforms

Transforms consume a `Node` and produce a new `Node` by transforming its underlying contract functions:

| Transform | Primary Effect | Parameter Handling | State Handling |
| :--- | :--- | :--- | :--- |
| **`batch(node)`** | Vectorizes over data (`vmap`) | Shared across batch | Vectorized per sample |
| **`ensemble(node, n)`** | Vectorizes over population (`vmap`) | Vectorized across members | Vectorized across members |
| **`stack(node, n)`** | Vectorizes depth layers (`scan`) | Stacked over depth | Stacked over depth |
| **`scan(node)`** | Sequential recurrence (`lax.scan`) | Shared across time | Carried step-to-step |
| **`scanned(node)`** | Sequence rollout (internalized carry) | Shared across time | Initialized and consumed internally |
| **`train_step(node, loss, opt)`** | Turns model into a trainer | Initial weights (param) | Weights & opt moments (state) |
| **`trained(trainer)`** | Runs optimization to completion | Evaluated weights | Returns final trained model + loss aux |

Because transforms operate on declared contract roles (`param` vs `state`), they nest seamlessly:
```python
# MAML: inner fine-tuning inside a batched meta-trainer
maml = train_step(batch(finetune(train_step(model, mse, optax.sgd(0.1)))), mse, optax.adam(1e-3))
```

---

## 5. The Binding Ladder

NodeJAX models system instantiation across five explicit stages:

```
statics -> private tree binding -> input evidence -> params -> state
```

1. **Static construction**:
   Architectural definitions where static parameters (channel widths, layer counts, flags) are open. `specialize(**overrides)` resolves open statics.
2. **Private tree binding**:
   A complete construction materializes named member definitions. Framework
   tree surgery may explicitly rebuild this stage; it is not a general public
   Node operation.
3. **Input evidence**:
   `with_input` resolves shape-dependent definition facts without binding
   runtime values.
4. **Parameter binding**:
   `parameterize` returns `PNode(definition, param)`. Its JAX leaves are exactly
   the parameter tree; the definition is static auxiliary data.
5. **State binding**:
   initialization returns `PSNode(definition, param, state)`. Calling it
   executes one step and returns a functional successor:
   ```python
   advanced, output = model(input_data)
   ```

`Node`, `PNode`, and `PSNode` are sibling views over the same `Def`, sharing a
small `BaseNode`. A `Generic` is earlier still: an unfinished construction
record with no executable definition. Re-entering any stage discards the
bindings after it. In particular, `specialize` reruns construction and does
not retain input evidence, params, state, or an ephemeral `map_members`
rewrite.

---

## 6. Functional Derivation (`derive`) and Methods

NodeJAX replaces classical OOP class inheritance with **functional derivation**:

```python
# Derive a new node by overriding specific contract functions:
base = Linear(64)
derived = derive(base, apply=custom_apply)
```

* `derive` inherits parameter and state constructors from `base` while replacing `apply`.
* **Bound Methods**: Methods declared on nodes bind to `PNode` and `PSNode` instances, receiving the instance's bound parameters and state automatically.

---

## 7. Statics and Reconfiguration (`specialize`)

Statics are values that determine graph structure and compile-time shapes (e.g. channel widths, time constants, `dt`, mode flags). They are recorded by `@node` at construction and do not enter PyTrees.

### Rebuilding with `specialize`
A built node tree can be reconfigured deterministically via `specialize`:
* `model.specialize(layer_name={'width': 128})` targets a specific member.
* `model.specialize(**{'*.train': False})` broadcasts static flags across all matching sub-nodes.
* Re-specializing re-runs the declarative construction top-to-bottom and
  materializes a fresh member tree, ensuring derived shapes and topology are
  recomputed consistently.

---

## 8. Randomness & PRNG Key Routing

Randomness is an explicit call dependency with two deliberately separate
representations:

1. **Signature Declaration**: An authored function that draws declares `rng`
   in its signature. Compilation records that the call requires randomness;
   `rng` is not retained as a data-input field.
2. **Single Boundary Key**: The caller supplies one raw key at the public
   boundary (`parameterize(rng=key)`, `initialize(rng=key)`, or
   `apply(rng=key)`). A deterministic definition rejects a surplus key and a
   stochastic definition rejects a missing one.
3. **Uniform Transform Argument**: Compiled calls receive a `MaybeKeyStream`
   explicitly beside their data. Composites give stochastic children keyed
   streams and deterministic children empty streams. Axis transforms split or
   broadcast the same value according to the child's declared requirement.
   No runtime bundle inspection or RNG `ContextVar` is involved.
4. **Scope-Local Authoring Stream**: At the leaf-authoring boundary, the final
   keyed value becomes a local `KeyStream`; authors draw independent
   keys via `rng.next()`:
   ```python
   def param(node, rng):
       w = jax.random.normal(rng.next(), (node.input.shape[-1], 64))
       return Struct(w=w)
   ```

A state or domain PyTree may separately contain a value named `rng`. That is
modeled state, not the framework call channel, and it advances only according
to that node's state transition.

---

## 9. Structural Parameter Sharing (`tie`)

Parameter sharing in NodeJAX is structural rather than pointer-based:
* `tie(pipeline, 'src_member', 'alias_member')` maintains a single copy of the shared tensor in the `param` PyTree.
* At apply time, the shared parameter is forwarded to all specified consumer slots.
* Gradients accumulate naturally via the chain rule with zero risk of PyTree de-aliasing or pointer desynchronization under transformations.

---

## 10. Sequential Boundaries & Episode Reset

When scanning models over long sequences in chunks, recurrent state and running statistics have different lifetime requirements:
* Recurrent hidden state should reset between distinct episodes.
* Running statistics (like BatchNorm or covariance estimators) should persist across episodes.

### Declaring Boundary Actions
Nodes can declare boundary reset rules via the `boundary` dictionary:
```python
def keep_statistics(carried, init, decided):
    # Reset fast hidden state to init, but carry running stats:
    return init.replace(stats=carried.stats)

Leaf(apply, init=init, boundary={'episode': keep_statistics})
```
Enclosing `scan(model, boundary='episode')` runs these declared boundary hooks when an episode boundary is reached.

---

# Part II: Authoring & Syntax Reference

## 11. Authoring Leaf Nodes with `Leaf`

Leaf functions are authored using human signatures:
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
```

### Reserved Signature Names
* `param`: The node's parameter PyTree.
* `state`: The node's active cyclic state PyTree.
* `input`: The primary input data.
* `node`: The resolved node metadata (e.g. `node.input.shape`).
* `rng`: The local `KeyStream` for PRNG draws.
* `self`: Context object for composite nodes and auxiliary logging (`self.sow(...)`).

---

## 12. Hand-Wired Composites (`Composite`)

When dataflow cannot be expressed as a linear pipeline (`>>`), use `Composite` to author custom branching and gating wiring:

```python
@node
def Highway(dim: int):
    members = Composite(transform=Linear(dim), gate=Linear(dim))

    def apply(self, input: jax.Array) -> jax.Array:
        h = jax.nn.relu(self.transform(input))
        t = jax.nn.sigmoid(self.gate(input))
        return h * t + input * (1.0 - t)

    return members(apply)
```

Inside `apply(self, ...)`, calling `self.member_name(...)` automatically slices the member's parameters and state, steps the member, and records updated state seamlessly.

---

## 13. Telemetry and Auxiliary Outputs (`Aux` / `sow`)

To emit intermediate diagnostics or auxiliary losses without altering primary dataflow:
```python
def apply(self, input):
    x = self.encoder(input)
    self.sow(latent_norm=jnp.linalg.norm(x))
    return self.decoder(x)
```
Auxiliary outputs collect into an `Aux` PyTree matching the model hierarchy and automatically vectorize under `batch`, `ensemble`, and `scan`.

---

## 14. Summary of Common Operations

```python
# 1. Pipeline Definition
model = Linear(64) >> nn.gelu >> Linear(10)

# 2. Shape Binding & Parameterization
bound_model = model.with_input(jnp.zeros((32, 128))).parameterize(rng=jax.random.PRNGKey(0))

# 3. Vectorization
batched_model = batch(bound_model)

# 4. Evaluation Re-specialization, then explicit compatible transfer
eval_node = bound_model.specialize(**{'*.train': False})
eval_model = eval_node.bind(bound_model.param)

# 5. Direct Execution
output = eval_model.apply(batch_x)
```
