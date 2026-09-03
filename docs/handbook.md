# NodeJAX Handbook

A public technical reference to NodeJAX's concepts, authoring system,
transforms, and composition. For the implementation architecture and authoring
interfaces, see the [design](design.md). For the argument behind the
model, see the [philosophy](philosophy.md).

---

# Part I: Core Architecture

## 1. The Core Contract

An authored Node uses up to three role functions: `param`, `init`, and
`apply`. The canonical `Contract` presents those roles through four uniform
entry points:

```
node.contract.param(param_input, rng)               -> param
node.contract.init(param, state_input, rng)         -> state
node.contract.prime(param, state_input, input, rng) -> state
node.contract.apply(param, state, input, rng)       -> (state, output)
```

At this canonical boundary, `rng` is a `MaybeKeyStream`: keyed when the call
requires randomness and empty when it does not. This keeps the contract
signature uniform without exposing the transform-oriented type to authored
leaf functions.

`init` and `prime` are two entry points to the single state-initialization
role. A definition records one initializer and whether it requires a real
runtime value. `init` handles the no-input form. `prime` resolves the
definition from a real input and supplies that value when required.

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
  * Name required values explicitly in authored signatures: `param`, `state`, `input`, `node`, and `rng`. A Leaf emits auxiliary values by returning `(output, Aux(...))`; its `apply` does not receive `self`.
  * Contain no child sub-nodes.

* **Composite construction (`Composite`, `serial`, `parallel`, `>>`)**:
  * Build behavior from registered members. An authored `Composite` may perform arbitrary JAX operations around its member calls.
  * Compose member hierarchies across all structural spaces:
    1. **`param`**: Member parameter PyTrees (`param.member_a`, `param.member_b`).
    2. **`state`**: Member state PyTrees (`state.member_a`, `state.member_b`).
    3. **`aux`**: Member auxiliary outputs (`aux.member_a`, `aux.member_b`).
    4. **`statics`**: Member configuration metadata.

> **The Structural Registration Rule**: Every child Node relationship must be registered as a member (`Composite(**members)`, `serial`, or a wrapper's inner). Closing over Nodes in private Python variables prevents parameter collection, state composition, RNG routing, and tree surgery.

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

### Why `scan` is a transform

JAX exposes `lax.scan(step, initial, sequence)` as an immediately applied control-flow primitive. That interface combines the declaration of a recurrence with one execution of it. The functional model does not require this coupling: a curried scan could accept `step` first and return a function of `initial` and `sequence`, while retaining the same tracing, lowering, and fixed-carry constraints.

NodeJAX makes that separation explicit because a Node contract already names the state and input roles. `scan(step)` declares a reusable sequence Node whose state remains externally carried. Calling `.scan(sequence)` on a bound Node executes that operation using its bound state and returns its successor. `scanned(step)` declares the alternative lifetime policy: initialize the state for each call, run the sequence, and consume the final state internally. The bound `.scan(sequence)` method is therefore the execution spelling of the `scan(step)` transform, not a separate recurrence mechanism.

Because transforms operate on declared contract roles (`param` vs `state`), they nest seamlessly:
```python
# MAML: inner fine-tuning inside a batched meta-trainer
maml = train_step(batch(finetune(train_step(model, mse, optax.sgd(0.1)))), mse, optax.adam(1e-3))
```

---

## 5. Binding Stages

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

`derive` extends an unbound Leaf definition without adding a Composite member:

```python
derived = derive(
    base,
    param=extra_param,
    init=extra_init,
    methods=method_overrides,
)
```

* An omitted `param`, `init`, or `apply` role is inherited.
* If parent and child both define `param` or `init`, their constructor inputs and returned `Struct` fields must be disjoint. Their fragments form one flat parameter or state value.
* An inherited transition may return only the state fields it updates; NodeJAX preserves the added fields. An explicit replacement transition over merged state must return every state field.
* Methods form a union in which the child overrides equal names, and tags form a union.
* **Bound Methods**: Methods declared on nodes bind to `PNode` and `PSNode` instances, receiving the instance's bound parameters and state automatically.

Current limits: an explicit state transition on an intermediate derived Node cannot be extended with another state fragment; define the complete transition on the most-derived Node. Random construction belongs to the derived Node as a whole, so the same root key does not guarantee that an inherited stochastic fragment reproduces its standalone base value after another constructor fragment is added.

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

## 10. Sequential Call Boundaries & Episode-Aligned Reset

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
`scan(model, boundary='episode')` runs matching hooks once at the start of each call. It names an episode boundary only when the caller aligns scan calls with episodes.

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
* `self`: The Composite or Wrapper invocation scope. It calls registered members and may collect auxiliary values with `self.sow(...)`. A Leaf `apply` does not accept it.

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

Inside `apply(self, ...)`, `self` is the composite's state, the one mutable thing in the step, and `self.member_name` is the member's bound view as of that read: it has a view's `param`, `state`, `bind`, `reset`, and `scan`, and its methods and members. Calling it runs the member from the view's state and stores the successor in the member's slot, so a later read of `self.member_name` sees the advance and repeated calls chain, while a view kept from before the call does not move. `bind(state=...)` and `reset(...)` store their state the same way and return the rebound view, which is how a run starts from state that arrives as data:

```python
def apply(self, observation, command, start):
    return self.replay.bind(state=start)(observation=observation, command=command)
```

The one difference from a view at the harness is deliberate: there, a call returns the successor beside the output; here the successor goes into the slot, so the same line serves a cyclic and an acyclic member.

---

## 13. Telemetry and Auxiliary Outputs (`Aux` / `sow`)

Leaf code returns an explicit `Aux` beside its primary output:
```python
def apply(param, input):
    output = input @ param.w
    return output, Aux(activity=jnp.linalg.norm(output))
```

A Composite or Wrapper may collect values around its member calls with `self.sow(...)`:
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
