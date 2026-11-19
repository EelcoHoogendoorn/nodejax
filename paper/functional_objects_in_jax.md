# Functional objects in JAX

JAX closes pure functions under composition and under higher-order function transformations. Composing two pure functions yields another pure function, and transforming a pure function returns a pure function that composes and transforms again. Above single functions, numerical programs assemble structured units such as controllers, physical plants, estimators, neural layers, caches, replay buffers, and optimizers whose execution depends on compile-time graph structure, post-compilation parameters, step-to-step feedback carries, per-call signals, and explicit randomness.

This document sets out ***Functional Object-Oriented Programming*** (***FOOP***), an object model for JAX tailored to the constraints of compilation, temporal recurrence, and vectorization so composite, feedback-carrying units retain the same closure under composition and transformation that `jax` gives pure functions. It develops an object contract, binding stages, and `Node` transforms in domain-neutral terms, illustrates how `nodejax` expresses neural architectures, block-diagram control, physics simulation, and reinforcement learning within one algebra, and compares that contract against class-based module libraries at the end.

| | Pure functions (`jax`) | Functional objects (`FOOP` / `nodejax`) |
| :--- | :--- | :--- |
| Compose | `lambda x: jnp.sum(jax.nn.sigmoid(x))` | `nn.RNN(64) >> nn.relu` |
| Transform | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

---

## Introduction

JAX derives its expressive economy from closure over pure functions on PyTrees [[1]](#references). Composing two pure functions `f` and `g` produces a pure function `lambda x: g(f(x))`, unrolling a pure step function with `jax.lax.scan` defines a pure sequence computation, and applying `jax.jit`, `jax.vmap`, or `jax.grad` returns another pure function that composes with every remaining transformation without adapter code.

Because positional arguments carry no intrinsic role metadata, that closure holds directly only when a program stays at the granularity of bare functions. Larger numerical systems organize into composite units that bundle compile-time configuration, post-compilation constants, recurrent step carries, runtime inputs, and hierarchical member names.

Representative application domains illustrate what those units have to carry together. In block-diagram control, a motor drive couples sensors, filters, observers, proportional-integral controllers, and a physical plant. In reinforcement learning, an experiment couples that plant to a recurrent policy, moving-average target networks, a replay buffer, and an update step. In sequence modeling, an adaptive predictor couples tied embeddings, running normalizers, stochastic regularizers, key-value caches, and online weight updates.

Early in JAX's development, `jax.example_libraries.stax` explored combinator composition [[2]](#references). Each layer constructor returned an `(init_fun, apply_fun)` pair, and `stax.serial` and `stax.parallel` composed sequences of pairs into a new `(init_fun, apply_fun)` pair, while Optax later applied an analogous `(init, update)` pair (`GradientTransformation`) to optimizers [[3]](#references). Because `stax` pairs lacked a step-carried feedback argument, named PyTree key paths, structural parameter sharing, and higher-order transforms over pairs, combinator design stopped at feed-forward neural networks.

Extending closure under composition and transformation from pure functions and feed-forward combinator pairs to general functional objects requires an object contract shaped by JAX's compilation and transformation boundaries, so compatible stages piped sequentially via `>>`, temporal unrolling, vectorization, and optimization compose within one type:

```python
step = nn.GRU(WIDTH) >> nn.Linear(OUT)                       # cyclic Node
predictor = batch(scanned(step))                             # cyclic -> acyclic
trainer = train_step(predictor, mse_loss, optax.adam(1e-3))  # param -> cyclic state
```

---

## Object contract

In JAX (and any functional array system built around compilation, recurrence, vectorization, and explicit PRNG keys), four primitives partition the values of a computation along distinct semantic boundaries. Together, `jax.jit`, `jax.lax.scan`, `jax.vmap`, and `jax.random` constrain and inform the design of a compatible functional object contract.

### Compile-time statics and post-compilation parameters

Tracing a Python callable into an XLA executable via `jax.jit` partitions its inputs into two regimes. Compile-time `statics` (`width=64`, `passes=8`, `train=True`) fix graph topology, control flow, loop bounds, array ranks, shapes, and dtypes at trace time; changing a static invalidates the trace and recompiles the executable. Traced PyTree leaves cross the compiled boundary as XLA buffers and vary between calls, across `vmap` lanes, or across `scan` iterations without recompilation.

Within the traced regime, parameters (`param`) are values bound to an object definition that remain invariant across the steps of one rollout while remaining changeable after compilation. For example, a linear layer's weight matrix, a controller's gains, an estimator's filter constant, and a discontinuous simulator's physical constants all belong in `param`, because each stays fixed during a rollout while remaining sweepable across `vmap` batches, tunable by evolution, or, where the computation is smooth, differentiable on a selected subset via `jax.grad` under one compiled executable.

### Cyclic state, per-call input, and auxiliary outputs

Once `statics` and step-invariant `param` are separated, temporal recurrence (`jax.lax.scan`) partitions the remaining traced values of a step into two roles. Per-step `input` and `output` carry the signal flowing from upstream stages to downstream stages within one evaluation (`input -> output`). Step-carried `state` holds the value produced on one evaluation and consumed by the same unit on the next, extending `lax.scan`'s `(carry, input) -> (next_carry, output)` signature to `(param, state, input) -> (next_state, output)`.

In a computation graph, a feed-forward unit (`input -> output`) is an acyclic vertex, whereas a unit that routes `next_state` back into its own `state` argument on the following step forms a directed feedback cycle across time. Such a unit is here termed ***cyclic***. Imperative object-oriented programming calls an object stateful when calls mutate storage hidden inside the instance; in a functional model neither an unbound definition nor an initialized PyTree snapshot mutates storage, because stepping a snapshot returns a new snapshot paired with its output. `cyclic` names the recurrent graph edge shared by integrators, recurrent cells, running normalizers, physical plants, key-value caches, replay buffers, and optimizers.

Nested units also emit diagnostics, regularizer terms, or recorded trajectories read by outer losses or monitors without changing downstream stage inputs. Accumulating an explicit auxiliary PyTree (`aux`) alongside `output` preserves unary pipeline composition while keeping every emitted array in the return value.

### Vectorization axes and explicit randomness

Vectorization (`jax.vmap`) maps or broadcasts `param`, `state`, and `input` along distinct axes depending on the transform, motivating separate top-level PyTrees for the three roles so transforms route each role uniformly without per-call axis specifications. Each row below lists the ordinary axis mapping applied to an inner unit by three standard architectural transforms:

| Transform | `param` axis | `state` axis | `input` / `output` axis |
| :--- | :--- | :--- | :--- |
| `batch` | broadcast (`None`) | mapped (`0`) | mapped (`0`) |
| `ensemble` | mapped (`0`) | mapped (`0`) | broadcast in, stacked out (`0`) |
| `stack` | scanned (`0`) | scanned (`0`) | threaded sequentially |

Keeping `param`, `state`, and `input` in separate PyTrees lets `batch` broadcast `param` while mapping per-sample `state`, and lets `ensemble` replicate independent `(param, state)` pairs across a shared input. Where a `cyclic` unit aggregates statistics across a batch (such as running moments in `BatchNorm`), declaring collective batch-shared state on the unit's contract metadata tells `batch` to keep that state entry unmapped across the batch axis while mapping per-sample carries along axis `0`, without special-casing concrete layer types.

Explicit randomness places a parallel requirement on the contract. Parameter generation, state initialization, and step evaluation may each be deterministic or stochastic. Declaring `rng` consumption per role on the contract allows composites and axis transforms to split independent PRNG keys across stochastic members and mapped lanes automatically.

Each row below summarizes the JAX primitive, the semantic split it imposes, and the contract role it motivates:

| Primitive | Semantic split | Contract role |
| :--- | :--- | :--- |
| `jax.jit` | Trace-time structure vs. XLA buffers | `statics` vs. `{param, state, input}` |
| `jax.lax.scan` | Step constant vs. step carry vs. signal | `param` vs. `cyclic` `state` vs. `input` / `output` (+ `aux`) |
| `jax.vmap` | Role-specific axis mapping | Separate `param`, `state`, and `input` PyTrees |
| `jax.random` | Deterministic vs. stochastic roles | Per-role `rng` on `param`, `init`, and `apply` |

---

## Functional objects

Whereas imperative object-oriented programming organizes around mutable instances with arbitrary peer methods, FOOP organizes around a computation-graph vertex. The term ***Node*** here means a unit centered on one privileged computation (`apply`), bundled with the `param` and `init` constructors that produce `apply`'s bound PyTree inputs. Three properties define the general FOOP model, namely a canonical contract over explicit role PyTrees, immutable bound `PyTree` containers that separate static `treedef` structure from traced `param` and `state` subtrees, and contract-to-contract `Node` transforms. `nodejax` realizes that model on top of JAX through four explicit binding views, path-addressed static specialization, upstream shape propagation, and named state-reset boundaries.

### Node contract

Every leaf primitive, wired composite, and transformed unit lowers to one canonical contract over immutable PyTrees:

```python
param(rng?) -> param
init(param, input?, rng?) -> state
apply(param, state, input, rng?) -> (next_state, output)
```

Alongside those three functions, a `Node` records which roles are active (`param`, `cyclic` `state`, and per-role `rng`), while optional `aux` outputs travel alongside `output`. Sequential piping (`>>`) chains unary stages, and named-member composition wires arbitrary branching, multi-input, or feedback dataflow across child `Node`s while nesting child `param`, `state`, and `aux` subtrees by member name into composite PyTrees of the same form, so an arbitrarily nested composite exposes the same external contract as an elementary `Node`.

Because `Node` transforms act on a single privileged `apply` transition, domain objects that also support read-only operations (such as an observation readout on a plant, action sampling or log-probability scoring on a policy, or an inverse map alongside a forward pass) attach those operations as read-only methods over bound `param` and `state`. An enclosing composite invokes those methods directly, and wrapping a method inside a single-member `Node` exposes that operation as `apply` whenever `batch`, `scan`, or `train_step` needs to transform it.

### Views and binding stages

In `nodejax`, configuration, shape propagation, parameter creation, and state initialization produce distinct immutable views across four explicit binding stages:

```python
node = blueprint.specialize(**{'*.width': 768, '*.train': False})  # Generic -> Node
sized = node.with_input(sample_input)                              # Node    -> Node
pnode = sized.parameterize(rng=param_key)                          # Node    -> PNode
psnode = pnode.initialize(rng=init_key)                            # PNode   -> PSNode
```

The term ***Generic*** here means an object blueprint constructed with unbound static arguments; a `Generic` already pipes (`>>`), nests inside composites, and wraps under transforms (`stack`, `ensemble`, `residual`) before widths, depths, or mode flags are chosen. Resolving unbound statics (`specialize`) and propagating input specifications (`with_input`) produces a concrete `Node` definition, and calling `specialize` again derives a new `Node` whenever compile-time structure changes (for example, turning `*.train` off to rebuild stochastic training blocks into deterministic inference blocks).

Separating architectural wiring from static binding also addresses why library-level reuse is rare in Python neural network code. When static hyperparameters and input dimensions have to be supplied inside eager constructors, composite builders either hard-code inner choices or forward every nested width, depth, activation, and mode flag through intermediate constructor signatures that never use them directly. Keeping `Generic` composition open until `specialize` and inferring upstream dimensions via `with_input` lets intermediate blocks, stacks, and ensembles stand as reusable library values configured by path from the outside.

Binding `param` to a `Node` yields an immutable ***PNode*** (parameter-bound `Node`) pairing `(node, param)`, and binding `state` yields an immutable ***PSNode*** (parameter-and-state-bound `Node`) bundling `(node, param, state)`. Both register as custom `PyTree` containers with `jax.tree_util` [[4]](#references), holding `node` in the static `treedef` while exposing `param` and `state` as dynamic `PyTree` subtrees so `PNode` and `PSNode` pass directly through `jax.jit`, `jax.vmap`, `jax.lax.scan`, and `jax.grad`. Calling an acyclic `PNode` evaluates `pnode(input) -> output`, whereas calling a `cyclic` `PSNode` evaluates `psnode(input) -> (next_psnode, output)`, so a `PSNode` passes directly as a `lax.scan` carry and can be saved, forked, or resumed at any step as an immutable PyTree value.

### Node transforms

A `Node` transform maps an input `Node` contract to a derived `Node` contract (`node -> transformed_node`) without inspecting Python source code or branching on concrete subtypes. Returning a derived `Node` gives every transform a common output interface, so further transforms compose structurally whenever array shapes, call contracts, and smoothness requirements (such as differentiability under `jax.grad`) align.

Axis, wiring, and temporal transforms (`batch`, `ensemble`, `stack`, `residual`, `parallel`, `tie`, `closed_loop`, `scan`, `scanned`, `cyclic`) rewrite `param`, `state`, `input`, `output`, `aux`, and `rng` routing purely from contract metadata. Among temporal transforms, `scan(node)` unrolls a `cyclic` step across a leading sequence axis while retaining its carry `state`, `scanned(node)` initializes `state` per call and drops the final carry to form an acyclic sequence map, and `cyclic(node)` turns an acyclic state-record pass `(s, u) -> (s_next, y)` into a `cyclic` `Node` carrying `s` in `state`. Tracing one recurrent readout across `scan` and `batch` shows how each transform rewrites the active roles:

```python
step = nn.GRU(16) >> nn.Linear(4)  # cyclic: param {gru, linear}, state {gru: (16,)},   (D,) -> (4,)
seq = scan(step)                   # cyclic: param unchanged,     state {gru: (16,)},   (T, D) -> (T, 4)
batched = batch(seq)               # cyclic: param broadcast,     state {gru: (B, 16)}, (B, T, D) -> (B, T, 4)
```

Replacing `scan(step)` with `scanned(step)` initializes and discards the `(16,)` carry inside each call, yielding an acyclic `(B, T, D) -> (B, T, 4)` sequence predictor whose bound view is a `PNode` rather than a `PSNode`.

Because `param`, `state`, and `input` sit in distinct contract roles, `Node` transforms can also move selected subtrees between roles without editing inner definitions:

```python
frozen = freeze(normalizer, saved_stats)                                      # cyclic state -> constant
driven = externalize(closed_loop(ctrl >> plant), 'pipe.plant')                # param -> apply input
trainer = train_step(system, loss_fn, optax.adam(1e-2), trainable='ctrl')     # param -> cyclic state
```

`freeze` fixes a `cyclic` `Node`'s `state` at a supplied value and discards step updates to return an acyclic `Node` (`tree_freeze` applies that step selectively by path or tag, such as `'running_stats'` inside `eval_mode`). `externalize` moves a selected `param` subtree into a per-call `apply` input so an outer caller can supply those parameters dynamically at evaluation time. `train_step` partitions `system.param` by `trainable` path so unselected parameters stay fixed in `trainer.param`, while selected parameters, `system.state`, and optimizer state (adapting Optax `GradientTransformation` pairs [[3]](#references) directly into optimizer `Node`s) compose into `trainer.state`.

Addressing nested members by path (`specialize`, `tie`, `externalize`, `trainable`) removes constructor parameter forwarding and shared-reference aliasing while coupling external rewrites to member hierarchy names (as `'pipe.plant'` inside `closed_loop` shows). Likewise, because `Node` fixes static topology in the PyTree `treedef` at trace time to match `jax.jit`, programs whose graph structure mutates dynamically in Python on each step sit outside the compiled `Node` algebra.

### State lifetimes and call boundaries

Different `cyclic` carries inside a composite reset on distinct schedules, such as an episodic recurrent hidden vector paired with a persistent running normalizer. A `Node` attaches named `boundary` hooks to its own state, and an enclosing `scan` claims matching boundary names across its subtree:

```python
core = state_reinit(nn.GRU(64), boundary='episode') >> nn.BatchNorm(momentum=0.1)  # cyclic Node with episode boundary
runner = batch(scan(scan(core, n=CHUNK), boundary='episode'))                      # resets GRU per episode, keeps BatchNorm
```

When `runner` starts an episode, every descendant declaring `'episode'` reinitializes its state while unclaimed carries persist untouched, keeping reset schedules attached to the owning `Node` rather than encoded in outer loop carries.

---

## Domain patterns

Across neural architectures, block-diagram control, physics simulation, and reinforcement learning, domain computations partition state and operations in four distinct ways while sharing the same `Node` algebra.

### Neural architectures

Neural architectures decompose into fine-grained stages whose primary operation is a forward map (`input -> output`) backed by post-compilation weights (`param`) and per-layer `cyclic` carries such as recurrent hidden vectors, normalization moments, or key-value caches.

Whereas conventional deep learning code places gradient updates in an outer loop outside the module abstraction because modules represent forward evaluation only, `nodejax` expresses a single supervised update as a `Node` transform (`train_step(model, loss_fn, optimizer)`). The resulting trainer is an ordinary `cyclic` `Node` whose `trainer.param` holds initial weights (together with any optimizer parameters such as learnable step sizes) while `trainer.state` bundles the evolving weights, `model.state`, and optimizer accumulator state. Because one optimization step satisfies the standard `Node` contract, multi-step training unrolls via `scan` (`trained`), episodic inner-loop adaptation (`finetune`) differentiates through `trainer.param`, and online test-time training (`next_step_ttt`) nests an adapting block directly inside a layer stack (`stack`). In `nodejax`'s `third_order_learning.py` example, a recurrent block is wrapped in test-time training and stacked between tied input and output projections:

```python
block = residual(nn.MinGRU(WIDTH)) >> nn.BatchNorm(momentum=0.1) >> nn.tanh                    # cyclic Node
adaptive = next_step_ttt(train_step(block, token_loss, learned_sgd(TTT_LR)))                   # param -> cyclic state
predictor = tie(Encoder(IN, WIDTH) >> stack(adaptive, n=DEPTH) >> Decoder(IN, WIDTH), 'encoder', 'decoder')
```

Enclosing `scanned(predictor)` further inside episodic `finetune`, `ensemble`, `batch`, and `trained(train_step(...))` differentiates end-to-end to third order without manual fast-weight or carry plumbing.

### Control and block-diagram systems

In block-diagram dynamical systems, state splits into many small pockets distributed across hardware and signal-processing blocks such as discrete derivatives, unit delays, low-pass filters, proportional-integral-derivative controllers, state observers, thermal models, batteries, and plants, each advancing its own local `cyclic` carry. Threading and updating dozens of small state tuples manually across multi-rate pipelines and feedback loops dominates procedural simulation code, whereas modeling each block as a `Node` aggregates those local carries into one hierarchical `state` PyTree routed automatically across `>>`, `closed_loop`, `scan`, and `batch` while keeping tunable or sweepable plant constants and controller parameters in `param`. In `nodejax`'s `test_motor_control.py` example, an ensemble of recurrent controllers drives a fixed-parameter direct-current motor plant in closed loop:

```python
core = nn.Linear(HIDDEN) >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH) >> nn.Projection()  # cyclic Node
controller = ensemble(core, n=MEMBERS) >> reduce(jnp.mean)                               # ensemble -> cyclic Node
rollout = batch(scanned(closed_loop(controller >> Motor(DT))))                           # cyclic -> acyclic
trainer = train_step(rollout, tracking_loss, optax.adam(0.02))                           # param -> cyclic state
```

On smooth dynamics (`Motor`), `train_step` differentiates through the closed-loop rollout directly. On quantized hardware stacks in `nodejax`'s `actuator` and `whip` examples, `vmap` sweeps randomized plant `param` leaves while differential evolution advances candidate controller populations inside `state` under one compiled XLA executable.

### Physics simulation

Although multi-body and particle simulators are also dynamical systems, their state organization differs sharply from block-diagram control. Instead of small private carries inside separate blocks, a simulator maintains a large shared ensemble state array (such as positions and velocities across particles or rigid bodies) updated within one timestep by sequential gather-and-scatter passes. Modeling every internal constraint pass as a separate `cyclic` `Node` would fight that shared array layout. Instead, free-motion prediction, iterative constraint projection, and velocity updates are written as acyclic `Node`s (`batch` over bodies, `repeat` over solver iterations) transforming the explicit state record, and `cyclic(...)` closes the completed timestep at the outer boundary. In `nodejax`'s extended position-based dynamics (`XPBD`) example (`pbd.py`), one solver step takes four lines:

```python
predict = batch(FreeMotion(dt))                         # acyclic particle pass
solver = repeat(constraints, n=PASSES)                  # acyclic constraint loop
finalize = batch(VelocityUpdate(dt, damping))           # acyclic velocity pass
step = cyclic(PBDStep(predict, solver, finalize))       # acyclic -> cyclic Node
```

Once closed by `cyclic(...)`, `step` is an ordinary `cyclic` `Node` that couples directly to actuator models and feedback controllers under `closed_loop`, `scan`, and `batch`.

### Reinforcement learning

Reinforcement learning algorithms alternate between rollout collection and parameter updates that evaluate the same policy in different ways, sampling actions against a plant during collection and scoring recorded actions under a surrogate loss during optimization. Because `Node` transforms (`batch`, `scan`, `train_step`) act on a `Node`'s single `apply` computation, an algorithm with multiple per-phase operations wraps `policy` in a separate `Node` per phase whose `apply` runs that phase's step. Once `train_step` moves the policy's trainable parameters into its own evolving `state`, the rollout `Node` also needs to evaluate those live weights rather than a separate parameter copy; applying `externalize` to the rollout `Node` turns its policy parameters into a per-call `apply` input supplied from the trainer's current state on each iteration. In `nodejax`'s `ppo.py` example, parallel chunk collection, recurrent chunk replay, and clipped surrogate updates are written as `Node`s:

```python
sampler = externalize(batch(scanned(scan(SamplingStep(policy, plant), n=CHUNK), n=CHUNKS), n=WORLDS), 'policy')
replay = batch(ChunkReplay(scan(CommandLikelihood(policy))), n=MINIBATCH)
actor_trainer = scan(scan(train_step(replay, clipped_surrogate(clip=CLIP, entropy_weight=ENTROPY), actor_opt), n=BATCHES), n=EPOCHS)
```

Orchestrating multi-phase `PPO` or `SAC` iterations inside an outer composite matches well-factored procedural JAX functions in line count rather than compressing them. The structural gain is whole-experiment closure, because the entire training state (`actor`, `critic`, `target`, replay `Buffer`, `plant`, optimizers) forms one `PSNode`, swapping an acyclic MLP `policy` for a `cyclic` GRU `policy` edits zero lines of the algorithm, and complete training runs compose under `scan`, `ensemble`, and `jax.grad`.

Each row below summarizes how state organization, composition idioms, and primary structural gains differ across the four domains:

| Domain | State organization | Composition idiom | Primary gain |
| :--- | :--- | :--- | :--- |
| Neural architectures | Per-layer carries and caches | `>>`, `residual`, `stack`, `tie`, `train_step` | Online adaptation and meta-learning inside layer graphs |
| Control and block diagrams | Small private carries per block | `>>`, multi-rate `scan`, `closed_loop` | Hierarchical `state` composition without manual carry plumbing |
| Physics simulation | Shared ensemble state array | Acyclic `batch`/`repeat` passes inside `cyclic` | Gather-scatter solver passes closed into a `cyclic` `Node` |
| Reinforcement learning | Policy, critic, replay buffer, optimizers | Per-phase `Node` wrappers plus `externalize` | Single-`PSNode` experiment and MLP/GRU parity |

---

## Conclusion

JAX standardized both pure array primitives closed under `jit`, `vmap`, `lax.scan`, and `grad` [[1]](#references) and `PyTree` container registration (`jax.tree_util`) separating static `treedef` structure from traced array leaves [[4]](#references). Leaving program organization above `PyTree` unspecified forced downstream libraries to invent incompatible encodings for compile-time `statics`, post-compilation `param`, `cyclic` `state`, reset lifetimes, and call-time `rng` routing, separating neural layers from plants, solvers, buffers, and optimizers.

A functional object contract (`Generic` blueprint, `Node` definition as `PyTree` `treedef`, `PNode` and `PSNode` bound `PyTree`s, and contract-preserving `Node` transforms) occupies that missing domain-neutral layer above `PyTree`. It prescribes zero layer catalogues, physical equations, or optimization algorithms, completing the combinator direction initiated by `stax` [[2]](#references) by extending closure under composition and transformation from pure functions to functional objects.

---

## Related work

Existing Python and JAX module libraries (`Flax Linen` [[5]](#references), `Haiku` [[6]](#references), `Equinox` [[7]](#references), `Flax NNX` [[8]](#references), `PyTorch` [[9]](#references), and `Keras 3` [[10]](#references)) standardize container representation or neural-layer execution without standardizing a unified `(param, state, input) -> (next_state, output)` transition contract or role-routing transforms. Two structural differences separate those module systems from FOOP.

### Eager construction and static-dynamic partitioning

When a module initializes in `PyTorch` [[9]](#references), `Equinox` [[7]](#references), or `Flax NNX` [[8]](#references), compile-time statics (`self.width`), post-compilation parameters (`self.weight`), `cyclic` state (`self.running_mean`), child modules, and initialization PRNG keys are populated onto one instance inside `__init__`.

Combining those roles on one instance leaves the `treedef` versus leaf boundary required by `jax.jit` to external conventions or post-hoc partitioning. `Flax Linen` [[5]](#references) and `Haiku` [[6]](#references) evaluate `class` methods inside ambient frame contexts (`Module.init`/`Module.apply`, `hk.transform`) that extract external parameter and state dictionaries. `Flax NNX` [[8]](#references) traverses mutable Python reference graphs (`PyGraphs`) via `nnx.split`, `nnx.merge`, and `nnx.update`. `Equinox` [[7]](#references) provides a general custom `PyTree` dataclass (`eqx.Module`) with arbitrary user methods and filters array leaves at runtime (`eqx.is_array` inside `eqx.filter_jit` and `eqx.filter_grad`, supplemented by `eqx.field(static=True)` tags); under runtime type filtering, passing `gain=1.0` versus `gain=jnp.array(1.0)` changes whether `gain` compiles as a trace-time constant or a post-compilation buffer. Allocating parameters inside `__init__` also requires concrete dimensions at construction time (`Flax NNX` performs no shape inference [[8]](#references)), forcing outer builders to accept and forward nested hyperparameters down constructor chains (`Keras 3` defers input fan-in to `build(input_shape)` [[10]](#references) while still forwarding architectural hyperparameters across builders).

### Unstandardized state and transition protocols

General `PyTree` containers in `Equinox` [[7]](#references) leave methods unspecialized, so `param` versus `cyclic` `state` separation, initialization, reset lifetimes, and step signatures remain user conventions. Within `eqx.nn`, `BatchNorm` routes state through `eqx.nn.StatefulLayer` and an external `eqx.nn.State` object indexed by Python object identity (`StateIndex`); under `eqx.filter_vmap` over a layer constructor (`stack`), `StateIndex` fails to allocate distinct per-layer state entries. Meanwhile, stock recurrent cells (`eqx.nn.GRUCell`, `eqx.nn.LSTMCell`) take an explicit positional `hidden` state (`(input, hidden) -> next_hidden`), and custom lifted `cyclic` layers fall back to a `(successor_module, output)` convention that mixes `param` and `state` leaves in one `PyTree`. `Flax NNX` [[8]](#references) mutates `nnx.Variable` attributes in place during `module(x)` for layers such as `BatchNorm`, while routing recurrent cells (`nnx.RNNCellBase`) through an explicit `initialize_carry` and `(carry, x) -> (carry, y)` tuple protocol; a single `Residual` wrapper cannot preserve both protocols, and an exception raised after `body(x)` mutates a `Variable` leaves the caller's module half-updated in place. `Keras 3` on JAX [[10]](#references) compiles single-model `keras.layers.RNN` loops into XLA, yet provides no module-level ensemble or inner-adaptation transform, so committee evaluation and meta-learning exit `keras.Layer` into stacked variable lists, `stateless_call`, and raw `jax.grad`.

Because none of these libraries standardize a `(param, state, input) -> (next_state, output)` contract with explicit role metadata and reset boundaries, cross-domain systems coupling controllers, plants, sensor pipelines, replay buffers, multi-horizon state resets, and Optax transitions [[3]](#references) assemble bespoke carry tuples (`(params, model_state, opt_state, env_state, rng)`) by hand, where replacing an acyclic policy or block with a `cyclic` one requires coordinated edits across rollout, storage, replay, and update loops.

---

## Companion documents

Source code, runnable domain programs, and reference documentation in this repository accompany this document:

- [`readme.md`](../readme.md) introduces the library overview and highlight examples.
- [`docs/philosophy.md`](../docs/philosophy.md) gives a shorter library-oriented rationale for `nodejax` with concrete authoring examples.
- [`docs/authoring.md`](../docs/authoring.md) and [`docs/handbook.md`](../docs/handbook.md) document `nodejax`'s authoring tiers, specifications, and transform surface.
- [`docs/comparison.md`](../docs/comparison.md) and [`examples/comparisons/`](../examples/comparisons/) provide full code listings and failure traces for the nine cross-framework benchmark suites summarized in the appendix, alongside a recurrent reinforcement learning comparison against Brax.

---

## Appendix: cross-framework benchmarks

Nine shared-task benchmark suites in `examples/comparisons/` evaluate identical tasks and budgets across `nodejax`, `Flax NNX`, `Flax Linen`, `Equinox`, `Haiku`, `PyTorch`, `Keras 3`, and raw `JAX` against framework-free task harnesses (`chunk` checks numerical trace parity against plain-JAX reference functions, while the remaining suites assert structural invariants, zero-drift weight sharing, mode isolation, transform probe matrices, and finite decreasing losses). Each row below records the suite name, the structural property tested, the `nodejax` measurement, and the corresponding range across the `class`-based implementations in that suite:

| Suite | Property tested | `nodejax` | `class` frameworks |
| :--- | :--- | :--- | :--- |
| `tower` | 8 stacked transforms + MAML | 58 lines, single algebra | 76–213 lines; exits module protocol |
| `ttt` | Online TTT on `MLP` and `RNN` | 53 lines, 1 wrapper | 92–105 lines; manual fast-weight carry |
| `chunk` | Multi-horizon state reset change | 1 `Node` edit (65 lines) | 2–6 loop edits (115–123 lines; `Haiku` 1 in 83) |
| `residual` | Library `residual` over all roles | 9-line transform (0 user lines) | 6–26 lines; split state protocols |
| `lift` | Lifted `stack` + `stack(stack(...))` | 41-line transform (all probes pass) | 41–42 lines; state / wrapper limits |
| `mode` | Stochastic train to frozen eval | 28 lines, acyclic evaluator | 30–64 lines; mutable flags or `State` |
| `generics` | Deferred sizing and specialization | 0 forwarded args (53 lines) | 4–5 forwarded args (61–129 lines) |
| `imu` | Input priming + noise + drift pipe | 43 lines, 1 `>>` expression | 61–70 lines; manual init / state wiring |
| `tie` | Tied input/output embeddings | 24 lines, 0 drift | 42–61 lines, 0 drift (capability tie) |

Line counts measure executable model, transform, and training code, excluding blank lines, comments, docstrings, imports, and test/reporting harnesses; `residual` and `lift` count the transform implementation itself rather than the shared diagnostic probe suite.

Empirical outcomes across the nine suites break down as follows:

- In the `tower` suite (`residual`, `stack`, `scanned`, running `Norm`, `ensemble`, `finetune`, `batch`, `trained`), `nodejax` (58 lines) stays inside one `Node` algebra, whereas `PyTorch` (76), `Equinox` (81), bespoke `Flax NNX` (98), and `Keras 3` (127, via stacked variable lists and `stateless_call`) exit their module protocols into manual closures, and reusable `Flax NNX` wrappers reach 213 lines before abandoning the protocol at MAML.
- In the `ttt` suite, a 3-line `next_step_ttt(train_step(...))` definition wraps both acyclic `MLP` and `cyclic` `RNN` predictors in `nodejax` (53 lines, versus a 47-line `RNN`-only raw-JAX baseline), whereas `PyTorch` (92), `Equinox` (96), and `Flax NNX` (104) branch on `hasattr` for carry initialization, `Haiku` (105) nests `hk.transform_with_state` inside `hk.lift`, and all four thread fast weights through the sequence loop by hand.
- In the `chunk` suite, moving a normalizer's reset boundary from outer run to per-recording is 1 `state_reinit` edit in `nodejax` (65 lines; `Haiku` is 1 rollout-loop edit inside `_recordings` in 83 lines that silently drops on reuse, while its multi-session trainer carries `state0` separately across raw `lax.scan` loops), versus 2 coordinated edits in `PyTorch` (115) and `Flax NNX` (118), 4 in `Flax Linen` (123), and 6 in `Equinox` (118), where omitting one edit silently computes stale statistics.
- In the `residual` and `lift` suites, `nodejax`'s library transforms (`residual`, 9 lines; `stack`, 41 lines) require zero custom wrapper code at the call site and pass every composition probe. At the framework level, `Flax NNX` wrappers (`Residual`, 6 lines; `LayerStack`, 41 lines) split between `Variable` mutation and `RNNCellBase` carries, while `Equinox` wrappers (`Residual`, 26 lines including its local `WithAux` protocol; `LayerStack`, 42 lines) split between `StatefulLayer` (`StateIndex` fails under `filter_vmap`) and `(successor, output)`; at the wrapper level, because neither framework records per-role `rng` metadata on modules, both `LayerStack` implementations infer stochasticity via `inspect.signature` and fail on `stack(stack(bare))`.
- In the `mode`, `generics`, `imu`, and `tie` suites, `class` frameworks match `nodejax` on single-network weight tying (`tie` is a five-way capability tie across `nodejax` at 24 lines, `Haiku` at 42, `Flax NNX` at 48, `PyTorch` at 50, and `eqx.nn.Shared` at 61), while requiring in-place flag mutation or manual `eqx.nn.State` threading in `mode` (28 vs. 30–64 lines), 4 to 5 forwarded constructor arguments in `generics` (53 vs. 61–129 lines), and manual stage-init wiring in `Flax NNX` (61 lines) or 4-site topology edits in `Equinox` (70 lines) in `imu` (43 lines in `nodejax`).

---

## References

1. R. Frostig, M. J. Johnson, and C. Leary (2018). *Compiling machine learning programs via high-level tracing*. Systems for Machine Learning (SysML). [`cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf`](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf)
2. JAX Developers. *`jax.example_libraries.stax` module documentation*. [`docs.jax.dev/en/latest/jax.example_libraries.stax.html`](https://docs.jax.dev/en/latest/jax.example_libraries.stax.html)
3. DeepMind Technologies Limited. *Optax: `GradientTransformation` API documentation*. [`optax.readthedocs.io/en/latest/api/transformations.html`](https://optax.readthedocs.io/en/latest/api/transformations.html)
4. JAX Developers. *Pytrees (`jax.tree` and custom pytree nodes)*. [`docs.jax.dev/en/latest/101/pytrees.html`](https://docs.jax.dev/en/latest/101/pytrees.html)
5. Flax Developers. *Flax Linen `flax.linen.Module` API reference*. [`flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html`](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html)
6. DeepMind Technologies Limited. *Haiku Basics (`hk.transform` and `hk.get_state`)*. [`dm-haiku.readthedocs.io/en/latest/notebooks/basics.html`](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html)
7. P. Kidger and C. Garcia (2021). *Equinox: neural networks in JAX via callable PyTrees and filtered transformations*. Differentiable Programming Workshop at NeurIPS 2021 (`arXiv:2111.00254`). [`ar5iv.labs.arxiv.org/html/2111.00254`](https://ar5iv.labs.arxiv.org/html/2111.00254)
8. Flax Developers. *Flax NNX Basics (`nnx.Module`, `nnx.Variable`, `nnx.split`, `nnx.merge`)*. [`flax.readthedocs.io/en/latest/nnx_basics.html`](https://flax.readthedocs.io/en/latest/nnx_basics.html)
9. A. Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems 32 (`arXiv:1912.01703`). [`ar5iv.labs.arxiv.org/html/1912.01703`](https://ar5iv.labs.arxiv.org/html/1912.01703)
10. Keras Team. *Keras 3: Multi-backend deep learning*. [`keras.io/keras_3/`](https://keras.io/keras_3/)
