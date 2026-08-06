# nodejax handbook

The explanatory companion to the pitch: patient rather than punchy, covering the unglamorous corners on purpose. Each section opens with the concept and states what the finished section covers; worked examples grow in over time. Sections marked (unsettled) describe design still in motion.

# Part I: core concepts

```
┌─────────────────────────────────────────┬─────────────────────────────────────────┐
│               Functions                 │              NodeJax Nodes                │
╞═════════════════════════════════════════╧═════════════════════════════════════════╡
│                                    Composition                                    │
╞═════════════════════════════════════════╤═════════════════════════════════════════╡
│         Composable Functions            │           Composable Nodes              │
│           (vanilla python)              │               (NodeJax)                   │
│ · · · · · · · · · · · · · · · · · · · · │ · · · · · · · · · · · · · · · · · · · · │
│     lambda x: multiply(add(x))          │   imu = Dt >> Dt >> noise >> drift      │
│                                         │   model = linear >> act >> projection   │
╞═════════════════════════════════════════╧═════════════════════════════════════════╡
│                            Composable Transformations                             │
╞═════════════════════════════════════════╤═════════════════════════════════════════╡
│  Composable Function Transformations    │    Composable Node Transformations      │
│             (vanilla jax)               │                 (NodeJax)                 │
│ · · · · · · · · · · · · · · · · · · · · │ · · · · · · · · · · · · · · · · · · · · │
│    jax.jit(jax.grad(jax.vmap(model)))   │      linear = ensemble(projection)      │
│                                         │        model = batch(stack(rnn))        │
└─────────────────────────────────────────┴─────────────────────────────────────────┘
```

## 1. The big picture

nodejax's shape mirrors jax's own, one level up. jax's world has two axes: pure functions compose with each other, and function transformations (`jit`, `grad`, `vmap`) apply to any function and compose with each other. nodejax lifts both axes to functions that carry params and state. A node is at most three pure functions against one rigid contract:

```
param : (param_input)           -> params
init  : (params, state_input)   -> state
apply : (params, state, input)  -> (state, output)
```

The three slots are binding times, and that is the organizing idea: statics bind at construction, params bind at `parameterize` and then hold still across calls, state evolves across steps within a run, input arrives fresh per call. A component is defined by which slots it uses, and the degenerate forms are first-class: a plant with no params, a layer with no state, a pure function with neither. Every node still has all three slots conceptually, which is what keeps every code path uniform.

Nodes compose into nodes (`>>`, `serial`, `composite`), and node transforms (`batch`, `ensemble`, `scan`, `train_step`) apply to any node and compose with each other. The rest of this handbook elaborates the four quadrants of that picture.

The section covers: exact signatures and return conventions, why cyclic apply returns `(state, output)`, and how the bound calling sugar hides the empty state of non-cyclic nodes.


## 2. State doctrine

State in nodejax is treated functionally, all the way down. No node is stateful in the object sense: a node with state is CYCLIC, a pure function that receives its state and returns the successor, `(state, input) -> (state, output)`. "Having state" means having a slot in that exchange, and the state itself is a value you hold, inspect, store, and pass back; the node never holds it for you. Composition passes state BETWEEN nodes rather than hiding it inside them: a composite's state is the named tree of its members' states, and the threading in and out of each member is derived from structure by default. Custom threading remains yours to write whenever the dataflow calls for it (a hand-wired composite does exactly that); the point is that 99% of the time you would rather not, and then you don't. That one decision is why a running normalizer, an RNN carry, a physics integrator, and a trainer's optimizer moments all compose in one pipeline with zero glue: they are the same kind of value in the same slot.

The section covers: what earns a place in state and what does not; seeds (`state_input`) versus the state itself; how mode flags dissolve (batchnorm eval is reusing a frozen state, dropout eval is the rate-zero build); the freeze family (freeze, tree_freeze, detach, tree_detach); scan's persist mapping for fast versus slow state; and nesting, where a wrapper's state holds its collaborators' states under named slots and seeds mirror those slots.

## 3. Composition

The point of composition is closure: putting nodes together yields a node, with nothing left over. `a >> b` produces a pipe whose params and state are trees named by member, whose step feeds each member's output to the next and threads every member's state in and out, and which satisfies the same contract as its parts, so it can itself be piped, transformed, or trained. The glue that dominates hand-rolled stateful systems, threading each component's state through every step, is exactly the part that is derived here rather than written.

The section covers: wires as bundles; member naming, automatic suffixes included; `parallel` for side-by-side strands; the aux channel, a second output riding alongside the wire for losses and diagnostics, with `split_aux` and `taps`; and what flattens across `>>` versus what stays atomic.

## 4. Node transforms

A node transform is a function from node to node: it consumes the three contract functions plus stored metadata and produces a new triple. Because every node declares which tree is params and which is state, a transform acts on ROLES rather than on any particular model: `batch` shares params and maps state per element because the roles say so, `ensemble` maps params per member, `scan` carries state over an axis. The deep transforms go further and move data BETWEEN roles: `train_step` demotes params to state (training is what that means), `ttt` does it per step, a feedback loop moves output into input. Each transform is written once, in a few dozen lines against the contract alone, is generic over every node, and returns a node, so transforms compose with each other and with everything else.

The section covers, one paragraph each: the axis family (batch, ensemble, stack, repeat), scan with persist and rng diversion, train_step, finetune, metasgd, ttt with reconstruction and the data-assembly doctrine, tie, freeze and detach, externalize, at, taps, residual.

## 5. Node, NodeDef, and binding

nodejax separates the program from the instance. A NodeDef is the program half: the three functions plus stored metadata, holding no array data. A Node is that program bound to a param pytree: the instance. The two share one calling surface, since every call a node answers, a def answers too with the param passed explicitly, so binding is an association for convenience, never a change in capability.

The section covers: `parameterize` versus `bind`; the def-answering calls (`d.apply(param, ...)`, `d.init(param, ...)`, `d.scan(param, ...)`); what flattening means in practice (a node flattens to exactly its params, so `jax.grad` with respect to a model is simply `jax.grad`, and two bindings of one def share a treedef, so jit caches hit and optimizer states line up); and when to hold a def versus a node.

## 6. FOOP: objects without classes

FOOP is Functional Object-Oriented Programming, the design stance underneath nodejax: the conviction that the purity JAX demands and the object ergonomics developers want are compatible, because neither requires what the other forbids. The systems this library serves are objects in the plain sense, complex bundles of immutable data with complex bundles of behavior acting on that data; an FOC controller is more than a function and more than a record. What object-ness does not require is Python's object model: mutation, reference identity, and the class machinery are incidental, and they are exactly the parts JAX cannot digest. A node keeps the essence and drops the accidents: the def carries the behavior, params and state carry the data as immutable pytrees, and the object is their association.

nodejax's object constructor is functions all the way: a def is built by lifting plain functions, specialization is function application, and the subclassing replacement is `derive`: override one function of an existing def, inherit the rest, and call the parent's contract function where you would call `super`. Methods live on defs and reach a bound node through them. Identity is value identity; there is no reference to hold, and therefore none to lose through a jax boundary.

The section covers: how derive recomputes a def's nature from its effective pieces, how methods bind, and how this object model compares to module classes and traced contexts.

## 7. The static distinction

A static is a value that decides the def's structure rather than flowing through it: sizes, rates, dt, architectural choices. It binds at construction and never enters a pytree, which aligns with jax's own traced-versus-static divide. The misclassification test runs in both directions: a static you wish you could train belongs in params (promoting one is how the examples meta-learn a learning rate), and a param that never receives gradients may really be a static. Where statics should ultimately live is NOT settled: closure captures are the current practice, ambient covers scope-wide values, and the generic stage is one contender for the rest.

## 8. rng doctrine

Two principles govern randomness. The first is referential transparency: entropy is a declared dependency, and you can tell what a function consumes by looking at its signature or call site. A function that draws names `rng`; a key is owed wherever one is declared and conjured nowhere; a component that initializes from explicit values simply does not declare rng, and nothing in the library assumes entropy where none is wanted. The second principle is that key management is boilerplate to remove, never a discipline to practice: the declared rng arrives as a scope-local mutable stream, every `rng.next()` yields a fresh key, and no line anywhere is dedicated to splitting, counting, or naming keys. The mutation is local to the scope, so purity is untouched; the stream is a deterministic function of the one key that crossed the boundary. Composition inherits both principles: the caller owes one key at the outermost boundary, and every composite splits it toward exactly the members that declared the need, ensembles per member, batches per element.

The section covers: the three homes of entropy (param construction, rng-as-state with auto-advance for streaming noise, apply-rng for per-call draws) and how to choose; seed stability under refactoring; and what is and is not reproducible.

## 9. Node kinds and the lattice

Two independent bits classify every node: parametric or not, cyclic or not. That gives a small lattice of four kinds, and transforms are moves in it: `scan` takes a cyclic node to a sequence-level one, `train_step` takes a parametric node to a cyclic non-parametric one (its params became state), `freeze` drops the cyclic bit. Reading an unfamiliar composition becomes tracking where each wrapper leaves you in the lattice, which turns "what does this stack of transforms even accept" from archaeology into arithmetic.

The section covers: the four kinds with an example each, and the lattice position of every library transform.

## 10. Time, streams, and axes

Axes carry meaning by convention, and the conventions are few. A stream is an input whose leading axis is time, its element the step-level input; `scan` consumes streams. `batch` adds a data axis, `ensemble` a member axis, `stack` a depth axis, and they nest predictably, which is how a committee of deep RNNs ends up with hidden state shaped `(MEMBERS, LAYERS, HIDDEN)` without anyone writing that shape down.

The section covers: the axis each transform adds and where, how scan elements relate to trainer stream elements (`input`, `target`), and reading composed shapes off a transform tower.

## 11. Bundles, specs, and validation

Every function in the contract takes its inputs as one declared pytree, its bundle, and the declaration is the signature itself: each free parameter name is a field, required when it has no default, optional carrying its default otherwise. The stored, inspectable form of that declaration is a spec, so every def can tell you exactly what its constructors and its apply expect before you call anything. Validation is loud and happens at the entry: unknown fields and missing required fields are errors at the call that supplied them, never a mystery downstream.

The section covers: the three input bundles (param_input, state_input, apply input) and the six-spec metadata surface; REQUIRED versus defaults; marker specs versus resolved specs; and the kwargs packing sugar with its one rule, a whole bundle or loose fields, never both.

# Part II: the authoring sugar

## 12. Authoring sugar and reserved names

The sugar has one job: turn the natural Python function you want to write into the contract-form function the def stores. It works by reading your signature, and a handful of reserved names act as delivery channels rather than bundle fields: `param`, `state`, `input`, `ndef`, `rng`, each delivering its thing in each of the three function kinds. Every free name is a bundle field, defaults make fields optional, and `*args`/`**kwargs` are definition-time errors, because the signature is the source of truth and an unreadable signature declares nothing. `self` is reserved too but is no data channel: it is the composite sugar's object view, covered in its own section. Everything the sugar produces you could write by hand against the raw contract; sugar is strictly a producer, never a requirement.

## 13. ndef: the def, delivered

A constructor may read its own definition: name `ndef` and the def arrives, resolved, bound just in time. This is reflection in general, with shape the most prominent use case (`ndef.input`, `ndef.apply_input_spec`, a fan-in read off the upstream), but specs, flags, and methods are equally readable. ndef never appears in the public contract; it is delivered into authored functions only, which is what keeps self-inspection an implementation convenience rather than an interface obligation.

The section covers: what resolution means here, the error you get when you read shapes nothing has resolved, and the three ways to fix it.

## 14. self and hand-wired composites

When `>>` is not the shape of your dataflow, `composite(apply, members=...)` lets you write the step against `self`: a scope-local, mutable, object-like view of the node, bound to the live params and state. Calling a member advances its state slice in place, reads see current values, and you write ordinary imperative wiring. Like the key stream, it is purely a scope-local abstraction: the sugar transforms your function into an ordinary pure apply, and to anything outside, only the node contract is visible.

The two spellings of the same def, side by side. The sugared form:

```python
def smoothed(gain, ema):
    def apply(self, input):
        return self.ema(self.gain(input))
    return composite(apply, members=dict(gain=gain, ema=ema), name='smoothed')
```

And the raw contract form the sugar produces, with the member threading explicit:

```python
def smoothed_raw(gain, ema):
    def apply_fn(param, state, input):
        _, u = gain.apply_fn(param.gain, (), input)          # non-cyclic member
        ema_state, y = ema.apply_fn(param.ema, state.ema, u)  # cyclic member
        return Struct(gain=(), ema=ema_state), y
    ...
```

Reading them together locates the sugar exactly: `self.gain(input)` is the member's contract apply plus the bookkeeping of slicing its param and state in and writing the new state back, and nothing else.

At two members the raw form is tolerable. The scale where it stops being tolerable is the real argument, so here is the shape of a real one: the FOC current controller from [`nodejax/examples/actuator/current_controller.py`](../nodejax/examples/actuator/current_controller.py), nine members including a stochastic current estimator, a noisy bus voltage sensor, and two one-tick memories, whose sugared apply reads as fifteen lines of imperative wiring. Desugared, abridged to its pattern:

```python
def current_controller_raw(dt, motor, bus_sensor, estimator, controller, fets, ff, limit):
    def init_fn(ndef, param, state_input, input=None):
        # one boundary key, split BY HAND toward the stochastic members
        # (the current estimator's sensor, the bus voltage sensor); the
        # split count is bookkeeping that must track the census of
        # stochastic members, and silently misroutes when it drifts
        k_est, k_bus = jax.random.split(state_input.rng, 2)
        return Struct(
            estimator=estimator.init_fn(param.estimator, Struct(rng=k_est)),
            bus_sensor=bus_sensor.init_fn((), Struct(rng=k_bus)),
            controller=controller.init_fn(param.controller, Struct()),
            fets=fets.init_fn(param.fets, Struct()),
            pwm_prev=DQ(), tgt_prev=DQ(),
            motor=(), ff=(), limit=())

    def apply_fn(param, state, input):
        # the bus arrives TRUE and is sensed here: noisy measurement,
        # its stream threaded like any other member state
        bus_state, bus = bus_sensor.apply_fn((), state.bus_sensor, input.bus)
        _, target = limit.apply_fn(param.limit, (), input.target)
        # methods (derate, voltage_terms) return values; contract applies
        # return (state, output) tuples
        target = fets.derate(param.fets, state.fets, target)
        est_state, i_est = estimator.apply_fn(param.estimator, state.estimator,
            Struct(value=input.current,
                   model=Struct(motor=param.motor,
                                v_mod=state.pwm_prev * bus,
                                velocity=input.velocity)))
        ctrl_state, v_fb = controller.apply_fn(param.controller, state.controller,
                                               target - i_est)
        di_ref = (target - state.tgt_prev) / dt
        terms = motor.voltage_terms(param.motor, target, di_ref, input.velocity)
        _, v_ff = ff.apply_fn(param.ff, (), terms)
        pwm = ((v_fb + v_ff) / jnp.maximum(bus, 1e-3)).clamp_norm(1.0)
        fets_state, _ = fets.apply_fn(param.fets, state.fets, i_est.norm2())
        return Struct(estimator=est_state, bus_sensor=bus_state,
                      controller=ctrl_state,
                      fets=fets_state, pwm_prev=pwm, tgt_prev=target,
                      motor=(), ff=(), limit=()), pwm
```

Every line that mentions `param.X` or `state.X` twice is threading, the collect Struct at the return must name every member every time, and the key split in init is a census that nothing checks. The sugared form contains none of it, and the rng census the raw init maintains by hand is derived there from which members declare rng, which is why adding a stochastic member to the sugared controller is one line and to the raw one is four edits.

The section covers: what self is not (it does not survive the call), when to wire by hand versus reaching for serial and parallel, and what a wired composite still owes the contract (specs, member exposure).

## 15. Shape inference: input versus with_input

Two different mechanisms share the word "input". `init(..., input=value)` feeds a REAL value, priming state from a first sample, the way a derivative's register should start at the first reading rather than at zero. `with_input(x)` binds a SPEC, telling a def its input shape so shape-reading constructors can run; no value is stored, only the shape. Pipes resolve member shapes by walking the composition, each member's def resolved against what its upstream produces, which is why one `with_input` at the boundary sizes every layer inside.

The section covers: when resolution happens for you, when you will be asked, the resolve-what-you-wrap rule for wrappers like train_step, and a maturity note: this machinery is younger than the rest and fails toward explicitness.

## 16. Ambient statics

Some construction values, dt above all, are needed by a dozen factories across a construction graph, and threading them through every call is noise. `ambient` is dynamic scope for exactly this, tightly fenced: `@ambient` declares eligibility at the definition site, `with ambient(dt=1e-4):` supplies values at the point of use, explicit arguments always win, and nothing exists past construction time, so the functional semantics of nodes are untouched.

The section covers: when to reach for it (a value many factories need) and when not to (anything a single call site can pass).

## 17. Generics (unsettled)

The deferred-construction stage: composing factories before their statics are known, then configuring the composed tree at one point (`mlp.specialize(linear={...})`), with `*.name` broadcasts and transform commutation over unspecialized families. Status: under design review; the guidance until it settles is closures first, ambient for scope, and this section documents what exists rather than what is promised.

# Part III: in practice

## 18. Writing your own wrapper

The escape hatch is the public surface: every transform in the library consumes the three functions plus stored metadata and produces the same, so when you outgrow the provided combinators you write the same kind of function the library's own transforms are. The recurring pattern in ambitious wrappers is channel crossing: a trainer carries params as state, ttt does it per step, byol binds weights from a sibling's state. The recipe that keeps a wrapper well-behaved: seeds nested under slots mirroring your state fields, boundary rng via `hoist_rng`, and computed specs declared to `derive` or `node_def` rather than guessed from a signature.

The section covers: when to be a Composite (member-keyed trees, channels intact) and when to be a leaf, with worked wrappers from the examples.

## 19. Structural rewriting

Beyond transforming behavior, you can rewrite structure: the def tree is data too. `map_members` rebuilds a composition bottom-up through each composite's rebuild recipe; `tree_freeze`, `tree_detach`, and `tree_filter` select members by name or spec, and a selector that matches nothing raises, never a silent identity. Composites are transparent to these walks; transform-produced defs and hand-wired leaves are opaque, and opacity is sometimes the right answer, since a batched pipe's trees genuinely are not member-keyed. (unsettled: pass-through rebuild for the axis transforms.)

## 20. Errors and the loudness doctrine

The philosophy in one line: absence is never silently defaulted. A missing required field, an unknown field, a selector that matches nothing, a shape that conflicts with a declared spec, a signature the sugar cannot read: each fails at the site that caused it, with a named error, at the earliest time it can be detected, definition time where possible. The payoff is that the distance between a mistake and its error message stays short, which in composed systems is most of debugging.

The section covers: the definition-time rejections (varargs, rng defaults), the bundle validation errors, and how to read each common message.

## 21. Interop

Contract functions are ordinary functions over ordinary pytrees, so the boundary with the rest of the world is thin: use them under raw jax directly, wrap any pure function as a node in a few lines, and move params in and out as trees (checkpoints, other frameworks' weights). Struct, the record pytree everything speaks, is the one data type worth knowing: field access, indexing, `replace`, `without`, and how it differs from a dict.
