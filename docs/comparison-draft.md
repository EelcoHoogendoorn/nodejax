# Comparing functional boundaries

> NodeJAX introduces no new execution model; it gives JAX's existing
> functional model a compositional object form.

Every framework here reaches a functional boundary through a different native
abstraction. The comparison asks where each program records state lifetimes
and transform policy, and whether that information remains usable when
transforms nest.

NodeJAX records param, state, input, and static structure at every component
boundary. Its transforms consume components and return components. The
examples below test the practical value of that choice against current, native
formulations in Equinox, Flax Linen, Flax NNX, Haiku, and PyTorch. Every source
comparison is runnable.[^validation]

## Where the functional boundary lives

| Library | Working representation | Transform boundary |
| :--- | :--- | :--- |
| Raw JAX | Functions and PyTrees | Values are passed explicitly to ordinary JAX transforms. |
| Equinox | A Module is a PyTree | Filtered transforms trace array leaves and treat non-array leaves as static by default; `eqx.nn.State` holds stateful-layer values. |
| Flax Linen | Module definition plus variable collections | `init` and `apply` form the pure boundary; lifted transforms describe collection behavior. |
| Flax NNX | Reference-aware Module graph with typed Variables | NNX transforms accept Modules directly and preserve graph updates; `StateAxes` describes mixed axis behavior when needed. |
| Haiku | Modules authored inside a transformed function | Haiku-aware transforms operate inside that context; the resulting pure `init` and `apply` functions compose with JAX outside it. |
| PyTorch | `nn.Module` with Parameters and buffers | `functional_call` evaluates explicit parameter and buffer mappings; `torch.func.grad` and `vmap` transform functions over those values. |
| NodeJAX | Component definition with distinct param and state values | The component contract is established before composition; component transforms preserve it. |

These are different placements of a functional boundary, not a division into
functional and nonfunctional libraries.

They are not interchangeable once transforms nest. If the boundary exists
only around a root program, or must be restated at every transform, then inner
composition has to reconstruct information that its members already know.
NodeJAX's bet is that the functional boundary belongs on every component, once.

## Primary case: state across nested boundaries

The [`chunk`](../nodejax/examples/comparisons/chunk/) family puts several state
lifetimes inside one program. A recurrent model processes samples inside
chunks, chunks inside recordings, and recordings inside a larger training run:

- recurrent hidden state crosses chunk boundaries and restarts for each
  recording;
- running normalization statistics cross both boundaries;
- parameters survive execution and change through training;
- an outer boundary independently resets calibration and optimizer state.

The design comparison is in how those facts are represented:

| Library | Formulation |
| :--- | :--- |
| [NodeJAX](../nodejax/examples/comparisons/chunk/chunk_nodejax.py) | Reset behavior is attached to the component that owns the state. Nested scan transforms carry the rest. |
| [Flax NNX](../nodejax/examples/comparisons/chunk/chunk_flax.py) | Typed Variables live on the Module, but lifetime changes are assignments in enclosing scan bodies that reach into those Variables. `StateAxes` is also restated at each scan. |
| [Flax Linen](../nodejax/examples/comparisons/chunk/chunk_flax_linen.py) | Every lifted scan restates collection policy. The enclosing structure also knows the collection paths that hold member state. |
| [Equinox](../nodejax/examples/comparisons/chunk/chunk_equinox.py) | Module and state are clean PyTree values, but enclosing functions construct their carries and decide which state is replaced at each boundary. |
| [Haiku](../nodejax/examples/comparisons/chunk/chunk_haiku.py) | `transform_with_state` supplies a pure root boundary, while lifetime changes remain lines in the enclosing rollout functions. |
| [PyTorch](../nodejax/examples/comparisons/chunk/chunk_torch.py) | The runner reaches into buffers and local carries to implement each lifetime. The Module does not declare them. |

In the NodeJAX source, a nested member owns its reset declaration. The
enclosing scan names the event without naming an internal state path. Anything
less is a leaky abstraction: if the outer loop must locate or reset a member's
state, it depends on the internal layout of the program it is meant to contain.

## Capstone: nested transforms

The [`tower`](../nodejax/examples/comparisons/tower/) family combines several
operations in one differentiable program:

- a residual RNN stack;
- a committee;
- a recurrent rollout;
- task-local inner adaptation;
- second-order MAML;
- outer Adam training.

The source contrast is about reuse:

- [Flax NNX](../nodejax/examples/comparisons/tower/tower_flax.py) constructs
  mapped Modules with `nnx.vmap`, scans Modules directly, clones each task's
  fast model, differentiates it with `nnx.value_and_grad`, and updates both
  inner and outer models with `nnx.Optimizer`. Axis behavior remains explicit
  through `in_axes`, `Carry`, and `StateAxes`. The
  [reusable variant](../nodejax/examples/comparisons/tower/tower_flax_reusable.py)
  factors those policies into functions built around a recurrent call
  interface. NNX keeps its Module graph throughout, which is a genuine
  strength. Its cost is that each transform site still legislates how every
  kind of Variable crosses that transform.
- [Equinox](../nodejax/examples/comparisons/tower/tower_equinox.py) benefits
  from Modules already being PyTrees. Recurrent values and optimizer state are
  explicit scan carries in this formulation. Differentiating a model is
  natural; composing state lifetimes remains the caller's work.
- [PyTorch](../nodejax/examples/comparisons/tower/tower_torch.py) retains its
  Module as the static definition, but the differentiable fast model leaves
  that object. It becomes an explicit name-to-tensor mapping evaluated by
  `functional_call` under `torch.func.grad`. The ordinary stateful Module and
  optimizer paradigm does not survive the inner adaptation boundary.
- [NodeJAX](../nodejax/examples/comparisons/tower/tower_nodejax.py) reads param,
  state, and input roles from the component contract. `stack`, `ensemble`,
  recurrent scanning, inner training, and outer training all return components
  that can be nested again.

## Adaptation as evolving state

The [`ttt`](../nodejax/examples/comparisons/ttt/) family asks a sharper
question. Test-time training treats an inner model's weights as values that
change during the forward pass. Those fast weights advance beside recurrent
hidden state, while the outer optimizer learns their initial values and update
rates.

The implementations place that adaptation at different boundaries:

- [raw JAX](../nodejax/examples/comparisons/ttt/ttt_rnn_by_hand.py) carries
  fast-weight and recurrent PyTrees through `lax.scan`;
- [Flax NNX](../nodejax/examples/comparisons/ttt/ttt_rnn_flax.py) clones the
  wrapped recurrent model, a `Forecast` containing a stock `SimpleCell`, then
  carries the clone with `nnx.scan`, differentiates Params with `nnx.grad`, and
  writes learned per-weight updates with `nnx.update`. The adapted model remains
  an NNX Module;
- [PyTorch](../nodejax/examples/comparisons/ttt/ttt_rnn_torch.py) keeps the
  Module only as a callable template. The learner that actually evolves is a
  name-to-tensor mapping carried outside the object, evaluated with
  `functional_call`, and differentiated with `torch.func.grad`;
- [Equinox](../nodejax/examples/comparisons/ttt/ttt_equinox.py) carries the
  fast Module itself as a PyTree, which makes this adaptation natural, while
  [Haiku](../nodejax/examples/comparisons/ttt/ttt_haiku.py) moves fast values
  out of its ambient parameter context for the local update loop;
- [NodeJAX](../nodejax/examples/comparisons/ttt/ttt_nodejax.py) applies one
  generic TTT component transform to an inner component, moving its declared
  params into evolving state for the duration of the computation.

This is not a cosmetic spelling difference. PyTorch and Haiku represent the
same learner twice: once in their ordinary Module or parameter context, and
again as explicit fast values inside adaptation. Their functional APIs make
the computation possible, but do not preserve the original state abstraction
through it. Equinox and NNX avoid that break in different ways. NodeJAX goes
further by making adaptation a component transform that reads the same param,
state, and input roles as every other transform.

## One value used twice

The [`tie`](../nodejax/examples/comparisons/tie/) family studies shared input
and output embeddings across component boundaries. A deliberately naive
Equinox variant demonstrates how repeated array leaves can diverge. A second
variant uses `eqx.nn.Shared`, which is Equinox's intended mechanism.

The correct mechanisms differ:

- NNX and PyTorch preserve reference identity in their graph or module
  representation.
- Haiku reuses one parameter path.
- Equinox uses `eqx.nn.Shared` to remove and restore the shared part.
- NodeJAX's `tie` rewrites the param structure so one stored value is routed to
  both declared consumers.

NodeJAX's distinctive choice is structural rather than referential. The plain
param PyTree itself contains one value.

## Configuration after composition

The [`generics`](../nodejax/examples/comparisons/generics/) family defines a
committee of deep towers, then varies width, depth, member count, and a nested
temperature. It also changes that temperature after training without rebuilding
the architecture from scratch.

NodeJAX leaves selected statics unbound while composing the architecture, then
specializes them by tree path. The NNX, Equinox, and PyTorch examples pass the
same values through constructors and read them back from built objects. Both
approaches work.

The NNX, Equinox, and PyTorch formulations forward configuration through
intermediate constructors. NodeJAX instead addresses unbound statics by their
place in the component tree. It removes forwarding-only arguments, but pays
for that reduction with tree-addressed names that must remain intelligible and
stable.

## Smaller exhibits

The [`mode`](../nodejax/examples/comparisons/mode/) family compares structural
rebuilding, returned PyTrees, object mode changes, and explicit call arguments
for dropout and running statistics.

The [`imu`](../nodejax/examples/comparisons/imu/) family composes finite
differences, random noise, drifting bias, and quantization. NNX scans its Module
graph, Equinox carries explicit functional state, and NodeJAX combines the
state declared by the leaf components.

## A concrete lifetime edit

Consider one change to the two-boundary program: normalization statistics
should reset for each recording instead of crossing recording boundaries. The
change must apply to both inference and training. Counting independent
executable locations in the checked sources, without first refactoring them
into new helpers, gives:

| Framework | Locations changed | Where the lifetime is expressed |
| :--- | ---: | :--- |
| NodeJAX | 1 | The `Norm` member's `state_reinit` declaration, reused unchanged under training. |
| Flax NNX | 2 | The recording scan bodies in inference and training. |
| Flax Linen | 4 | Lifted-scan collection policy and state-tree construction in inference and training. |
| Equinox | 6 | State initialization, returned carry, and outer carry construction in inference and training. |
| Haiku | 1 | The shared recording rollout body, which resets ambient state by name. |
| PyTorch | 2 | The recording loops in inference and training, each reaching into Module buffers. |

The location matters as much as the count. Haiku also changes in one place,
but that place is the enclosing rollout and it names state owned by a nested
member. NodeJAX changes in the member declaration itself. That is the difference
between factoring repeated code and preserving an abstraction boundary.

## Critical appraisal

Everything above tries to separate executable facts from interpretation. What
follows is deliberately judgmental.

My reading is not neutral: NodeJAX has the strongest abstraction here for
stateful components under nested transforms. The lifetime edit is the decisive
exhibit. In the checked sources, an outer runner that must know the buffer names
or state-tree paths of a nested member is coupled to the layout that composition
was supposed to contain. The edit count is not a beauty contest. It is
maintenance surface, and the location of the edit reveals who owns the fact.
Haiku reaches one location through good factoring; NodeJAX puts the rule on the
component whose state it governs. Those are not equivalent.

The nested-transform examples make the same case from another direction. In
NodeJAX, the products of `stack`, `ensemble`, scanning, inner training, and
outer training remain components. That is a real abstraction, not shorter
spelling for a carry assembled elsewhere. The framework has identified one
boundary that survives the programs JAX users actually build, including the
awkward ones where learned values become temporary evolving state.

NodeJAX does not eliminate policy. Components declare param, state, input, and
boundary behavior; generic transforms define what those roles do in each
relationship. That division is exactly the product claim, but it also means
this source comparison leaves substantial library-side machinery out of view.
The examples establish less application routing, not a simpler total system. A
mistaken component declaration can propagate the wrong behavior just as
systematically as a correct one.

Nor can a component contract infer a relationship that has not been declared.
Current NodeJAX transforms apply uniform role-wide policies; a member that mixes values
which should be shared and mapped by the same transform must be factored into
finer components or use a more specific transform. I often regard that as
healthy pressure toward explicit structure, but it is still a constraint. NNX
can state such mixed behavior directly with `StateAxes`.

PyTorch has the largest conceptual fracture in these examples. Its Module
model is excellent for ordinary eager work, then stops being the model once
fast weights evolve under differentiation. In MAML and TTT, the live learner is
a name-to-tensor mapping passed to `functional_call`; the Module is a template
for interpreting it. `torch.func` makes the computation possible, but it asks
the user to switch to a JAX-like explicit-state program inside an object system
that no longer describes the computation's changing values. Calling that
seamless Module composition would conceal the important fact.

Haiku makes a related trade more consistently. It admits that the durable
functional artifact is the transformed root's `init` and `apply`, not the
individual Module authored inside it. That is conceptually cleaner than
pretending the object survives every transform, but it still means a nested
component cannot carry its complete semantics independently of the enclosing
transform context. Linen is powerful and principled, but its lifted-transform
policy is bureaucratic: callers repeatedly specify how collections and random
streams cross each transform. Explicitness is valuable; repeatedly restating
facts that belong to a member is leakage.

NNX is the strongest counterargument to NodeJAX. Its current transforms really
do preserve Module graphs, typed Variables, sharing, and updates. Any claim
that object-preserving transforms are unique to NodeJAX is simply false. An
NNX scan can be defined as a reusable callable or Module method, and `nnx.RNN`
packages one behind a normal Module interface. Definition and execution need
not be textually fused. The narrower difference is the product of the generic
transform: `nnx.scan` transforms a callable over a live graph, while NodeJAX
transforms the component definition and returns another component before
params or state have been bound. General Module packaging and axis policy for
the graph remain additional NNX authoring decisions.

The checked NNX examples expose a deeper split. In `chunk`, normalization
statistics are mutable Variables while recurrent hidden values are explicit
JAX carries. In `tower`, the running statistics join the explicit carry because
both they and the hidden values should restart for each rollout. Params and
optimizer values remain graph Variables. These are valid formulations, but
they give one program two state vocabularies: an update may appear in a returned
carry or be propagated through object identity. A reader must inspect both the
function interface and the Module graph to find the complete evolving state.

That split provides two useful lifetime defaults, not a general lifetime
model. An explicit carry usually belongs to one invocation; a Variable usually
persists with its graph object. Nested programs need finer distinctions. A
value may cross chunks but restart for each recording, survive adaptation steps
but restart for each task, or follow a boundary unrelated to the call that
happens to contain it. `StateAxes` says how selected Variables cross one
transform. It does not say when they restart or which component owns that
decision. Once the two defaults are insufficient, lifetime policy returns to
enclosing bodies. A purpose-built Module method or wrapper can hide those
paths, but that is component-specific factoring rather than a generic lifetime
declaration understood by later transforms.

NodeJAX keeps those concerns separate. State always advances as a value;
component declarations determine how it responds to named boundaries. The
outer composition names the boundary, while the member that owns the state
defines the effect. In my view this is both a more complete lifetime model and
a less leaky abstraction than choosing between mutable storage and explicit
carry.

NNX's reference graph still buys something real. It naturally preserves
aliases, cycles, and shared mutable objects with several methods. Pure state
loses no computational expressiveness, and NNX itself must present object
updates as value transformations to JAX, but a plain value tree is not a
reference graph. NodeJAX deliberately prefers structural sharing and explicit
successor values. That makes branching, resetting, and inspecting state
simpler, while general shared mutable identity requires a different structural
formulation.
For the programs tested here I prefer that trade, but it is a trade rather than
an unqualified dominance claim. NNX is also likely the better choice today for
many conventional neural networks.

Equinox is the best argument for wanting less framework. A Module that is just
a PyTree is elegant, direct, and easy to differentiate. For research code with
one obvious loop and uncomplicated state, I would often choose it over
NodeJAX. Equinox leaves orchestration to ordinary functions and explicit
carries. That is clarity at small scale and repetition when boundaries nest.
Whether it is a cost depends on the program, which is why the `chunk` example
is more informative than a feed-forward model comparison.

The case against NodeJAX is serious. In practice it is a small DSL whose syntax
is ordinary Python, with little production history, tooling, or shared
community knowledge. Its [authoring layer](../nodejax/authoring.py) assigns
semantics to Python argument names such as `param`, `self`, `state`, `node`,
and `rng`. That is concise, but it is also magic: renaming an argument can
change the component's meaning in ways a type checker will not explain.

Tree-addressed specialization is powerful but string-like and hostile to
careless refactoring: structural names become an interface, and rebuilding the
tree can retarget configuration. Named boundaries and tags form another schema;
if they proliferate, local declarations can become a global protocol.

Automatic routing creates another obligation. When it works, difficult
programs become startlingly small. When it fails, a user may have to understand
both the JAX error and the machinery that generated the transformed program.
NodeJAX therefore needs exceptional structural inspection, error messages, and
debugging tools. Without them, the elegant surface risks becoming a puzzle box.

Interop is another cost. Existing JAX, Flax, or Equinox code does not acquire
the NodeJAX contract automatically; somebody must adapt it. Deferred input
binding makes generic composition possible, but can move shape and
configuration failures later than eager construction. The examples do not
establish serialization, distributed training, or production tooling at
scale, all areas where the established libraries have a large head start.

The examples also select for NodeJAX's thesis. They stress mixed state
lifetimes, nested transforms, structural sharing, and adaptation because those
are the problems the project claims to solve. That is a legitimate way to test
a design, but it is not evidence that every model benefits from the design. A
conventional feed-forward network with one training loop will often be easier
to explain, debug, and hand to another engineer in NNX, Equinox, or PyTorch.
The rival implementations use current native APIs, but they were written for
this repository rather than reviewed by each framework's maintainers. They are
credible checked formulations, not a ceiling on what an expert could design.

Even `ensemble(Hyperplane(), n)`, the project's most academic one-line
provocation, is not an exclusive capability result. Linen can produce a
transformed Module class, NNX can package mapped construction and application
as a Module, and the exact Equinox example can appear to work through matrix
rank behavior. The stronger practical exhibit is the residual tower. NodeJAX
writes `stack(residual(RNN(...)), ...)`; the reusable NNX source puts the
residual addition inside the cell body. Every framework can define a residual
wrapper, so this is not a capability verdict. It is a judgment about what an
abstraction invites. When wrapping an arbitrary component preserves its main
contract with negligible friction, structural relationships tend to become
visible composition instead of arithmetic embedded in member code. The NodeJAX
slogan is earned by that low-friction closure, especially for stateful members.

My conclusion is narrow but strong. For ordinary model building, NodeJAX is
not yet the default I would recommend. For programs in which nested state
lifetimes and repeated higher-order composition are the architecture rather
than an occasional technique, it is the best design in this comparison. The
core idea is real. Whether NodeJAX becomes broadly compelling depends on making
its magic inspectable and ensuring the small surface does not conceal a private
language users must learn by failure.

## Framework references

- [JAX key concepts](https://docs.jax.dev/en/latest/key-concepts.html)
- [Equinox overview](https://docs.kidger.site/equinox/),
  [stateful operations](https://docs.kidger.site/equinox/api/nn/stateful/), and
  [`Shared`](https://docs.kidger.site/equinox/api/nn/shared/)
- [Flax Linen lifted transforms](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/transformations.html)
- [Flax NNX transforms](https://flax.readthedocs.io/en/stable/guides/transforms.html)
- [Haiku transforms](https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html)
- [PyTorch function transforms](https://docs.pytorch.org/docs/stable/func.html),
  [`stack_module_state`](https://docs.pytorch.org/docs/stable/generated/torch.func.stack_module_state.html), and
  [`functional_call`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html)

[^validation]: The examples are executable correctness and source exhibits,
    not accuracy, timing, memory, or performance comparisons. The chunk family
    is checked against shared plain-JAX references, the tower formulations have
    convergence checks, and the remaining examples check or display their
    defining behavior. They were run on 20 August 2026 with Python 3.14.6, JAX
    0.10.2, Equinox 0.13.8, Flax 0.12.8, Haiku 0.0.17, PyTorch 2.13.0, and Optax
    0.2.8. The lifetime-edit variants were also executed; the census excludes
    comments, checks, reporting, and dead state routing. Install the optional
    dependencies with `pip install -e '.[test,comparisons]'`, then run
    `python -m pytest nodejax/tests/test_examples_run.py::test_the_comparison_reproduces_its_reference -q`.
