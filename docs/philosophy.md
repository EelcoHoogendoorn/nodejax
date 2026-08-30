# NodeJAX: Nodes That Compose Like JAX Functions

## JAX already has the execution model; NodeJAX gives it composable structure

NodeJAX is a purely functional, compositional object layer for stateful
differentiable programs in JAX. It introduces no new execution model. It
structures JAX programs so that meaningful pieces remain easy to combine.

JAX transformations compose functions. A function can be differentiated,
vectorized, compiled, or carried through time, and the result is still a
function. NodeJAX asks for the same closure from larger program units.

NodeJAX calls its compositional unit a **Node**, emphasizing its organizing
role in the JAX computation graph rather than any particular application
domain. A layer is a Node, but so is a controller, simulator, estimator,
optimizer, training step, or complete system. Composition connects Nodes.
Transformations reinterpret Nodes. Both produce Nodes again.

| | JAX functions | NodeJAX Nodes |
| :--- | :--- | :--- |
| Closure under composition | `f = lambda x: sum(sigmoid(x))` | `node = nn.RNN(64) >> nn.relu` |
| Closure under transforms | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

Python functions compose, and JAX transformations return functions. NodeJAX
extends that property to Nodes and Node transformations.

## A Node carries the distinctions JAX transformations need

The Node contract is not an alternative to JAX. It is the smallest
compositional boundary that retains the information JAX transformations
repeatedly ask for.

Consider a simple authored Node:

```python
@node
def LowPass(width, dt=0.01):
    def param(rate=1.0, gain=1.0):
        return Struct(rate=rate, gain=gain)

    def init():
        return jnp.zeros(width)

    def apply(param, state, input):
        next_state = state + dt * param.rate * (
            param.gain * input - state
        )
        return next_state, next_state

    return Leaf(apply, param=param, init=init)
```

None of those distinctions is framework taxonomy for its own sake. Each is
present because a JAX transformation requires it:

| JAX primitive | What JAX requires | What the Node declares |
| :--- | :--- | :--- |
| `jax.jit` | Program structure must be separated from traced values | `width` and `dt` are statics; an open `width` remains recorded in a `Generic` until the program can be built |
| `jax.grad` | Differentiable values must be explicit arguments | `rate` and `gain` form the parameter PyTree |
| `jax.lax.scan` | The step must have the shape `(carry, input) -> (next_carry, output)` | `state` and `input` occupy exactly those positions in `apply` |
| `jax.vmap` | Mapped and shared axes must be chosen independently | Params, state, and present input remain distinct roles |

The Node contract is therefore a compact description of how the Node
participates in JAX transformations. No transform has to recover that meaning
from array positions or object internals.

## Transform products are first-class Nodes

The litmus test for composition is not whether a transform can manipulate the
right arrays. JAX already makes that easy. The test is whether it preserves the
Node contract: can the result be composed, bound, and transformed again without
its caller knowing how it was produced?

Consider a scalar affine unit:

```python
def Projection():
    def param(node, rng):
        width = node.input.shape[-1]
        return Struct(
            w=jax.random.normal(rng.next(), (width,)) / jnp.sqrt(width),
            b=jnp.zeros(()),
        )

    def apply(param, input):
        return input @ param.w + param.b

    return Leaf(apply, param=param)
```

`Projection` maps one vector to one scalar.

A vector-valued linear layer can then be written as a population of scalar units:

```python
linear = ensemble(Projection(), n=3)
```

Stacking the arithmetic is easy. The usual blockers are more mundane: input
shape must reach each scalar unit, parameter construction must acquire the new
axis, random initialization needs independent keys, and any state must follow
the same transformation. The result must then publish an ordinary output
contract so the next Node can consume it. In most frameworks, some of
that information falls out of the abstraction, and the caller supplies the
missing glue.

That is the practical meaning of a first-class transform product. `ensemble(Projection(), n)` pipes through `>>`, batches under `batch`, trains under `train_step`, and accepts further transforms without manually forwarding shapes, params, state, or RNG. If transform composition makes writing a useful primitive attractive rather than merely possible, composition friction has actually been removed.

## FOOP gives functional programs object structure

NodeJAX calls its object model **Functional Object-Oriented Programming**, or
FOOP.

JAX transforms pure functions over PyTree values. Conventional Python OOP—the
contrasting **POOP**—has inherently mutable semantics. Frameworks that begin
with the latter must recover the former at the JAX boundary, and the mismatch
returns as split/merge passes, string-keyed variable collections, filtering
rules, or ambient transformation contexts. These are not unrelated API quirks;
they are the recurring cost of clashing object models between Python and JAX.

FOOP starts from the other end. Every Node is functional at the boundary where
it is defined. A finished object graph never has to be dismantled so that JAX
can discover which values are structure, params, state, or input; its Nodes
already say so.

FOOP keeps hierarchical composition, dot-notation inspection, encapsulation,
named methods, and reusable definitions. It makes composition, generic
construction, and ordinary transformations the primary mechanisms for
architectural reuse.

Importantly, POOP squats on the natural binding site for static arguments:
Python gives the class header a dedicated block for inheritance, including
multiple inheritance. For differentiable architecture, that allocation is
backwards: composing and specializing structure are needed by virtually every
application, while even single inheritance is hardly ever used. NodeJAX does
provide a `derive` operation, but it is not given such unearned prominence.

In POOP, all construction is pushed into an eager `__init__`, which fills a
mutable object dictionary with disjoint kinds of things: architectural statics,
trainable arrays, evolving state, child objects, caches, and incidental
bookkeeping. Those share a binding site because Python provides one, not
because they share a lifetime or transformation semantics. A framework must
later separate them again before JAX can do useful work.

## One state model covers every kind of evolution

In JAX, sequential evolution has one foundational form:

```text
(carry, input) -> (next_carry, output)
```

NodeJAX organizes its state model around that same reality:

```text
apply(param, state, input) -> (next_state, output)
```

Any other framework representation of state must eventually take this form to
participate in a scan. NodeJAX uses it directly and provides the syntax and
utilities to work with it.

An RNN hidden value, BatchNorm statistics, optimizer moments, a simulator's
velocity and temperature, and an autoregressive KV cache are the same kind of
thing at this level: each is the value the next step must receive from the
previous step.

They share one state mechanism because they evolve in the same mathematical
way, not because they mean the same thing. There is no separate mutable-module
state, scan carry, optimizer store, and simulation context to reconcile.

Calling a state-bound cyclic-Node does not mutate it. It returns a successor:

```python
advanced, output = model(input_data)
```

## State lifetimes are declared where the state lives

A uniform state model does not imply that all state lives equally long. An RNN carry may reset at the start of every episode-aligned scan call while running statistics continue across calls; a simulated plant may restart for each trial-aligned call while adaptive calibration persists.

Lifetime belongs to the meaning of the state. An enclosing temporal structure names structural call boundaries such as episodes or trials; each state-owning Node decides how its own state responds when that call begins. The loop need not inspect member state, and the member need not know which loop or training process contains it.

This keeps recurrence generic for scan-aligned lifetimes while reset policy travels with the state it describes. A stateful Node can be composed, wrapped, or moved without forcing callers to reconstruct that policy from its internal layout.

The choice between functional state or State-like containers when writing code in other frameworks is often driven by lifetime-management concerns; running stats persist so they are more convenient as a persistent container. Without an attached scan-boundary declaration, structural reset choices must instead be reconstructed at call sites.

## Generic Nodes make reusable libraries possible

A reusable Node cannot know the dimensions, modes, and surrounding
architecture of every application that may use it. If those choices must all
be fixed at construction, every enclosing Node must expose and forward
them. Deep composition becomes parameter drilling, and reuse gives way to
copy-and-edit variants.

An architecture with open statics is a `Generic`: not yet executable, but
still available for composition. A larger structure can therefore be written
before every choice below it has been fixed.

```python
block = nn.LayerNorm() >> nn.Attention() >> residual(nn.MLP())
tower = stack(block, n=12)

model = tower.specialize(
    **{'*.width': 768, '*.heads': 12, '*.ratio': 4}
)
```

Generic construction keeps decisions open until the information that decides
them exists. Some arrive through specialization; dimensions implied by
dataflow can arrive from input shape. Enclosing Nodes need not drill every
nested choice through their own constructor merely to keep an architecture
reusable.

This lets a library publish the architectural idea itself—a residual block,
attention mechanism, recurrent cell, observer, or controller—rather than one
project's concretely sized instance of it. Users can compose that idea into a
larger system before supplying application-specific facts. The library author
does not have to predict every context, and every wrapper does not have to
mirror every option below it. Without that, source code is reusable only by
being copied, edited, or wrapped in parameter plumbing. A Generic Node is
reusable as an architectural component.

## Training belongs to the same compositional system

Training is usually presented as an external harness around the real model.
Mathematically, it is another stateful process.

```python
trainer = train_step(model, loss_fn, optax.adam(1e-3))
```

The model's initial weights are params to the training process; its active weights and optimizer moments are state. This relative change of lifetime is enough to bring training inside the same algebra. JAX already makes a training loop a scan, but `lax.scan(step, initial, sequence)` combines recurrence declaration and execution by API convention, not functional necessity. Because the Node contract already identifies state and input, NodeJAX can make `scan(step)` itself a transform and bind the initial state and sequence later. The resulting Node derives its carry from the contract, so populations, inner adaptation, and outer optimization compose without introducing another execution model.

This is not limited to neural layers. A controller, its estimator, the plant
it controls, and the process that tunes them can inhabit the same Node
tree. NodeJAX models differentiable programs, not one particular category of
model.

## Randomness stays explicit without becoming bookkeeping

Functional randomness often appears to demand a choice between an implicit
global generator and manually splitting and threading keys through every
intermediate function.

NodeJAX separates the dependency from the bookkeeping. A stochastic role says
that it needs entropy and the caller supplies a key at the public boundary.
Composition determines how that entropy reaches the stochastic parts beneath
it. Local code receives a scope-local mutable stream and draws as often as its
own logic needs:

```python
def param(node, rng):
    shape = node.input.shape
    w = jax.random.normal(rng.next(), shape)
    b = jax.random.normal(rng.next(), shape)
    return Struct(w=w, b=b)
```

Randomness is never hidden in runtime data or global state. A stochastic call
requires a key, and a deterministic call rejects one. At the same time, local
code does not count splits or know the topology of the surrounding
composition.

The dependency remains explicit; the splitting does not dominate the program.
This follows the broader rule: automate consequences, not dependencies.

## NodeJAX succeeds when composition requires less glue

NodeJAX aims to make the Node both the natural unit of software
organization and the natural unit of JAX transformation.

A Node carries the distinctions required for that goal. FOOP gives it object
structure without abandoning pure value semantics. The framework succeeds to
the extent that meaningful Nodes combine directly and the plumbing needed
to preserve their meaning disappears from ordinary code.

NodeJAX is under active development. The [cookbook](cookbook.md) teaches the
API, the [handbook](handbook.md) is the public technical reference, the
[design](design.md) describes the implementation architecture, and the
[comparison](comparison.md) considers related framework choices.
