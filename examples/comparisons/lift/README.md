# Reusable lifted stack

Each column builds one reusable stack primitive and runs it against a shared
parity contract. The comparison is about transform authoring, not model
accuracy.

Every implementation must provide:

- independently constructed parameters for every layer;
- a sequential layer axis, where one layer's clean output feeds the next;
- mutable state owned independently by every layer;
- an explicit apply-time RNG stream, independent draws, and reproducibility;
- rejection of both a missing required RNG and an unsolicited RNG;
- auxiliary values stacked over the layer axis without entering the carry;
- support for a layer that emits no auxiliary values.

All three columns satisfy all seven. `nests` is measured and printed rather
than asserted, so a column that cannot compose with itself still reports the
rest.

## What a generic transform has to know, and how it finds out

A transform over an arbitrary layer needs facts about that layer before it can
run one: does it construct parameters, does it carry state, does it draw
entropy, what does its call accept, what does its return value mean.

There are two ways to obtain them. Inspect the layer, or require it to declare
them. Both other columns inspect: they read
`inspect.signature(member.__call__)` to decide whether a member draws.

Inspection reads a layer's implementation surface, and a transform's
implementation surface is not its semantic surface. `LayerStack.__call__` must
declare an optional RNG argument because it may wrap a stochastic layer. That
declaration records what it can forward, not what it needs. Nothing in the
signature separates the two, so a transform whose purpose is to be transparent
about its member's properties is opaque about those properties once it is
itself a member.

## Where that leads

The defect is fixable. Pass an explicit `takes_rng=` to the constructor, set a
class attribute, define a marker protocol. Each of those replaces an inspected
fact with a declared one.

The same then applies to the next fact, and both columns have already been
through it twice. `WithAux` exists because a returned tuple is ambiguous:
inspection cannot separate a `(value, aux)` pair from a layer returning two
things, so aux is declared. The `make_layer` convention exists because
construction order and RNG splitting cannot be discovered, so construction is
declared. Two facts declared, one inspected, and the inspected one is the one
that breaks.

Continue and the set converges: what a layer builds, what it carries, what it
draws, what its call accepts, what its output means. That is the NodeJAX
contract. The claim here is not that the contract is elegant or that Flax and
Equinox should ship one. It is that compositionality forces each fact to become
declared rather than discovered, one failure at a time.

The facts are also not arbitrary. They are what JAX's transforms ask of a
value: what is static, what is differentiated, what is carried, what belongs to
this call.

## Member contexts

Beyond the one member the parity contract asserts, each column offers its
transform four more, varying what the member owns. Every one owns parameters,
since that is what a layer axis ranges over. What changes is whether it carries
state and whether it draws at apply.

The fifth is the framework's OTHER mechanism for mutable state, which all three
have in a different form:

- NodeJAX has one state role, so the second kind is a tag on that role.
  `single_batch_state` marks state that must not gain an axis when the node is
  mapped, which is what running statistics need and what `nn.BatchNorm` uses.
- NNX has Variable subclasses. `nnx.BatchStat` rather than a plain
  `nnx.Variable`, which is what `nnx.StateAxes` keys on when a lifted transform
  gives one collection a different axis from another.
- Equinox has `eqx.nn.StateIndex` and `eqx.nn.State`, threaded through the call
  as a second argument and returned alongside the output. It is the mechanism
  `eqx.nn.BatchNorm` uses, and it is separate from the returned-successor
  convention this column's protocol adopted.

| context | NodeJAX | Flax NNX | Equinox |
| --- | --- | --- | --- |
| parameters only | carried | carried | carried |
| draws at apply | carried | carried | carried |
| carries state | carried | carried | carried |
| both | carried | carried | carried |
| other state kind | carried | carried | rejected |

In the two columns that carry it, the other state kind gains the layer axis and
advances like ordinary state, `(4, 8)` in both.

Equinox rejects it, and the reason is the one that decides the whole column.
`eqx.nn.State` is threaded as a second positional argument, so
`OtherStateLayer.__call__(value, state)` does not fit a protocol whose members
are called as `layer(carry)`. The protocol already had to choose a state
convention, and it chose returned successors, which is the one Equinox does not
use for `BatchNorm`. Either choice rejects the other. This is the same reason
the column takes none of Equinox's stock layers, which do not agree on a call
shape among themselves:

```
Linear     (x, key)                        -> Array
LayerNorm  (x, state, key)                 -> Array or (Array, State)
Dropout    (x, key, inference, ...)        -> Array
BatchNorm  (x, state, key, inference)      -> (Array, State)
GRUCell    (input, hidden, key)            -> hidden
```

## Two structural limits in the Equinox column

The call-shape disagreement above is a convention, and a library could tighten
it. The next two are not. They follow from modules being pytrees with no role
labels, and from state being addressed by an object created at construction.
They lock together: the mechanism that separates state from parameters is the
one that cannot be built per-layer, and the mechanism that can be built
per-layer is the one that cannot separate them.

**`eqx.nn.State` cannot be given one slot per layer.** `eqx.nn.StateIndex` is
addressed by Python object identity, created when the constructor runs, and
`eqx.filter_vmap` traces a constructor once. The column reports both counts:

```
eqx.nn.State slots for 4 layers: 4 built separately, 1 built under filter_vmap
```

The state leaf still comes out `(4, 8)`, so it looks right until it is scanned.
Inside the scan the weights are sliced to `(8,)` as usual, while a state read
returns the whole `(4, 8)`, because the index names a model-global slot and
identity is not something a scan can index per iteration. The carry types then
disagree:

```
scan carry input has type f32[8] but the output carry has type f32[4,8]
```

Building the four layers in a Python loop does give four slots of `(8,)` each,
and it works. That is unrolling, so compile time grows with depth, which is
what a lifted stack exists to avoid.

**Returned successors cannot separate state from parameters.** This is the
convention the column adopted instead, and it puts `weight` and `mean` in one
pytree as ordinary arrays. `eqx.is_inexact_array` selects both, so
`eqx.filter_grad` returns a gradient for both:

| | gradient leaves | reaches the running statistic |
| --- | --- | --- |
| NodeJAX | 1 | no: `param` and `state` are separate trees |
| Flax NNX | 1 | no: `nnx.grad` differentiates `nnx.Param` |
| Equinox | 2 | yes |

An optimizer handed that gradient updates the running statistic as if it were a
parameter. The field name is the only thing that distinguishes them, and a
transform does not know the field names of a layer someone else wrote. `grad`
in each column's printed line reports this.

`nnx.Dropout` carries a caveat of its own. It works only if the `Rngs` passed
at the call contains a stream named `dropout`, and raises `KeyError` at run
time otherwise. `layer_stack` can inspect whether a member takes `rngs`; it
cannot inspect which named streams the member will request. The caller must
know the member's stream names, and nothing declares or checks them.

Inspection is also worthless in this column. Every stock Equinox layer declares
`key=`, whether or not it uses one:

| layer | declares `key` | uses it |
| --- | --- | --- |
| `Linear` | yes | no |
| `LayerNorm` | yes | no |
| `BatchNorm` | yes | no |
| `MLP` | yes | no |
| `GRUCell` | yes | no |
| `Dropout` | yes | yes |

The convention is deliberate and makes layers interchangeable at call sites. It
also means a signature carries no information about entropy, so the method the
column depends on cannot work here at all.

## Self-composition

`stack(stack(...))` composes, with a stochastic inner layer included. Neither
other column composes with itself: the inner transform declares its optional
key argument because it may forward one, and the outer transform's signature
check reads that as a layer that draws.

Flax NNX and Equinox differ in state model. NNX keeps a mutable graph and
propagates `Variable` updates through lifted transforms. Equinox has no mutable
state, so a stateful layer returns a copy of itself and a plain `lax.scan`
stacks the successors. One splits a graph, the other partitions a pytree. Both
fail at the same question for the same reason, so the result follows from the
method rather than from either library.

## Differences that remain

| Concern | NodeJAX | Flax NNX | Equinox |
| --- | --- | --- | --- |
| How the entropy fact is known | declared on the definition | inspected from the call signature | inspected, and every layer declares `key` |
| Member contexts carried | 5 of 5 | 5 of 5 | 4 of 5: not the `State` mechanism |
| RNG stream names | none at the boundary; one key, routed structurally | caller must know them | passed positionally as keys |
| Accepts its own product | yes, stochastic inner included | no | no |
| State | explicit value returned by apply | mutable `Variable` on the graph | a new module returned by the call |
| Parameters against state | separate contract roles | separate `Variable` types | leaves of one pytree, told apart by field name |
| Aux marker | `Aux`, known to every transform | `WithAux`, local to this file | `WithAux`, local to this file |
| Axis policy | part of the transform contract | the adjacent `nnx.vmap` and `nnx.scan` | plain `filter_vmap` and `lax.scan` |
| Initialization from an arriving value | supported by priming | omitted | omitted |

Run the columns:

```sh
python -m examples.comparisons.lift.lift_nodejax
python -m examples.comparisons.lift.lift_nnx
python -m examples.comparisons.lift.lift_equinox
```
