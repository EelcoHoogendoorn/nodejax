# NodeJAX design

This document describes the implementation architecture of NodeJAX. The
[philosophy](philosophy.md) explains why the framework exists. The
[handbook](handbook.md) explains how to use it.

## One definition

A Node `Def` is its complete immutable description. Even the simplest Node
lowers to this common internal contract, while its public view exposes only the
operations relevant to its binding stage. Because every Node has the same
internal form, composition and transforms can operate generically across setup
and application, and their products remain ordinary Nodes.

```text
                                  authoring

       T2 Node authoring       T3 transform authoring       T4 framework authoring
               \                        |                         /
                \                       v                        /
                 +--------------------> Def <-------------------+
                                        |
                       +----------------+----------------+
                       |                |                |
                       v                v                v
              Node / PNode / PSNode  `node` / `self`  Contract
                  T1 public views     T2 views         T3 view
```

## Interaction levels

The T1 through T4 labels in the diagram mark four levels of interaction with
a definition:

| Level | Intended work | Main interface |
| :--- | :--- | :--- |
| T1 | Compose, specialize, bind, transform, and execute existing Nodes. | `nodejax.nn`, `nodejax.control`, stock `transforms`, composition, and the public Node views they produce |
| T2 | Author leaves, composites, wrappers, and factories with ordinary Python functions. | `@node`, `Leaf`, authored `Composite` and `Wrapper`, the reserved `node` argument, and the Composite or Wrapper `self` scope |
| T3 | Author reusable transforms against the complete Node contract. | `@transform`, `Contract`, `Wrapper.roles`, `Composite.roles`, and `nodejax.transforms.transform` helpers |
| T4 | Framework-internal development. | `Def` and its canonical call, capture, and layout records |

These levels are design targets, not an access-control system. Application
code can use T4 when it needs to implement framework-level behavior. It then
owns the same invariants and maintenance cost as framework code.

The intended lines still matter. Node authors should not need canonical
call records. Transform authors should not need authored Python signatures or
manual definition rebuilding.

## Node authoring

`@node` records an ordinary Python factory as static construction. The factory
builds one of three Node forms.

### `Leaf`

A leaf has no members. Its authored functions may
construct a local parameter pytree, construct or prime a local state pytree,
and implement arbitrary JAX computation over params, state, and input.

Reserved arguments such as `param`, `state`, `node`, and `rng` express the roles a leaf function uses. Leaf `apply` returns auxiliary values explicitly and does not receive `self`. NodeJAX lowers the authored signature once. Composition and transforms never inspect it again.

### `Composite`

A composite declares a fixed tree of named members. That member tree defines
the common layout followed by its parameter, state, auxiliary-output, and
static trees. The same member path identifies the corresponding Node in
each tree where it contributes a value.

A composite supports two authoring forms. In the routed form, the body defines
dataflow by calling `self.member(...)`, and the invocation scope supplies each
member's params and state. In the raw form,
`(param, state, input) -> (next_state, output)`, the author operates on the
complete member-shaped trees directly. Both forms may use ordinary JAX
operations, but neither changes the declared member structure during
execution.

The composite owns no additional unregistered trainable or evolving value. If
it needs one, that value belongs in a leaf registered as another member.

This fixed structure lets composition, transforms, selection, inspection,
state traversal, and lifetime handling follow one definition tree.

### `Wrapper`

A wrapper is the one-member structural form. It changes the contract around a
Node without pretending that the wrapper owns a separate parameter or
state subtree. A transparent wrapper can retain the member's value layout
while changing its behavior, metadata, or transform semantics.

Earlier NodeJAX versions allowed a composite Node to own local params and state
beside member-owned values. That added no expressivity because local values
could be represented by a leaf member. It did force every setup, execution,
and transform path to merge two ownership trees. The strict leaf/composite
split removes that framework machinery.

## Transform authoring

A transform consumes a complete Node and returns another Node. The
result must expose the same general contract so it can enter composition or a
later transform without an adapter.

`Contract` is the T3 view of a definition. It presents the same role facts and
operations whether the operand began as a leaf, composite, wrapper, trainer,
or another transform product. This allows a transform to describe its own
semantics without recovering how the operand was authored.

The definition normalizes every Node around parameter construction,
state initialization or priming, and state-transition application. It records
whether those roles exist, whether they require RNG or real input, how runtime
arguments are formed, and what input evidence has been resolved. Public values
may be sparse, while `Contract` supplies the fixed definition-aligned values
used inside transformed calls.

That standardization removes the usual cross-product of special cases. A
transform author decides which axes are mapped, which values are broadcast,
what becomes carry, how RNG is split, and which bindings remain valid. The
transform does not inspect function signatures, infer roles from values, or
dispatch over concrete Node classes.

[`ensemble`](../nodejax/transforms/axes/ensemble.py) is the direct example. It maps
params and state over a new axis, broadcasts runtime input, stacks outputs, and
splits role RNG when required. The common vmap helpers handle absent roles and
call formation. `Wrapper.roles` turns those role functions into another
complete definition.

`@transform(preserves=...)` declares whether existing parameter or state
bindings may be reattached to the result. The transform builder itself always
works from the unbound definition, so binding-stage behavior is handled once
instead of being repeated in every transform.

## Framework-level authoring

Some operations change definition structure rather than only placing new
roles around a member contract. They belong at T4.

The update arithmetic in `train_step` is simple. Its construction is deeper
because it changes lifetimes: the model's parameter constructor becomes the
trainer's parameter constructor, while active weights, model state, and
optimizer state become trainer state. Preserving that relationship for
unbound, bound, input-dependent, stochastic, and transformed models requires
the trainer to retain the model's exact parameter construction form.

The current [`tie`](../nodejax/transforms/structure/tie.py) also works at T4. It removes
alias fields from the stored public parameter tree, expands the source value
into the fixed member-shaped tree used during execution, and records sparse
ownership in the definition layout. Its path-based rewrite is not a general
parameter-identity system and is not fully compositional.

The open design question is whether parameter identity should survive
arbitrary transforms as definition data, or whether one explicit lowering
operation should consume sharing metadata before training. Either approach
must make identity explicit rather than infer it from paths, array aliasing,
or payload shape.

## Definition and bound values

An incomplete static factory is a `Generic`, not a fourth Node form or a
partial `Def`. `specialize` supplies missing statics and replays the recorded
factory tree into a complete definition.

`Node`, `PNode`, and `PSNode` are sibling views over one `Def`:

- `Node` exposes an unbound definition.
- `PNode` adds its parameter pytree.
- `PSNode` adds its parameter and live state pytrees.

`PNode` and `PSNode` are JAX pytrees whose children are the bound values; their
shared `Def` is static auxiliary data. Calling a `PSNode` returns a successor
view rather than changing the original.

A state-bound `train_step` product follows the same rule. Its successor state
contains a new active model-parameter tree and optimizer state. The trainer's
own params remain the differentiable initialization.

Definition construction happens outside traced JAX execution while params and
state remain ordinary transformed values. Binding follows a fixed order:

```text
statics -> member tree -> input evidence -> params -> state
```

Re-entering an earlier stage discards later bindings unless an operation
explicitly preserves them. A rebuilt definition does not retain weights or
state merely because an earlier view contained them.

When a bound Node is inserted as a member, the parent records its
finished param or state as an explicit capture. The member tree still contains
only definitions. The capture is a construction default belonging to that act
of composition, not a mutation of the child definition.

## Design invariants

- One `Def` owns the facts that define a complete Node.
- A leaf owns raw values; a composite owns a fixed named member structure.
- Parameter, state, auxiliary-output, and static trees follow the declared
  Node tree.
- Public and authoring interfaces are views or invocation scopes over the
  definition, not alternative definition models.
- Role presence, priming, input evidence, RNG requirements, layout, and scan-boundary state actions are explicit definition data.
- Caller-owned values do not change. Local construction and invocation scopes
  may use mutation that cannot escape as model data.
- A completed composition or transform returns an ordinary public Node view;
  deferred static construction remains a `Generic`.

## Where the implementation lives

- [`definition.py`](../nodejax/core/definition.py) defines `Def`, construction
  records, captures, and layout.
- [`contract.py`](../nodejax/core/contract.py) defines the canonical roles and the
  transform-facing `Contract`.
- [`lifting.py`](../nodejax/core/lifting.py) and
  [`authoring.py`](../nodejax/core/authoring.py) lower Node authoring into
  definitions.
- [`composite.py`](../nodejax/core/composite.py),
  [`compose.py`](../nodejax/core/compose.py), and
  [`wrapper.py`](../nodejax/core/wrapper.py) build structured definitions.
- [`node.py`](../nodejax/core/node.py), [`pnode.py`](../nodejax/core/pnode.py), and
  [`psnode.py`](../nodejax/core/psnode.py) provide the public views.
- [`transform.py`](../nodejax/transforms/transform.py) defines the supported transform
  authoring interface.
