# NodeJAX Architecture and Transform Design

Implementation architecture of `Def` lowering, canonical `Contract` roles, T3 transform construction, and T4 structural seams. For public usage and T1/T2 authoring APIs, see the [handbook](handbook.md). For boundary and composition design across domains, see the [authoring guide](authoring.md).

## 1. One definition (`Def`) and four interaction levels

Every completed Node lowers to a single immutable `Def` value (`nodejax/core/definition.py`). Public views, authoring scopes, and transform builders all project that same `Def` rather than maintaining separate representations:

```text
       T2 Node authoring       T3 transform authoring       T4 framework authoring
               \                        |                         /
                \                       v                        /
                 +--------------------> Def <-------------------+
                                        |
                       +----------------+----------------+
                       |                |                |
                       v                v                v
              Node / PNode / PSNode  `node` / `self`  Contract
                  T1 public views     T2 scopes        T3 view
```

| Level | Scope of work | Primary interfaces |
| :--- | :--- | :--- |
| T1 | Compose, specialize, bind, transform, and execute existing Nodes | `Node`, `PNode`, `PSNode`, `Generic`, combinators, stock `transforms`, `nn`, `control` |
| T2 | Author leaves, composites, and single-member wrappers with Python functions | `@node`, `Leaf`, `derive`, `Composite`, `Wrapper`, `node`, `self`, `KeyStream` |
| T3 | Author reusable transforms over arbitrary Nodes without inspecting Python signatures | `@transform`, `Contract`, `Wrapper(...).roles(...)`, `Composite(...).roles(...)`, `MaybeKeyStream`, `nodejax.transforms.transform` helpers |
| T4 | Modify definition records, parameter layouts, or lifetime rewrites directly | `Def`, `ContractCalls`, `Layout`, `Captures`, `Construction` |

## 2. Lowering T2 authoring into `Def`

`@node` records a Python factory and its static arguments inside `Construction` so `specialize(...)` can replay construction with updated statics. Calling a complete factory produces one of three lowered structural forms on `Def`:

### `Leaf`

`Leaf` compiles plain Python `param`, `init`, and `apply` functions once (`nodejax/core/lifting.py`). Signature inspection happens exclusively at `Leaf` lowering: reserved names (`param`, `state`, `input`, `node`, `rng`) bind to canonical contract slots, any remaining parameter names become fields of a `CallForm` input specification, and downstream combinators and transforms never inspect the authored Python signature again.

### `Composite`

`Composite(**members)` registers a fixed `Struct` of child `Def` values (`Def.members`). Parameter, state, `Aux`, and static trees all mirror those member paths (`param.<member>`, `state.<member>`, `aux.<member>`). A composite owns no unregistered trainable weights or evolving state outside its declared members; any additional parameter or state value belongs in a registered `Leaf` member.

When authored in routed form (`def apply(self, ...)` or `def init(self, ...)`), NodeJAX runs an abstract discovery trace over `apply` (`nodejax/core/wiring.py`) to propagate member input shapes (`apply_input_spec`), determine RNG consumption, and compile a pure functional step over member-aligned `Struct` trees.

### `Wrapper`

`Wrapper(name=member)` registers a single child `Def` while setting `Layout(transparent_member=name)` so `param`, `state`, and `Aux` keep the inner member's layout without adding an extra `Struct` key. Both authored T2 behavioral wrappers (`Wrapper(policy=policy)(apply)`) and T3 transforms (`Wrapper(member=member).roles(...)`) share this single-member representation.

### `Captures` and binding order

`Def.members` stores only unbound `Def` values. When a caller passes an already-bound `PNode` or `PSNode` as a member during composition, `_promote_members` extracts its `Def` into `Def.members` and stores its bound array PyTrees in `Def.captures` (`Captures(param=..., state=...)`). Captured values act as per-composition defaults during parent parameterization or initialization; rebuilding the member tree clears captures so rewritten subtrees construct from their unbound stages.

Binding stages advance strictly from left to right:

```text
Generic -> Def (member tree) -> input_spec -> PNode (param) -> PSNode (param, state)
```

Re-entering an earlier stage (such as calling `.specialize(...)` on a bound `PNode`) replays the factory and discards downstream parameter and state bindings.

## 3. Canonical `Contract` and T3 transform authoring

`node.contract` returns `Contract(def)`, the uniform T3 projection of any `Def` (`nodejax/core/contract.py`). Regardless of whether a Node began as a `Leaf`, `Composite`, `Wrapper`, or another transform product, `Contract` exposes four normalized entry points:

```text
contract.param(param_input, rng)               -> param
contract.init(param, state_input, rng)         -> state
contract.prime(param, state_input, input, rng) -> state
contract.apply(param, state, input, rng)       -> (next_state, output)
```

At the `Contract` boundary:

- `param` and `state` are `()` when `!contract.parametric` or `!contract.cyclic`.
- `rng` is always a `MaybeKeyStream`: keyed when `contract.{param,init,apply}_takes_rng` is `True` and empty otherwise, supporting `.child(takes_rng)` for member routing and `.axis(takes_rng, count)` for mapped/scanned axes.
- `init` constructs non-priming state (`!contract.init_requires_input`), while `prime` resolves shape-dependent definitions against a concrete or abstract `input` and passes `input` into input-primed initializers.

`nodejax/transforms/transform.py` provides `@transform` alongside `vmap_param`, `vmap_init`, `vmap_prime`, `vmap_apply`, `scan_inputs`, `scan_steps`, and `bind`. `@transform(preserves=..., internalizes=...)` handles `Generic` deferral and bound-view reattachment in one place:

- `preserves=()` (default) requires an unbound `Node` when parameter and state layouts change (such as `ensemble` or `stack`).
- `preserves='param'` accepts `Node` or `PNode` and reattaches `param` when parameter layout is unchanged (such as `batch` or `scan`).
- `preserves='param,state'` accepts `Node`, `PNode`, or `PSNode` and reattaches both `param` and `state` (such as `remat` or `detach`).
- `internalizes='state'` accepts a `PSNode` and hands it directly to the builder so its bound state serves as the run's initial carry while reattaching only `param` (used by `scanned`, `carried`, and `repeated`).

`ensemble` (`nodejax/transforms/axes/ensemble.py`) illustrates a complete T3 transform built with `@transform` and `Wrapper(...).roles(...)`:

```python
from nodejax import Aux, Leaf, Node, PNode, Wrapper, node
from nodejax.transforms.transform import (
    transform, vmap_apply, vmap_init, vmap_param, vmap_prime,
)

@transform
def ensemble(member: Node, n: int, axis: str = 'ensemble') -> Node:
    if not (member.parametric or member.cyclic):
        raise TypeError(f'ensemble of {member!r} requires params or state')

    def param_fn(contract, param_input, rng):
        return vmap_param(
            contract.members.member, contract, rng, param_input, count=n,
        )

    def init_fn(contract, param, state_input, rng):
        return vmap_init(
            contract.members.member, contract, rng, param, state_input,
            count=n, param_axis=0,
        )

    def prime_fn(contract, param, state_input, input, rng):
        return vmap_prime(
            contract.members.member, param, state_input, input, rng,
            count=n, param_axis=0, input_axis=None, state_axis=0,
        )

    def apply_fn(contract, param, state, input, rng):
        return vmap_apply(
            contract.members.member, param, state, input, rng,
            param_axis=0, state_axis=0, input_axis=None,
            axis_name=axis, count=n,
        )

    return Wrapper(member=member).roles(
        destructurable=False,
        name=f'ensemble({member.name})',
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
    )
```

## 4. T4 structural operations

Operations that alter parameter ownership or shift values across contract lifetimes operate directly on `Def` and `Layout` at T4:

- `train_step` (`nodejax/transforms/learning/train_step.py`) rebinds lifetimes: the inner model's parameter constructor becomes the trainer's `param` constructor (holding differentiable initial weights), while active weights, inner model `state`, and optimizer `state` combine into the trainer's `state`.
- `tie` (`nodejax/transforms/structure/tie.py`) removes aliased member slots from public `Layout.param_members` so shared tensors appear once in `param`, then expands the source tensor into aliased slots before member execution so gradients sum via standard reverse-mode AD.
- `externalize` (`nodejax/transforms/wiring/externalize.py`) removes selected parameter subtrees from `param` (recording `Layout.externalized_param_paths`) and exposes them as explicit call-time inputs while keeping the member `Def` in `Def.members` for state, RNG, and shape propagation.

## 5. Module map

| Module | Responsibility |
| :--- | :--- |
| `nodejax/core/definition.py` | `Def`, `Layout`, `Captures`, and `Construction` |
| `nodejax/core/contract.py` | `Contract`, `ContractCalls`, `ParamCall`, `InitCall`, `ApplyCall`, and `CallForm` |
| `nodejax/core/lifting.py`, `nodejax/core/authoring.py` | Signature compilation, `Leaf`, and `derive` |
| `nodejax/core/wiring.py`, `nodejax/core/composite.py`, `nodejax/core/compose.py`, `nodejax/core/wrapper.py` | `self` discovery traces, `Composite`, `serial` / `parallel` / `sum_junction`, and `Wrapper` |
| `nodejax/core/ambient.py`, `nodejax/core/generic.py` | `@node`, `ambient`, `Generic`, and `specialize` |
| `nodejax/core/node.py`, `nodejax/core/pnode.py`, `nodejax/core/psnode.py` | Public `BaseNode`, `Node`, `PNode`, and `PSNode` PyTree views |
| `nodejax/core/rng.py`, `nodejax/core/spec.py`, `nodejax/core/introspection.py` | `KeyStream` / `MaybeKeyStream`, shape specs, and `describe` / `tree_view` / `summary` |
| `nodejax/transforms/transform.py` | `@transform` and T3 `vmap_*` / `scan_*` / `bind` helpers |
