# Working on NodeJAX

This is the repository-wide instruction entry point for coding agents. Read it
before changing code. If `CLAUDE.md` exists, read it completely as well; it
contains additional local prose, naming, and workflow decisions and is equally
binding. Nested `AGENTS.md` files may only add rules for their own subtree.

## Values do not change underneath the caller

If code receives a Node, parameter tree, state tree, definition, or input, it
does not change that value. It computes and returns another value.

```python
# Good: the old state still means what it meant before the call.
next_state = state.replace(count=state.count + 1)
return next_state

# Bad: the caller's state changes behind its back.
state.count += 1
return state
```

This rule is about what a value means to its caller. It is not a claim that no
Python object may ever change internally.

These mutations are fine:

- A constructor fills the fields of the new object it is constructing.
- A function builds a fresh local `dict` or `list`, fills it, and consumes it
  before returning.
- A cache remembers an answer without changing the answer or any other
  caller-visible behavior. `cached_property` is fine.
- An invocation-local helper such as `KeyStream` or an authored wiring scope
  advances while one call is running, then does not escape as model data.

These mutations are not fine:

- Changing an object received from a caller.
- Changing a bound tree, definition, or retained collection as a side effect
  of an operation that promises to return a value.
- Requiring a caller to know that an apparently functional operation mutated
  something elsewhere.

Python allows somebody to reach into a private attribute such as `_def` and
assign to it. That misuse does not make the design mutable and is not a reason
to freeze every Python field. Do not write a test that mutates private storage
and then report the mutation as a framework defect.

Use a simple immutable container where it naturally matches the data:

- Return immutable collections by default. Keep mutable accumulators inside
  the function that builds and consumes them so their mutation cannot appear
  to escape its scope. If a helper genuinely must return mutable working
  storage, make that exceptional contract explicit in its name, return type,
  and docstring.
- Use `Struct` for a fixed set of named fields when attribute access makes the
  code clearer.
- Use `frozendict` when mapping access makes the code clearer.
- Framework-returned object hierarchies use Nodes, ordinary objects, or
  `Struct` records so named parts have normal attribute access.
- Keep fresh local working dictionaries as ordinary `dict` values.

Do not add custom `__setattr__` machinery, post-init wrappers, defensive-copy
layers, or proxy types merely to make private mutation impossible.

## Polymorphism and type probing

In NodeJAX library code and first-party NodeJAX examples, `hasattr` and
`isinstance` are forbidden unless the user explicitly approves the use. Do not
replace them with exact-class dispatch over a hierarchy. Put behavior on the
owning class and dispatch through a common method, property, or explicit
contract instead.

Comparison implementations for other frameworks are exempt from this style
rule. They should use that framework's ordinary idiom.

## Nullability and compatibility

Do not add defensive `None` states, optional branches, compatibility shims, or
dual representations. Values are concrete unless absence is part of the
contract. When a representation changes, update every producer and consumer in
the same change.

## Documentation

Files under `docs/` are intentional public writing. Edit them only when the
user explicitly asks for a documentation change.

## Scope and verification

- Targeted user-requested edits and focused tests may proceed without asking.
- Present a plan and get confirmation before sweeping core-contract refactors,
  deleting files, or destructive Git operations.
- Verify behavior by running the relevant code. Do not turn an assumption into
  a stated result.
- Tests describe an evolving design. Do not preserve a stale behavior solely
  to keep an old assertion green.
