# The transparent wrapper

A residual wrapper is the smallest useful transform: `x + f(x)`. It owns no parameters, state, RNG, or construction policy. Everything except the addition belongs to its member, which makes residual a clean test of whether a framework can turn a wrapper into a set-and-forget library operation.

NodeJAX's version is the whole transform:

```python
@node
def residual(body):
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name=f'res({body.name})')
```

The apply body states only the residual's semantics. The common Node contract carries parameter construction, state initialization and priming, RNG, aux, member methods, and deferred construction through the wrapper.

## The common unary case

All three columns can write a correct residual for a member whose signal is its first argument and whose output is one addable value. Extra call arguments can be forwarded mechanically, so RNG does not need signature inspection.

| | deterministic nesting | stochastic nesting | state representation | parameter gradient reaches state |
| --- | --- | --- | --- | --- |
| NodeJAX | works | works | canonical separate state | no |
| Flax NNX | works | works | graph `Variable` | no |
| Equinox | works | works | `eqx.nn.State` | no |

The old NNX and Equinox failures in this comparison came from locally inspecting whether the inner call accepted an RNG keyword. That confused the ability to forward RNG with a requirement to consume it. Both implementations now pass arguments through and both nest correctly.

## What set-and-forget requires

The unary call is not the hard part. A reusable wrapper needs the same answer for every value it may receive as a member:

- how the member is constructed, including fresh independent copies;
- how parameters are initialized from configuration or input shape;
- how state is initialized or primed from a real input;
- which arguments form the signal and where the output lives;
- which state protocol is in use and what its lifetime is;
- how RNG, aux, and member methods cross the wrapper;
- how later transforms see all of those facts.

NodeJAX records these facts on one Def and every transform returns another ordinary Node. The NNX and Equinox implementations can cover strong local protocols, but neither `nnx.Module` nor `eqx.Module` declares this complete surface.

## The NNX boundary

NNX is strong inside its unary graph protocol. Parameters and graph-resident Variables remain on the member, `nnx.grad` selects `Param`, graph-aware transforms propagate mutation, and `sow` plus `capture` keeps aux outside the returned value. Deterministic and stochastic wrappers both nest.

NNX also uses explicit functional carry. A stock RNN cell publishes `initialize_carry`, `num_feature_axes`, and `(carry, input) -> (carry, output)`. The unary residual cannot identify the signal argument or the output slot, and wrapping the cell removes the interface expected by `nnx.RNN`. A cell-specific residual can preserve both graph Variables and carry, but it is a second adapter for a second protocol. Cache-bearing modules add another lifecycle method through `init_cache`.

Construction creates values before the residual sees the module. Wrapping one configured module is ordinary NNX, but a later stack or ensemble that needs independent initializations must receive a factory which constructs the body and residual together. Transform order therefore changes the kind of value being composed. Reusing the same configured module also reuses its graph identity, so state and parameters are shared unless the caller deliberately constructs or clones another one.

Mutation adds an execution boundary. If the member updates a Variable and then returns an incompatible output, the residual addition fails after the update has happened. Ordinary JAX transforms may accept an NNX graph but do not preserve NNX mutation and filtering semantics; the enclosing operation must use the NNX-aware transform.

The executable reports the two central failures directly:

```text
functional-state=NO cell-interface=NO
```

## The Equinox boundary

Equinox has a stronger native answer than the old comparison credited. A residual can subclass `eqx.nn.StatefulLayer`, delegate `is_stateful()`, and forward the external `eqx.nn.State`. `make_with_state` then initializes state through the wrapper, `eqx.nn.Sequential` recognizes the wrapped layer, nested residuals work, and the running statistic remains outside the parameter gradient.

That convention still does not cover every `eqx.Module`. The lifted-stack comparison uses another natural functional convention, `(successor, output)`, because native `StateIndex` values do not provide independent per-layer slots when constructed under `filter_vmap`. A returned-successor body is not a `StatefulLayer`, so the residual cannot distinguish its pair from an ordinary structured output. The residual and lifted stack can each support state, but their stateful members are not interchangeable.

Other Modules may use positional recurrent state, publish special methods or attributes, or return structured outputs with their own meaning. Equinox does not identify the residual signal, output projection, or aux at the Module level. This column uses a local `WithAux` marker. Fresh stacking and ensembling also require a constructor factory because a configured Module does not retain the call that built it.

State updates are functional, but `State.set` consumes the old State. A member can produce its successor before the residual discovers that its output cannot be added. The failed wrapper call then leaves the old State invalid rather than returning a successor.

The executable reports both sides of the state split:

```text
native-state=yes successor-state=NO
```

## The NodeJAX boundary

The stock NodeJAX residual preserves input-primed state and member methods, and its result keeps the construction information needed for later specialization. It consumes the same state contract whether the body was authored as a leaf, a composition, or another transform.

This particular six-line transform is intentionally unary. Its apply role declares only `input`, so a body's additional runtime arguments are not republished. NodeJAX can represent and route named call fields, but a transform must declare which one is the residual signal and what the others mean. The executable reports that limit explicitly:

```text
priming=yes methods=yes extra-arguments=NO
```

## Aux

NodeJAX carries `Aux` through the common call contract. NNX bodies `sow` intermediates and an enclosing `capture` transform collects them, so capture is part of the execution context. Equinox has no common aux return, so this column defines `WithAux` and teaches its residual to split and rebuild that marker. All three preserve aux in this example, but only one convention applies to every NodeJAX transform.

Run the columns:

```sh
python -m examples.comparisons.residual.residual_nodejax
python -m examples.comparisons.residual.residual_nnx
python -m examples.comparisons.residual.residual_equinox
```
