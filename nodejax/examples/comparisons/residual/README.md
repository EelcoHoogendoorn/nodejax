# The transparent wrapper

A residual wrapper is the smallest transform: `x + f(x)`. It owns no axis, no
parameters, no state and no entropy. Everything except the addition is
forwarding.

The lifted-stack family next door is the same probe with axis machinery added.
This one has none, so a column that fails here fails at forwarding and nothing
else.

NodeJAX's version is the whole transform:

```python
@node
def residual(body):
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name=f'res({body.name})')
```

No line mentions params, state, entropy or aux. `self.body(input)` is a member
call, so the machinery forwards what the member declares.

## What the other columns must supply

Before either can write `x + inner(x)` it needs two facts about the member it
was handed.

Whether the member takes a key. The call spelling differs, so the wrapper reads
`inspect.signature(inner.__call__)` and branches.

Whether the member emitted aux. Adding a value to a `(value, aux)` pair is a
type error, so the wrapper recognises aux, splits it, adds to the clean half
and re-emits. Recognising it requires a marker type, declared locally.

Equinox needs a third. With no mutable variable there is nowhere to put carried
state, so every member returns `(successor, output)` whether it carries
anything or not, and the wrapper rebuilds itself around the successor.

## Self-composition

| | deterministic member | stochastic member |
| --- | --- | --- |
| NodeJAX | works | works |
| Flax NNX | fails | works |
| Equinox | fails | works |

`Residual.__call__` declares an optional key because it may forward one. The
declaration records what it can pass on, not what it needs. The next wrapper up
inspects the signature, sees the key, and treats the member as one that draws,
then demands a key a deterministic body never consumes.

The wrapper therefore composes around a member that draws and fails around one
that does not. `nests` in the shared harness requires both cases for that
reason.

## Scope of the result

Flax NNX and Equinox differ in state model. NNX keeps a mutable graph and
propagates `Variable` updates. Equinox has no mutable state and returns
successors through a pytree. Both fail at the same question for the same
reason: the entropy fact is recovered from a Python signature, and a wrapper's
signature describes what it can forward.

The result follows from recovering a member's facts by inspection, so it is not
specific to either library.

## What the wrapper cannot repair

The wrapper owns no state, so how a framework tells state from parameters is
not its doing. It still decides what a caller gets back, which shows up in the
gradient.

| | gradient leaves | reaches the running statistic |
| --- | --- | --- |
| NodeJAX | 1 | no: `param` and `state` are separate trees |
| Flax NNX | 1 | no: `nnx.grad` differentiates `nnx.Param` |
| Equinox | 2 | yes |

Equinox has no mutable variable, so a returned successor is where a member's
state has to live, and a successor puts `weight` and `mean` in one pytree as
ordinary arrays. `eqx.is_inexact_array` selects both, so `eqx.filter_grad`
returns a gradient for both, and an optimizer handed it updates the running
statistic as if it were a parameter. The field name is the only thing that
distinguishes them.

The lifted-stack family reports a second Equinox limit that does not arise
here. `eqx.nn.State`, the mechanism that would separate the two, is addressed
by an object created when the constructor runs, so it cannot be given one slot
per layer under `eqx.filter_vmap`. A residual wrapper constructs nothing over
an axis, so it never meets that. It inherits the consequence anyway, because
the successor convention is what a transform is left with once `State` is
unavailable.

## Aux markers

Each column here defines a `WithAux` marker. So does each column of the
lifted-stack family. Four files, four aux conventions, none agreeing: a body
written for the residual wrapper is unreadable to the stack.

Each file invents the minimum convention it needs, correctly. A
framework-level `Aux` is what makes those conventions agree, and no single file
is in a position to declare one.

Run the columns:

```sh
python -m nodejax.examples.comparisons.residual.residual_nodejax
python -m nodejax.examples.comparisons.residual.residual_nnx
python -m nodejax.examples.comparisons.residual.residual_equinox
```
