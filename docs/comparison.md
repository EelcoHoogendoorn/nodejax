# Framework comparison

[`nodejax/examples/comparisons/`](../nodejax/examples/comparisons/) compares
NodeJAX with Flax NNX, Equinox, Flax Linen, Haiku, PyTorch, and Keras.

## Complex composition

The [composition stress test](../nodejax/examples/comparisons/tower/) probes
whether transformed components remain composable when transforms become the
architecture. It combines a residual recurrent depth stack, an ensemble, time
recurrence, task-local adaptation, second-order MAML, and outer Adam training.

The complete NodeJAX compute graph is five assembly lines:

```python
rnn = stack(residual(RNN(HIDDEN)), n=LAYERS)
member = Up(HIDDEN) >> Norm(HIDDEN, MOMENTUM) >> rnn >> Readout(HIDDEN)
committee = ensemble(scanned(member), n=MEMBERS) >> reduce(jnp.mean)
adapt = finetune(train_step(committee, mse, optax.sgd(INNER_LR)))
return trained(train_step(batch(adapt), mse, optax.adam(OUTER_LR)))
```

Every line consumes a Node and returns a Node. Parameter construction, state,
RNG, axes, adaptation, and training remain in one composition.

Direct NNX takes 183 source lines. Making its model graph compositional within
a local recurrent-step protocol takes 344 and adds residual, serial, stack,
scan, ensemble, and reduction abstractions. That closure ends at the model:
MAML still uses another interface. The local contract is compositional within
that model vocabulary, not across the complete transformed program.

Every rival's native component covers only part of the program. NNX ends before
`maml_fit`; Equinox before adaptation and training; PyTorch and Keras leave
`Module` or `Layer` for `functional_call` or `stateless_call`. State falls into
unrelated containers according to which API owns the loop. The results are not
competitive with NodeJAX on readability.

Keras does not belong beside PyTorch on compute, and the tower column measures
the difference. On the JAX backend `keras.layers.RNN` lowers to one `lax.scan`
and `keras.ops.vectorized_map` is `jax.vmap`, so the time and member axes stay
vectorized: the traced meta step holds at 427 equations for 2 through 32
members and compiles in 0.4 s, against PyTorch's 266 s for the same unrolled
tape. Gradients are JAX's because Keras ships no gradient API, which is its own
documented advice rather than an escape.

What Keras has no representation for is the member axis. Constructing one
cannot be mapped: `keras.initializers` reject a traced seed even though
`keras.random.normal` accepts one, and `add_weight` under a vmap returns
identical rows instead of failing, so a committee built that way trains as
three copies of one member. The column builds three towers separately and
stacks their weights, leaving `trainable_variables` a flat list in which
nothing records which value is a parameter, which is a running statistic, or
what a leading axis counts.

The [test-time-training comparison](../nodejax/examples/comparisons/ttt/) makes
the split exact. NodeJAX composes inner `train_step`, `next_step_ttt`,
`scanned`, `batch`, outer `train_step`, and `trained`; every result is a Node.
NNX adds local `TTT`, `Batched`, `TrainStep`, axis, and carry policies around a
cloned Module. Equinox carries its Module through raw JAX. Haiku and PyTorch
move fast weights into dictionaries; PyTorch then uses `functional_call` and a
handwritten update because `torch.optim` cannot update those values, which is
pseudo-JAX inside PyTorch.

Fast weights and hidden state advance together, yet rivals route one through
native storage and the other through explicit carry. NodeJAX publishes both in
one component state contract consumed unchanged by every enclosing transform.

## Chunked recurrence: control over state lifetime

The [`chunk`](../nodejax/examples/comparisons/chunk/) comparison nests samples,
chunks, and recordings to test whether state lifetime belongs to its component
or leaks into control flow. Hidden state, normalization, and optimizer state
each live across a different boundary.

Rivals split that state among NNX Variables, scan `Carry`, optimizers, PyTorch
buffers, detached tensors, or explicit tuples of params, optimizer state,
model state, and hidden state. The division follows framework boundaries, not
the values' lifetimes.

Changing normalization to reset per recording touches these independent
executable locations:

| Framework | Locations | Owner of the rule |
| :--- | ---: | :--- |
| NodeJAX | 1 | The normalization Node |
| Flax NNX | 2 | Inference and training scans |
| Flax Linen | 4 | Collection policy and state construction |
| Equinox | 6 | Initialization, returned state, and outer carries |
| Haiku | 1 | A rollout that names nested state |
| PyTorch | 2 | Loops that manipulate Module buffers |

NodeJAX makes the policy legible where it belongs:
`state_reinit(RNN(), 'recording')`. An enclosing scan announces the boundary
without inspecting state layout. Elsewhere lifetime is encoded in scans, carry
tuples, collection paths, buffer mutation, or reset lines outside the model.

Haiku's count of one is not equivalent: its rule belongs to one rollout; the
NodeJAX rule belongs to the state and survives other rollouts and trainers.
The competing code must be audited, not read: omitting a reset still produces
a valid program with wrong semantics. Only in NodeJAX is control over state
lifetime not a leaky abstraction.

## Reusable component transforms

The [`residual`](../nodejax/examples/comparisons/residual/) comparison asks for
exact compositionality in the smallest case: one wrapper that accepts every
tested member, publishes its exact contract, and accepts its own output.

The arithmetic is only `input + body(input)`. The complete NodeJAX transform is
six lines and says exactly that; param, state, RNG, input, and aux behavior
propagate through `self.body(input)`.

NNX and Equinox wrap this addition in protocol machinery. NNX's native `sow`
and `capture` handle aux, but its wrapper inspects signatures and branches on
RNG forwarding. Equinox also defines state and aux return conventions and
reconstructs successor modules. Both mistake a nested deterministic residual
for a stochastic component. A transform that rejects its own output is not
compositional.

The [stacked-component comparison](../nodejax/examples/comparisons/lift/) adds
independent construction, sequential state, RNG splitting, optional aux, and
strict RNG errors. NNX keeps its Module but locally defines axis, RNG, and
Variable-class policy. Equinox partitions its Module into arrays and statics,
reconstructs each layer inside `lax.scan`, then partitions and rebuilds every
successor. Params and running state occupy one tree, so `filter_grad`
differentiates both. [NodeJAX `stack`](../nodejax/transforms/stack.py) reads all
of these roles from the contract.

The result is not close. NodeJAX handles all five contexts, nests, and
differentiates only params. NNX handles five but does not nest; Equinox handles
four, does not nest, and differentiates running state. Both bury the stack
under protocol machinery yet produce incomplete transforms.

## Generic construction

The [`generics`](../nodejax/examples/comparisons/generics/) comparison leaves
width, depth, ensemble size, and nested temperature unresolved. NodeJAX keeps
that unfinished architecture composable and specializes it by tree path.

Keras propagates input fan-in through its Functional graph and lazy
`build(input_shape)`. It therefore matches NodeJAX on input-derived parameter
shape. Width, depth, ensemble size, and temperature must still be supplied to
a Python builder before a Keras Model exists.

NNX, Equinox, and PyTorch forward both input fan-in and architectural statics
through their constructor chains. The checked forwarding counts are NodeJAX 0,
Keras 4, and NNX, Equinox, and PyTorch 5. NodeJAX separates input-derived
construction from arbitrary open statics and composes both.

NodeJAX wins this comparison. It is the only checked implementation where the
unfinished architecture is itself a composable value. Keras, NNX, Equinox,
and PyTorch compose only after a Python builder or constructor has fixed the
architectural statics.

## IMU: setup and input propagation

The [`imu`](../nodejax/examples/comparisons/imu/) pipeline combines derivatives
requiring priming values, additive noise, drifting random bias, and
quantization to test whether composition also propagates setup.

NodeJAX derives the pipeline's priming, state, input, and RNG behavior from the
leaf Nodes under ordinary serial composition. NNX construction routes priming
values and named RNG streams to the members. Equinox defines and threads the
complete state container through the pipeline.

Input shape, priming values, and entropy must reach every affected component.
Only NodeJAX's forward pipeline also composes that setup. NNX wires priming and
RNG in `IMU.__init__`; Equinox repeats the topology in `IMUState`, `init`, and
`step`.

## Parameter sharing

The [`tie`](../nodejax/examples/comparisons/tie/) comparison isolates how each
framework represents one shared parameter. NNX and PyTorch reuse one object;
Haiku reuses a parameter path; Equinox uses `eqx.nn.Shared`; NodeJAX stores one
parameter value and explicitly routes it to both uses.

NNX, PyTorch, Haiku, `eqx.nn.Shared`, and NodeJAX produce zero drift. The naive
Equinox object copy drifts. The NodeJAX parameter tree contains one table, and
the Node definition routes it to both uses. There is no capability advantage
here: NNX and PyTorch use object identity, while NodeJAX uses a declarative
route.

## Findings

1. NodeJAX wins the complex-composition comparison on readability. Its entire
   transformed architecture is the five-line definition; the alternatives
   distribute it across carries, axis declarations, variable mappings, and
   training procedures.
2. NodeJAX wins state-lifetime composition. Lifetime policy stays attached to
   the stateful Node while enclosing scans and trainers change.
3. NodeJAX wins reusable transform authoring. Its checked transforms preserve
   one contract and accept their own products; the checked NNX and Equinox
   wrappers require local protocols and fail exact closure.
4. NodeJAX wins generic construction and composed setup. Unresolved statics,
   priming, parameter construction, and RNG routing remain in program
   composition instead of constructor plumbing.
5. RNG is as effortless as in PyTorch without giving up JAX's referential
   transparency. A caller supplies one key exactly when the graph consumes
   entropy; transforms derive splitting and forwarding from the contract.
6. Parameter sharing is a draw on capability. The frameworks differ only in
   whether sharing is represented by identity, path, or a declarative route.
