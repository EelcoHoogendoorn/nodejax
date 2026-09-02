# Framework comparison

[`examples/comparisons/`](../examples/comparisons/) compares
NodeJAX with Flax NNX, Equinox, Flax Linen, Haiku, PyTorch, and Keras.

## Complex composition

The [composition stress test](../examples/comparisons/tower/) probes
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

The [test-time-training comparison](../examples/comparisons/ttt/) makes
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

The [`chunk`](../examples/comparisons/chunk/) comparison nests samples,
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
The competing code must be audited, not read: omitting a reset still produces a valid program with wrong semantics. In NodeJAX this scan-aligned lifetime policy stays attached to the stateful Node instead of leaking into enclosing control flow.

## Recurrent PPO: where the abstraction stops

[`ppo.py`](../examples/rl/ppo.py) implements recurrent PPO for fixed-horizon continuing control; it does not model data-dependent termination. [`distributions.py`](../examples/rl/distributions.py) supplies the learned Gaussian policy Node, and [`ppo_pendulum.py`](../examples/rl/ppo_pendulum.py) supplies the domain policy, value, data, and evaluation Nodes. [`test_ppo_pendulum.py`](../examples/rl/test_ppo_pendulum.py) assembles them over a weakly actuated pendulum with either a feed-forward or a GRU policy. The reference point is brax's [`ppo/train.py`](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py). This is a comparison of authoring boundaries, not throughput or feature count.

Brax's network abstraction ends at feed-forward policy and value calls. Its `train` function then constructs and operates the learning system: optimizer, normalizer, environment rollout, minibatches, nested updates, device coordination, evaluation, and checkpointing. Its nested `minibatch_step`, `sgd_step`, `training_step`, and `training_epoch` functions share dependencies through closure capture. `TrainingState` names only part of the program state: the outer training scan still carries `(TrainingState, environment state, key)`, while the optimization scan carries `(optimizer state, params, key)`. One host-side function owns nearly every lifecycle, and a new concern cuts across its private carries and conventions.

The NodeJAX example leaves PPO explicit: generalized advantage estimation, chunking, shuffling, clipping, and the actor and critic schedules remain statements about the algorithm. Node contracts remove a different class of authoring work:

- no anonymous carry tuples
- no towers of nested transformed functions
- no manual RNG splitting or RNG keys in scan carries
- no recurrent state threaded through policy calls or trainer state
- no hand-authored training-state dataclass
- no host-side epoch or minibatch loops, and no metric buffers: updates are nested scans over one permutation array, and training history is the program's aux
- no separately maintained recurrent-state buffer: the scanned Node records its state alongside the rollout

One honest qualification: the pendulum is continuing control, so every episode boundary here is structural and declared, and no `done` flag exists to handle. Part of brax's carry traffic exists because its tasks terminate. A terminating environment would not change the authoring boundary, though: the per-world reset is one `where` over the plant's own state inside the plant's apply, written for a single world with a scalar `done`, vectorized by `batch`, and invisible to the scans, trainers, and replay around it. What no framework removes is the algorithm content that termination adds, such as masking inside an advantage recursion; that stays an explicit statement here as anywhere.

One `PPO` Node holds the sampler, the value, and the actor and critic trainers as members, receiving them assembled. Scans around the trainer Nodes express minibatches, epochs, and critic passes. The training program composes the training-data Node with `carried(ppo_learner(...))`, so one jitted Node application runs every training iteration and returns the iteration bound to its final state, with the trained policy read out of the actor trainer. Both optimizer states are ordinary Node state rather than locals in one host-side training function.

The policy exposes the same contract with identity or GRU memory. The policy owns `sample`, `logprob`, and `entropy` as methods and the plant owns `observe`, so distribution and observation semantics ride their Nodes through batching, scanning, and the trainers; brax also has distribution objects, routed into each loss function beside the networks. The PPO author still chooses sequence chunks and selects the recorded state at each chunk boundary, because those are PPO decisions. Replay then initializes a fresh scan from that ordinary data. A test verifies the property in both directions: every chunk reproduces the log-probabilities observed during collection, and replay from the neighboring recorded states does not, so the check cannot pass vacuously.

Stock brax PPO can be made recurrent, but recurrence crosses its acting, rollout, storage, minibatching, loss, and evaluation interfaces because the network abstraction does not represent recurrent state transport. In this fixed-horizon NodeJAX example it crosses the places where PPO cares about chunk boundaries; state transport remains the responsibility of the Node and its transforms. Brax implements more production features, but adding features to that training function does not improve its authoring boundary.

The same separation holds for SHAC: [`shac.py`](../examples/rl/shac.py) contains the injected learner and training program, while [`shac_pendulum.py`](../examples/rl/shac_pendulum.py) contains its domain Nodes. The PPO and SHAC assemblies share the `Pendulum` plant and use `ControlledStep` for deterministic closed-loop rollouts. Their policy and value Nodes remain algorithm-specific.

## Reusable transforms

The [`residual`](../examples/comparisons/residual/) comparison asks for compositionality in the smallest case: `input + body(input)`. NodeJAX, NNX, and Equinox can all implement and self-compose that unary case. The earlier NNX and Equinox RNG failures were mistakes in the comparison wrappers, not framework limits.

The limit appears when residual must accept arbitrary framework-native members. NNX graph `Variable` state and explicit functional `Carry` use different calls and initialization interfaces. Equinox native `StatefulLayer` and `State` work through residual, while the returned-successor convention needed by the lifted-stack comparison is a different protocol again. A wrapper cannot infer which argument is the signal, where the output lives, how state starts, or which member methods an enclosing API requires.

The [lifted-stack comparison](../examples/comparisons/lift/) makes the consequence concrete: a state mechanism that composes through one transform may not compose through the next. Supporting each form requires another adapter for construction, state, RNG, aux, call binding, and member interfaces.

A truly set-and-forget residual therefore needs something close to NodeJAX's Def and Contract machinery. The six-line NodeJAX transform is small because those facts already have one representation, not because forwarding them is intrinsically simple. Its own apply remains explicitly unary; supporting additional runtime arguments would still require the transform to declare which field is the residual signal.

## Generic construction

The [`generics`](../examples/comparisons/generics/) comparison leaves
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

The [`imu`](../examples/comparisons/imu/) pipeline combines derivatives
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

The [`tie`](../examples/comparisons/tie/) comparison isolates how each
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
2. NodeJAX wins scan-aligned state-lifetime composition. Boundary policy stays attached to the stateful Node while enclosing scans and trainers change.
3. NodeJAX wins reusable transform authoring. Its transforms preserve one common contract. NNX and Equinox compose within selected local protocols, but transparent reuse across construction, initialization, calls, state, aux, and member interfaces requires protocol-specific adapters that collectively recreate such a contract.
4. NodeJAX wins generic construction and composed setup. Unresolved statics,
   priming, parameter construction, and RNG routing remain in program
   composition instead of constructor plumbing.
5. RNG is as effortless as in PyTorch without giving up JAX's referential
   transparency. A caller supplies one key exactly when the graph consumes
   entropy; transforms derive splitting and forwarding from the contract.
6. Parameter sharing is a draw on capability. The frameworks differ only in
   whether sharing is represented by identity, path, or a declarative route.
