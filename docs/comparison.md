# Framework Comparison

[`examples/comparisons/`](../examples/comparisons/) and [`examples/rl/`](../examples/rl/) implement identical tasks across NodeJAX, Flax NNX, Flax Linen, Equinox, Haiku, PyTorch, Keras 3 (on JAX), and Brax. Every suite shares a pure-NumPy/JAX `*_common.py` reference harness with zero `nodejax` imports in rival files, so line counts, edit counts, and failure modes come from runnable parity-checked code.

## 1. Deeply transformed architectures (`tower` and `ttt`)

The [`tower`](../examples/comparisons/tower/) stress test stacks eight transformations into one program: a residual RNN cell, scanned over depth (`stack`), unrolled over time (`scanned`) behind a running normalizer, widened into a committee (`ensemble` + `reduce`), adapted per task (`finetune(train_step(...))`), mapped over task batches (`batch`), and meta-trained (`trained(train_step(...))`).

In NodeJAX ([`tower_nodejax.py`](../examples/comparisons/tower/tower_nodejax.py), 114 lines total), the entire architecture and nested meta-trainer is five lines:

```python
rnn = stack(residual(RNN(HIDDEN)), n=LAYERS)
member = Up(HIDDEN) >> Norm(HIDDEN, MOMENTUM) >> rnn >> Readout(HIDDEN)
committee = ensemble(scanned(member), n=MEMBERS) >> reduce(jnp.mean)
adapt = finetune(train_step(committee, mse, optax.sgd(INNER_LR)))
return trained(train_step(batch(adapt), mse, optax.adam(OUTER_LR)))
```

Every line consumes a `Node` and returns a `Node`. No rival framework's module abstraction survives this program intact:

| Implementation | Total lines | Where the abstraction breaks |
| :--- | ---: | :--- |
| NodeJAX (`tower_nodejax.py`) | 114 | Nowhere; architecture, state, adaptation, and meta-training stay in one `Node` algebra |
| Flax NNX bespoke (`tower_nnx.py`) | 162 | Architecture is scattered across per-site `nnx.StateAxes`, `nnx.split_rngs`, and manual `(h_layers, stats)` carries |
| Equinox (`tower_equinox.py`) | 168 | Modules cover parameters only; depth, time, committee, inner SGD, and outer Adam drop to raw `lax.scan` and `vmap` |
| PyTorch (`tower_torch.py`) | 181 | Abandons `nn.Module` for `torch.func.functional_call` dictionaries and unrolls depth and time in Python across a dynamic autograd tape |
| Keras 3 on JAX (`tower_keras.py`) | 205 | Compiles via `lax.scan` (427 jaxpr equations), yet fails to construct an ensemble under `vmap` because `keras.initializers` crash on traced seeds and `add_weight` under `vmap` silently aliases every member onto one tensor so a 3-member committee trains as 3 identical clones |
| Flax NNX reusable (`tower_nnx_reusable.py`) | 322 | Invents custom `Residual`, `Serial`, `Stack`, `Scan`, and `Ensemble` wrappers (2.8x NodeJAX's file size) and still abandons that protocol at `maml_fit` |

The [`ttt`](../examples/comparisons/ttt/) benchmark isolates reusable online test-time training inside an outer meta-learner, testing each framework's `TTT` wrapper on both a stateless `MLP` predictor and a recurrent `RNN` predictor (1024 meta-weights each, verified in [`ttt_common.py`](../examples/comparisons/ttt/ttt_common.py)):

| Implementation | Total lines | What reusable `TTT` requires |
| :--- | ---: | :--- |
| Raw JAX bespoke (`ttt_jax.py`) | 89 | Hand-rolled `RNN`-only `lax.scan` baseline with zero reusable wrapper; switching to `MLP` or another optimizer rewrites the scan body |
| NodeJAX (`ttt_nodejax.py`) | 107 | One 3-line `ttt_meta(model)` function (`next_step_ttt(train_step(model, xent, learned_sgd(TTT_LR0)))`) wraps both stateless `MLP` (`PNode`) and cyclic `RNN` (`PSNode`) with zero protocol branching |
| PyTorch (`ttt_torch.py`) | 154 | Branches on `hasattr(model, 'init_carry')`, evicts weights out of `nn.Module` into dicts for `functional_call`, mangles `.` into `/` for `ParameterDict` rate keys, abandons `torch.optim` for inner updates, and unrolls time in Python |
| Equinox (`ttt_equinox.py`) | 162 | Branches on `hasattr(model, 'init_carry')` because `eqx.Module` has no state-passing call contract, partitions arrays via `eqx.filter` for rates, and threads `(fast, carry, prev)` through `lax.scan` by hand |
| Haiku (`ttt_haiku.py`) | 164 | Nests an inner `hk.transform_with_state` inside the outer `hk.transform`, lifts inner parameters via `hk.lift`, registers a parallel `hk.get_parameter` rate tree, and threads `(fast_p, fast_s, prev)` by hand |
| Flax NNX (`ttt_nnx.py`) | 178 | Branches on `hasattr(model, 'initialize_carry')` across NNX's split `Module(x)` vs. `Cell(carry, x)` protocols, clones `self.model` per sequence, subclasses `nnx.Param` (`Rate`) to separate inner and outer parameters, and threads `(carry, prev)` manually |

## 2. Multi-horizon state lifetimes (`chunk`)

The [`chunk`](../examples/comparisons/chunk/) benchmark nests steps, chunks, recordings, and outer runs where hidden state, running normalization statistics, and optimizer momentum each reset on different boundaries. Every implementation is asserted against identical plain-JAX numerical references in [`chunk_common.py`](../examples/comparisons/chunk/chunk_common.py).

Changing normalization statistics to reset per recording exposes who actually owns state lifetimes:

| Framework | Independent edits | Failure mode when one edit is missed |
| :--- | ---: | :--- |
| NodeJAX (`chunk_nodejax.py`, 260 lines) | 1 | `state_reinit(Norm(), 'recording')` on the owning `Node`; unclaimed boundary names raise an immediate error |
| Haiku (`chunk_haiku.py`, 247 lines) | 1 | Reset logic lives inside one outer rollout function by string key; reusing the block in another rollout or trainer silently drops the reset |
| Flax NNX (`chunk_nnx.py`, 266 lines) | 2 | Inference and training scans diverge silently |
| PyTorch (`chunk_torch.py`, 310 lines) | 2 | Imperative buffer-reset loops diverge between evaluation and training |
| Flax Linen (`chunk_linen.py`, 290 lines) | 4 | Unchecked string collection paths and manual init dicts silently compute wrong statistics |
| Equinox (`chunk_equinox.py`, 310 lines) | 6 | Initialization, step return tuples, and outer carries still type-check and run with stale state |

In NNX, Linen, Equinox, and PyTorch, state lifetimes are encoded implicitly in surrounding loop plumbing rather than on the state-owning unit. Omitting one reset line produces a valid program that silently computes the wrong trajectory, forcing a reader to audit every enclosing loop body.

## 3. Train and evaluation mode (`mode`)

The [`mode`](../examples/comparisons/mode/) benchmark trains a `Dropout` + `BatchNorm` network and builds a deterministic single-sample evaluator with frozen statistics:

- PyTorch (`mode_torch.py`, 71 lines) and Flax NNX (`mode_nnx.py`, 71 lines) mutate mode booleans in place on shared module instances, leaving stochastic machinery and mutable stat slots alive on the object behind flag discipline.
- Haiku (`mode_haiku.py`, 79 lines) pollutes every forward call signature with a threaded `is_training: bool` flag.
- Equinox (`mode_equinox.py`, 108 lines) rewrites boolean leaves via `eqx.nn.inference_mode` while still forcing every caller to thread `eqx.nn.State` in and out by hand and pass `key=None`.
- NodeJAX (`mode_nodejax.py`, 70 lines) builds a distinct evaluation `Node` via `nn.eval_mode(final)` (`final.specialize(**{'*.train': False})` plus `tree_freeze(..., tag='running_stats')`). `Dropout` specializes to the identity so the evaluator rejects `rng` keys outright, and `BatchNorm` statistics freeze into constants (`not evaluator.cyclic`).

## 4. Reusable transforms and leaky module protocols (`residual` and `lift`)

Writing a toy unary `x + body(x)` works in any library. [`residual`](../examples/comparisons/residual/) and [`lift`](../examples/comparisons/lift/) test what happens when `residual` and `stack` have to act as general library transforms over state-carrying, stochastic, input-primed, `Aux`-emitting, and self-nested blocks.

Both Flax NNX (`residual_nnx.py`, 244 lines; `lift_nnx.py`, 344 lines) and Equinox (`residual_equinox.py`, 269 lines; `lift_equinox.py`, 359 lines) fracture across incompatible state protocols inside their own ecosystems:

- NNX splits graph `Variable` mutation from functional RNN `(carry, input)` cells, so a general NNX `Residual` cannot identify the signal argument or preserve the cell interface expected by `nnx.RNN` (`functional-state=NO cell-interface=NO`). Worse, if `body(x)` mutates a `Variable` and the residual `+` then fails on a shape mismatch, the module is left mutated in a corrupted half-stepped state.
- Equinox's `StatefulLayer` (`eqx.nn.State`) cannot cross into lifted `stack` because `eqx.nn.StateIndex` fails to create independent per-layer state slots under `filter_vmap`, forcing `lift_equinox.py` onto a second `(successor, output)` convention that `StatefulLayer`-based `Residual` cannot compose (`native-state=yes successor-state=NO`). Similarly, `State.set` invalidates the old `State` before `x + out` runs, corrupting state on downstream shape errors.
- Both rival `stack` implementations inspect `inspect.signature(member.__call__)` to detect RNG consumption. Because `LayerStack.__call__` must declare an optional `rng` parameter in case it wraps a stochastic layer, inspection confuses forwarding `rng` with requiring `rng`, breaking `stack(stack(bare_layer))` unless the author bolts on explicit `takes_rng`, `WithAux`, and `make_layer` declarations that piecemeal reinvent NodeJAX's `Def` and `Contract`.

NodeJAX's `residual` is 6 lines in [`nodejax/transforms/wiring/residual.py`](../nodejax/transforms/wiring/residual.py) (`residual_nodejax.py` 197 lines; `lift_nodejax.py` 243 lines) because every `Leaf`, `Composite`, `Wrapper`, and transform product already normalizes to one `Def` and `Contract`.

## 5. Recurrent reinforcement learning (`ppo`, `shac`, and `sac` vs. Brax)

[`examples/rl/ppo.py`](../examples/rl/ppo.py), [`examples/rl/shac.py`](../examples/rl/shac.py), and [`examples/rl/sac.py`](../examples/rl/sac.py) test where abstractions stop in full RL algorithms compared against Brax's [`ppo/train.py`](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py).

Brax's network abstraction dies at a stateless feed-forward call. Everything above that call lives in one monolithic `train` function of nested closures (`minibatch_step`, `sgd_step`, `training_step`, `training_epoch`) passing ad-hoc carry tuples (`(TrainingState, env_state, key)` outside, `(opt_state, params, key)` inside) where `TrainingState` fails to hold the whole training state. Making Brax PPO recurrent cuts across acting, rollout storage, minibatch slicing, loss signatures, and evaluation because nothing in its module layer transports state. Brax ships broader production features (multi-device `pmap`, episodic `done` resets, evaluation hooks, checkpointing), whereas its module boundary helps zero lines above the MLP.

In NodeJAX, policies (feed-forward or `GRU`), physical plants, `nn.EMA` target networks, Optax trainers, and off-policy replay (`Buffer(capacity)` in [`replay.py`](../examples/rl/replay.py)) are all ordinary `Node`s sharing one state carry:

- Swapping a feed-forward policy for a `GRU` policy changes zero lines in `ControlledStep`, `SHACIteration`, `PPOIteration`, or `SACIteration`.
- `PPOIteration` records `policy.state` during collection via `scan(..., record=True)` and resumes shuffled chunks from their recorded initial states via `ChunkReplay` (`self.run.bind(state=initial)(...)`), with zero hand-rolled state buffers or RNG carries.
- `SACIteration` keeps its ring buffer, cursor, and fill count as ordinary `Node` state, compiling rollout collection, buffer insertion, minibatch sampling, and gradient updates into a single `scanned` transition.

## 6. Deferred configuration, priming, and weight tying (`generics`, `imu`, and `tie`)

In [`generics`](../examples/comparisons/generics/), width, depth, ensemble size, and inner `Block` temperature are left unspecified while assembling `serial(entry=nn.Linear(), blocks=stack(Block()), readout=nn.Linear(1))`, then resolved by path via `.specialize(...)` (`generics_nodejax.py`, 104 lines, `THREADING_TAX = 0`). NNX (`generics_nnx.py`, 139 lines), PyTorch (`generics_torch.py`, 141 lines), and Equinox (`generics_equinox.py`, 182 lines) cannot compose an unfinished architecture at all, forwarding 5 configuration arguments through intermediate `__init__` chains that never use them (`THREADING_TAX = 5`). Keras 3 (`generics_keras.py`, 250 lines) matches NodeJAX on dataflow-inferred fan-in via lazy `build(input_shape)`, while still forwarding 4 architectural hyperparameters through Python builder functions (`THREADING_TAX = 4`).

In [`imu`](../examples/comparisons/imu/), chaining two input-primed finite derivatives, white noise, Ornstein-Uhlenbeck drift, and quantization (`Derivative >> Derivative >> Noise >> Drift >> Quantize`) is a single `>>` expression in NodeJAX (`imu_nodejax.py`, 90 lines) whose priming, physical state container, and call-time `KeyStream` splitting derive from the leaves. NNX (`imu_nnx.py`, 101 lines) wires priming samples and named `nnx.Rngs` streams manually inside `IMU.__init__`, while Equinox (`imu_equinox.py`, 115 lines) repeats the stage topology across `IMUState`, `IMU.init`, and `IMU.step` plus manual step-key splitting so inserting one state-carrying stage requires four coordinated edits.

In [`tie`](../examples/comparisons/tie/), tying input and output embeddings is a five-way capability draw across NodeJAX (`tie_nodejax.py`, 62 lines), Haiku (`tie_haiku.py`, 81 lines), Flax NNX (`tie_nnx.py`, 89 lines), PyTorch (`tie_torch.py`, 89 lines), and Equinox's `eqx.nn.Shared` (`tie_equinox.py`, 115 lines); all five keep a single parameter leaf for the optimizer and produce zero drift (only naive Equinox array aliasing duplicates the leaf on PyTree flatten and drifts on step 1).

## Verdicts

| Dimension | Suites | Outcome |
| :--- | :--- | :--- |
| Whole-program closure | `tower`, `ttt`, `rl` | Rival module abstractions stop at the forward pass of one network and collapse into closure soup, `functional_call` dicts, `hasattr` carry branching, or 320-line partial wrappers once adaptation, physics rollouts, replay buffers, and meta-optimizers enter; NodeJAX keeps the full program inside one `Node` algebra |
| State lifetimes and modes | `chunk`, `mode` | Encoding reset schedules and train/eval modes in loop carries or mutable flags scatters invariants across 2 to 6 sites where one missed line silently computes wrong results; attaching `state_reinit` and `nn.eval_mode` to the `Node` reduces those edits to 1 checked declaration |
| Transform composability | `residual`, `lift` | Signature inspection and split state protocols (`Variable` vs. `Carry`, `StatefulLayer` vs. `(successor, output)`) fail on nested or cross-protocol composition; one normalized `Def` and `Contract` makes a 6-line wrapper work across every `Node` |
| Deferred sizing and setup | `generics`, `imu` | `Generic` blueprints eliminate constructor-chain parameter forwarding (0 forwarded args vs. 4-5), and `>>` composes input priming and call-time `KeyStream` routing alongside forward execution |
| Weight sharing | `tie` | Capability tie across NodeJAX `tie`, NNX, PyTorch, Haiku, and `eqx.nn.Shared` |
