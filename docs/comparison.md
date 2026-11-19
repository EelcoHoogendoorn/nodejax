# Framework Comparison

`Flax NNX`, `Flax Linen`, `Equinox`, `Haiku`, `PyTorch`, and `Keras 3` standardize container representation or single-network forward execution without standardizing a unified `(param, state, input) -> (next_state, output)` transition contract, whereas `nodejax` defines a computation-graph `Node` contract one level below domain libraries. [`examples/comparisons/`](../examples/comparisons/) and [`examples/rl/`](../examples/rl/) implement identical tasks across these frameworks and Brax to measure what happens as programs cross `cyclic` state, multi-horizon resets, physical rollouts, and nested optimization. Every suite shares a pure-NumPy/JAX `*_common.py` task harness with zero `nodejax` imports in rival files (`chunk` checks numerical trace parity against plain-JAX reference functions, while the remaining suites assert structural invariants, zero-drift weight sharing, mode isolation, transform probe matrices, and finite decreasing losses).

Core line counts throughout this document measure executable model, transform, and training code via AST and token filtering (excluding blank lines, comments, docstrings, imports, diagnostic probes, and test/reporting harnesses); `residual` and `lift` report the transform implementation itself.

## 1. Deeply transformed architectures (`tower` and `ttt`)

The [`tower`](../examples/comparisons/tower/) stress test stacks eight transformations into one program: a residual RNN cell, scanned over depth (`stack`), unrolled over time (`scanned`) behind a running normalizer, widened into a committee (`ensemble` + `reduce`), adapted per task (`finetune(train_step(...))`), mapped over task batches (`batch`), and meta-trained (`trained(train_step(...))`).

In `nodejax` ([`tower_nodejax.py`](../examples/comparisons/tower/tower_nodejax.py), 58 core lines), the entire architecture and nested meta-trainer is five lines:

```python
rnn = stack(residual(RNN(HIDDEN)), n=LAYERS)
member = Up(HIDDEN) >> Norm(HIDDEN, MOMENTUM) >> rnn >> Readout(HIDDEN)
committee = ensemble(scanned(member), n=MEMBERS) >> reduce(jnp.mean)
adapt = finetune(train_step(committee, mse, optax.sgd(INNER_LR)))
return trained(train_step(batch(adapt), mse, optax.adam(OUTER_LR)))
```

Every line consumes a `Node` and returns a `Node`. No rival framework's module abstraction survives this program intact:

| Implementation | Core lines | Where the abstraction breaks |
| :--- | ---: | :--- |
| `nodejax` (`tower_nodejax.py`) | 58 | Nowhere; architecture, state, adaptation, and meta-training stay in one `Node` algebra |
| `PyTorch` (`tower_torch.py`) | 76 | Abandons `nn.Module` for `torch.func.functional_call` dictionaries and unrolls depth and time in Python across a dynamic autograd tape |
| `Equinox` (`tower_equinox.py`) | 81 | Modules cover parameters only; depth, time, committee, inner SGD, and outer Adam drop to raw `lax.scan` and `vmap` |
| `Flax NNX` bespoke (`tower_nnx.py`) | 98 | Architecture is scattered across per-site `nnx.StateAxes`, `nnx.split_rngs`, and manual `(h_layers, stats)` carries |
| `Keras 3` on JAX (`tower_keras.py`) | 127 | Compiles single-model `keras.layers.RNN` via `lax.scan`, yet provides no module-level ensemble or inner-adaptation transform, exiting `keras.Layer` into Python-built member lists, `stack_members`, `stateless_call`, and raw `jax.grad` |
| `Flax NNX` reusable (`tower_nnx_reusable.py`) | 213 | Invents custom `Residual`, `Serial`, `Stack`, `Scan`, and `Ensemble` wrappers (3.7x `nodejax`'s core code) and still abandons that protocol at `maml_fit` |

The [`ttt`](../examples/comparisons/ttt/) benchmark isolates reusable online test-time training inside an outer meta-learner, testing each framework's `TTT` wrapper on both an acyclic `MLP` predictor and a `cyclic` `RNN` predictor (1024 meta-weights each, verified in [`ttt_common.py`](../examples/comparisons/ttt/ttt_common.py)):

| Implementation | Core lines | What reusable `TTT` requires |
| :--- | ---: | :--- |
| Raw `JAX` bespoke (`ttt_jax.py`) | 47 | Hand-rolled `RNN`-only `lax.scan` baseline with zero reusable wrapper; switching to `MLP` or another optimizer rewrites the scan body |
| `nodejax` (`ttt_nodejax.py`) | 53 | One 3-line `ttt_meta(model)` function (`next_step_ttt(train_step(model, xent, learned_sgd(TTT_LR0)))`) wraps both acyclic `MLP` and `cyclic` `RNN` `Node`s with zero protocol branching |
| `PyTorch` (`ttt_torch.py`) | 92 | Branches on `hasattr(model, 'init_carry')`, evicts weights out of `nn.Module` into dicts for `functional_call`, mangles `.` into `/` for `ParameterDict` rate keys, abandons `torch.optim` for inner updates, and unrolls time in Python |
| `Equinox` (`ttt_equinox.py`) | 96 | Branches on `hasattr(model, 'init_carry')` because `eqx.Module` leaves step methods unspecialized, partitions arrays via `eqx.filter` for rates, and threads `(fast, carry, prev)` through `lax.scan` by hand |
| `Flax NNX` (`ttt_nnx.py`) | 104 | Branches on `hasattr(model, 'initialize_carry')` across NNX's split `Module(x)` vs. `Cell(carry, x)` protocols, clones `self.model` per sequence, subclasses `nnx.Param` (`Rate`) to separate inner and outer parameters, and threads `(carry, prev)` manually |
| `Haiku` (`ttt_haiku.py`) | 105 | Nests an inner `hk.transform_with_state` inside the outer `hk.transform`, lifts inner parameters via `hk.lift`, registers a parallel `hk.get_parameter` rate tree, and threads `(fast_p, fast_s, prev)` by hand |

## 2. Multi-horizon state lifetimes (`chunk`)

The [`chunk`](../examples/comparisons/chunk/) benchmark nests steps, chunks, recordings, and outer runs where hidden state, running normalization statistics, and optimizer momentum each reset on different boundaries. Every implementation is asserted against identical plain-JAX numerical references in [`chunk_common.py`](../examples/comparisons/chunk/chunk_common.py).

Changing normalization statistics to reset per recording exposes who actually owns state lifetimes:

| Framework | Independent edits | Failure mode when one edit is missed |
| :--- | ---: | :--- |
| `nodejax` (`chunk_nodejax.py`, 65 core lines) | 1 | `state_reinit(Norm(), 'recording')` on the owning `Node`; unclaimed boundary names raise an immediate error |
| `Haiku` (`chunk_haiku.py`, 83 core lines) | 1 | Rollout reset lives inside `_recordings` by string key and silently drops on reuse, while its multi-session trainer carries `state0` separately across raw `lax.scan` loops |
| `PyTorch` (`chunk_torch.py`, 115 core lines) | 2 | Imperative buffer-reset loops diverge between evaluation and training |
| `Flax NNX` (`chunk_nnx.py`, 118 core lines) | 2 | Inference and training scans diverge silently |
| `Flax Linen` (`chunk_linen.py`, 123 core lines) | 4 | Unchecked string collection paths and manual init dicts silently compute wrong statistics |
| `Equinox` (`chunk_equinox.py`, 118 core lines) | 6 | Initialization, step return tuples, and outer carries still type-check and run with stale state |

In `Flax NNX`, `Flax Linen`, `Equinox`, and `PyTorch`, state lifetimes are encoded implicitly in surrounding loop plumbing rather than on the state-owning unit. Omitting one reset line produces a valid program that silently computes the wrong trajectory, forcing a reader to audit every enclosing loop body.

## 3. Train and evaluation mode (`mode`)

The [`mode`](../examples/comparisons/mode/) benchmark trains a `Dropout` + `BatchNorm` network and builds a deterministic single-sample evaluator with frozen statistics:

- `Flax NNX` (`mode_nnx.py`, 30 core lines) and `PyTorch` (`mode_torch.py`, 33 core lines) mutate mode booleans in place on shared module instances, leaving stochastic machinery and mutable statistics alive on the object behind flag discipline.
- `Haiku` (`mode_haiku.py`, 39 core lines) threads an explicit `is_training: bool` flag through every forward call signature.
- `Equinox` (`mode_equinox.py`, 64 core lines) rewrites boolean leaves via `eqx.nn.inference_mode` while still forcing every caller to thread `eqx.nn.State` in and out by hand and pass `key=None`.
- `nodejax` (`mode_nodejax.py`, 28 core lines) builds a distinct evaluation `Node` via `nn.eval_mode(final)` (`final.specialize(**{'*.train': False})` plus `tree_freeze(..., tag='running_stats')`). `Dropout` specializes to the identity so the evaluator rejects `rng` keys outright, and `BatchNorm` statistics freeze into constants (`not evaluator.cyclic`).

## 4. Reusable transforms and split module protocols (`residual` and `lift`)

Writing a toy unary `x + body(x)` works in any library. [`residual`](../examples/comparisons/residual/) and [`lift`](../examples/comparisons/lift/) test what happens when `residual` and `stack` have to act as general library transforms over `cyclic`, stochastic, input-primed, `Aux`-emitting, and self-nested blocks.

Both `Flax NNX` (`Residual`, 6 core lines; `LayerStack`, 41 core lines) and `Equinox` (`Residual`, 26 core lines including its local `WithAux` protocol; `LayerStack`, 42 core lines) split across incompatible state protocols at the framework level and hit signature-inspection limits at the wrapper level:

- `Flax NNX` splits graph `Variable` mutation from functional RNN `(carry, input)` cells, so a general NNX `Residual` cannot identify the signal argument or preserve the cell interface expected by `nnx.RNN` (`functional-state=NO cell-interface=NO`). Worse, if `body(x)` mutates a `Variable` and the residual `+` then fails on a shape mismatch, the module is left mutated in a corrupted half-stepped state.
- `Equinox`'s `StatefulLayer` (`eqx.nn.State`) cannot cross into lifted `stack` because `eqx.nn.StateIndex` fails to allocate independent per-layer state entries under `filter_vmap`, forcing `lift_equinox.py` onto a second `(successor, output)` convention that `StatefulLayer`-based `Residual` cannot compose (`native-state=yes successor-state=NO`). Similarly, `State.set` invalidates the old `State` before `x + out` runs, corrupting state on downstream shape errors.
- At the wrapper level, because neither framework records per-role `rng` metadata on modules, both `LayerStack` implementations inspect `inspect.signature(member.__call__)` to detect stochasticity. Because `LayerStack.__call__` must declare an optional `rng` parameter in case it wraps a stochastic layer, inspection confuses forwarding `rng` with requiring `rng`, breaking `stack(stack(bare_layer))` unless the author bolts on explicit `takes_rng`, `WithAux`, and `make_layer` declarations that piecemeal reinvent `nodejax`'s `Def` and `Contract`.

`nodejax`'s library transforms are 9 core lines in [`nodejax/transforms/wiring/residual.py`](../nodejax/transforms/wiring/residual.py) and 41 core lines in [`nodejax/transforms/iteration/stack.py`](../nodejax/transforms/iteration/stack.py), require zero user wrapper code, and pass every composition probe because every `Leaf`, `Composite`, `Wrapper`, and transform product already normalizes to one `Def` and `Contract`.

## 5. Recurrent reinforcement learning (`ppo`, `shac`, and `sac` vs. Brax)

[`examples/rl/ppo.py`](../examples/rl/ppo.py), [`examples/rl/shac.py`](../examples/rl/shac.py), and [`examples/rl/sac.py`](../examples/rl/sac.py) test where abstractions stop in full RL algorithms compared against Brax's [`ppo/train.py`](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py).

Brax's network abstraction stops at a stateless feed-forward call. Everything above that call lives in one monolithic `train` function of nested closures (`minibatch_step`, `sgd_step`, `training_step`, `training_epoch`) passing ad-hoc carry tuples (`(TrainingState, env_state, key)` outside, `(opt_state, params, key)` inside) where `TrainingState` fails to hold the whole training state. Making Brax PPO recurrent cuts across acting, rollout storage, minibatch slicing, loss signatures, and evaluation because nothing in its module layer transports state. Brax ships broader production features (multi-device `pmap`, episodic `done` resets, evaluation hooks, checkpointing), whereas its module boundary helps zero lines above the MLP.

In `nodejax`, policies (acyclic MLP or `cyclic` `GRU`), physical plants, `nn.EMA` target networks, Optax trainers, and off-policy replay (`Buffer(capacity)` in [`replay.py`](../examples/rl/replay.py)) are all ordinary `Node`s sharing one state carry:

- Swapping an acyclic MLP policy for a `cyclic` `GRU` policy changes zero lines in `ControlledStep`, `SHACIteration`, `PPOIteration`, or `SACIteration`.
- `SamplingStep` captures `policy_state = self.policy.state` in its returned transition `Struct`, and `PPOIteration` resumes shuffled chunks from their recorded initial states via `ChunkReplay` (`self.run.bind(state=initial)(...)`), with zero hand-rolled state buffers or RNG carries.
- `SACIteration` keeps its ring buffer, cursor, and fill count as ordinary `Node` state, compiling rollout collection, buffer insertion, minibatch sampling, and gradient updates into a single `scanned` transition.

## 6. Deferred configuration, priming, and weight tying (`generics`, `imu`, and `tie`)

In [`generics`](../examples/comparisons/generics/), width, depth, ensemble size, and inner `Block` temperature are left unspecified while assembling `serial(entry=nn.Linear(), blocks=stack(Block()), readout=nn.Linear(1))`, then resolved by path via `.specialize(...)` (`generics_nodejax.py`, 53 core lines, `THREADING_TAX = 0`). `Flax NNX` (`generics_nnx.py`, 61 core lines), `PyTorch` (`generics_torch.py`, 62 core lines), and `Equinox` (`generics_equinox.py`, 73 core lines) cannot compose an unfinished architecture at all, forwarding 5 configuration arguments through intermediate `__init__` chains that never use them (`THREADING_TAX = 5`). `Keras 3` (`generics_keras.py`, 129 core lines) matches `nodejax` on dataflow-inferred fan-in via lazy `build(input_shape)`, while still forwarding 4 architectural hyperparameters through Python builder functions (`THREADING_TAX = 4`).

In [`imu`](../examples/comparisons/imu/), chaining two input-primed finite derivatives, white noise, Ornstein-Uhlenbeck drift, and quantization (`Derivative >> Derivative >> Noise >> Drift >> Quantize`) is a single `>>` expression in `nodejax` (`imu_nodejax.py`, 43 core lines) whose priming, physical state container, and call-time `KeyStream` splitting derive from the leaves. `Flax NNX` (`imu_nnx.py`, 61 core lines) wires priming samples and named `nnx.Rngs` streams manually inside `IMU.__init__`, while `Equinox` (`imu_equinox.py`, 70 core lines) repeats the stage topology across `IMUState`, `IMU.init`, and `IMU.step` plus manual step-key splitting so inserting one `cyclic` stage requires four coordinated edits.

In [`tie`](../examples/comparisons/tie/), tying input and output embeddings is a five-way capability draw across `nodejax` (`tie_nodejax.py`, 24 core lines), `Haiku` (`tie_haiku.py`, 42 core lines), `Flax NNX` (`tie_nnx.py`, 48 core lines), `PyTorch` (`tie_torch.py`, 50 core lines), and `Equinox`'s `eqx.nn.Shared` (`tie_equinox.py`, 61 core lines); all five keep a single parameter leaf for the optimizer and produce zero drift (only naive Equinox array aliasing duplicates the leaf on PyTree flatten and drifts on step 1).

## Verdicts

| Dimension | Suites | Outcome |
| :--- | :--- | :--- |
| Whole-program closure | `tower`, `ttt`, `rl` | Rival module abstractions stop at the forward pass of one network and collapse into closure soup, `functional_call` dicts, `stateless_call` lists, `hasattr` carry branching, or 213-line partial wrappers once adaptation, physics rollouts, replay buffers, and meta-optimizers enter; `nodejax` keeps the full program inside one `Node` algebra |
| State lifetimes and modes | `chunk`, `mode` | Encoding reset schedules and train/eval modes in loop carries or mutable flags scatters invariants across 2 to 6 sites where one missed line silently computes wrong results; attaching `state_reinit` and `nn.eval_mode` to the `Node` reduces those edits to 1 checked declaration |
| Transform composability | `residual`, `lift` | Signature inspection and split state protocols (`Variable` vs. `Carry`, `StatefulLayer` vs. `(successor, output)`) fail on nested or cross-protocol composition; one normalized `Def` and `Contract` makes a 9-line `residual` and 41-line `stack` work across every `Node` |
| Deferred sizing and setup | `generics`, `imu` | `Generic` blueprints eliminate constructor-chain parameter forwarding (0 forwarded args vs. 4-5), and `>>` composes input priming and call-time `KeyStream` routing alongside forward execution |
| Weight sharing | `tie` | Capability tie across `nodejax` `tie`, `Flax NNX`, `PyTorch`, `Haiku`, and `eqx.nn.Shared` |
