# Framework Comparison Suites

Each subdirectory implements an identical benchmark across NodeJAX and rival Python/JAX frameworks (`Flax NNX`, `Flax Linen`, `Equinox`, `Haiku`, `PyTorch`, `Keras 3`, and raw `JAX`).

## Conventions

- `<suite>_common.py` defines shared synthetic data, constants, and verification assertions with zero `nodejax` or framework-specific imports.
- `<suite>_nodejax.py`, `<suite>_nnx.py`, `<suite>_linen.py`, `<suite>_equinox.py`, `<suite>_haiku.py`, `<suite>_torch.py`, `<suite>_keras.py`, and `<suite>_jax.py` each expose a runnable `main()` verified by [`examples/tests/test_examples_run.py`](../tests/test_examples_run.py).
- Rival implementations import nothing from `nodejax`.

## Suites

| Suite | Question tested | Implementations |
| :--- | :--- | :--- |
| [`tower/`](tower/) | Does the module abstraction survive eight stacked transforms (residual, depth `stack`, time `scanned`, running-stat `Norm`, `ensemble`, inner `finetune`, task `batch`, outer `trained`)? | `nodejax`, `nnx`, `nnx_reusable`, `equinox`, `keras`, `torch` |
| [`ttt/`](ttt/) | What does a reusable online test-time-training wrapper cost across both a stateless `MLP` and a recurrent `RNN` inside an outer meta-learner? | `nodejax`, `nnx`, `equinox`, `haiku`, `torch`, bespoke `jax` |
| [`chunk/`](chunk/) | How many places must change when one state-carrying block resets on a different nested horizon (step, chunk, recording, outer run)? | `nodejax`, `nnx`, `linen`, `equinox`, `haiku`, `torch` |
| [`mode/`](mode/) | How does a `Dropout` + `BatchNorm` network transition from stochastic training to deterministic, batch-isolated evaluation? | `nodejax`, `nnx`, `equinox`, `haiku`, `torch` |
| [`residual/`](residual/) | Can a 6-line `x + body(x)` wrapper compose over state-carrying, stochastic, input-primed, `Aux`-emitting, and recurrent blocks? | `nodejax`, `nnx`, `equinox` |
| [`lift/`](lift/) | Can a lifted `stack` transform compose over state-carrying, stochastic, `Aux`-emitting layers and nest inside itself (`stack(stack(layer))`)? | `nodejax`, `nnx`, `equinox` |
| [`generics/`](generics/) | How many intermediate constructors must forward configuration parameters before an architecture is specialized (`THREADING_TAX`), and can a trained model re-specialize? | `nodejax`, `nnx`, `equinox`, `keras`, `torch` |
| [`imu/`](imu/) | How do input-primed state (`Derivative`), Ornstein-Uhlenbeck drift (`Drift`), and call-time RNG splitting (`Noise`, `Drift`) compose across a five-stage sensor pipeline? | `nodejax`, `nnx`, `equinox` |
| [`tie/`](tie/) | How does each framework tie input and output embedding weights across separate submodules without optimizer duplication or drift? | `nodejax`, `nnx`, `equinox`, `haiku`, `torch` |

See [`docs/comparison.md`](../../docs/comparison.md) for the full cross-suite analysis and [`examples/rl/`](../rl/) for the recurrent reinforcement learning comparison against Brax.
