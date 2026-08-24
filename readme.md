# NodeJAX

[![tests](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml/badge.svg)](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml)

> **Purely functional JAX middleware: JAX without PyTorch envy.**

A learned controller, its recurrent state, a differentiable physical system,
and the optimizer training the whole experiment can be assembled as one
component:

```python
core = Up(HIDDEN) >> stack(RNN(HIDDEN), n=LAYERS) >> Readout(HIDDEN)
controller = Norm() >> ensemble(core, n=MEMBERS) >> reduce(jnp.mean)
rollout = batch(scanned(closed_loop(controller >> Motor(DT))))

model = rollout.parameterize(rng=jax.random.PRNGKey(0)).initialize()
trainer = train_step(model, mse, optax.adam(0.02))
final, aux = trained(trainer).apply(
    input=reference_batches,
    target=reference_batches,
)
```

A committee of recurrent controllers drives a simulated motor in closed loop.
The outer trainer differentiates through the complete rollout, including the
controller, feedback, actuator saturation, and motor dynamics. Recurrent carry,
running statistics, plant state, active weights, and optimizer moments all use
the same explicit state contract.

The same system can go one step further. Build its recurrent core from a
learning process:

```python
from nodejax.transforms.train_step import learned_sgd

adaptive_rnn = reconstruction_ttt(
    train_step(RNN(HIDDEN), mse, learned_sgd(0.01))
)
adaptive_core = Up(HIDDEN) >> adaptive_rnn >> Readout(HIDDEN)
controller = Norm() >> ensemble(adaptive_core, n=MEMBERS) >> reduce(jnp.mean)
```

Now each controller adapts its own weights during every control step, while the
outer trainer learns the initial weights and adaptation rates through the
simulated physics. The outer rollout and training expressions stay unchanged.

This is abridged from the executable
[`test_motor_control.py`](nodejax/tests/test_motor_control.py), which contains
the component definitions, data construction, and assertions that the system
learns.

## The idea

NodeJAX introduces no new execution model; it gives JAX's existing functional
model an immutable compositional object form.

```python
@node
def RNN(activation: Callable) -> Node:
    def param(memory: float):
        return Struct(memory=memory)

    def init():
        return 0

    def apply(param, state, input):
        next_state = activation(input + param.memory * state)
        return next_state, next_state

    return Leaf(apply, param=param, init=init)
```

The separate static argument binding facilitates `jit`; a `grad` of its parameters is unambiguous. `apply` has exactly the transition shape that `scan` requires. The separate roles tell `vmap` what a transform may share or map.

Anything you can do with JAX, you can do within this contract. Composition and valid transforms preserve this contract, so their results remain components and can be composed again.

Randomness remains equally explicit. A public role call accepts one raw
`rng=key` exactly when that role requires entropy. Internally, a
`MaybeKeyStream` routes and splits the key according to declared child
requirements. An authored leaf that declares `rng` receives a keyed
`KeyStream` and draws with `rng.next()`.

## See it working

Examples are written as tests so their claims can be checked:

- [`test_train_loop.py`](nodejax/examples/test_train_loop.py) shows an ordinary
  training loop in compiled chunks, with host-side logging and early stopping.
- [`test_kv_cache.py`](nodejax/examples/test_kv_cache.py) treats a decoder cache
  as ordinary state, including branched decoding and batched users with shared
  params.
- [`test_maml_composed.py`](nodejax/examples/test_maml_composed.py) expresses
  MAML by placing an adapting trainer inside an outer trainer.
- [`test_gan.py`](nodejax/examples/test_gan.py) represents both sides of a GAN
  and their optimizers inside one differentiable program.
- [`examples/comparisons/`](nodejax/examples/comparisons/) contains runnable
  NodeJAX, JAX, Equinox, Flax, Haiku, and PyTorch formulations of the same
  problems.

The complete gallery lives in [`nodejax/examples/`](nodejax/examples/).

## Read next

- The [cookbook](docs/cookbook.md) builds from a minimal component to state,
  composition, transforms, and training.
- The [handbook](docs/handbook.md) is the technical reference for the contract,
  binding stages, authoring system, and transforms.
- The [design philosophy](docs/philosophy.md) explains why components, explicit
  lifetimes, value semantics, and transform closure belong together.
- The [framework comparison](docs/comparison.md) examines ecosystem tradeoffs
  through side-by-side implementations.

## Status

NodeJAX is under active development and its API is evolving. The test suite and
example programs are the executable specification. It is licensed under the
[MIT License](LICENSE).

## Install

NodeJAX requires Python 3.11 or newer. To install from a source checkout:

```console
git clone https://github.com/EelcoHoogendoorn/nodejax.git
cd nodejax
pip install -e .
```
