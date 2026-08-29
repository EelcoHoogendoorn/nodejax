# NodeJAX

[![tests](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml/badge.svg)](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml)

NodeJAX is purely functional JAX middleware that organizes code into composable Node transformations.

JAX makes functions closed under composition and transformation. NodeJAX gives stateful Nodes the same property.

| | JAX functions | NodeJAX Nodes |
| :--- | :--- | :--- |
| Closure under composition | `f = lambda x: sum(sigmoid(x))` | `node = nn.RNN(64) >> nn.relu` |
| Closure under transforms | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

## Example

```python
core = nn.Linear(HIDDEN) >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH) >> nn.Projection()
controller = ensemble(core, n=MEMBERS) >> reduce(jnp.mean)
rollout = batch(scanned(closed_loop(controller >> Motor(DT))))

model = rollout.with_input(input[0]).parameterize(rng=jax.random.PRNGKey(0)).initialize()
trainer = train_step(model, tracking_loss, optax.adam(0.02))
final, aux = trained(trainer).apply(input, target)
```

A committee of residual GRU stacks drives a simulated motor in closed loop. Recurrent carry, plant state, active weights, and optimizer moments all use the same explicit state contract.

This is abridged from the executable [`test_motor_control.py`](examples/test_motor_control.py).

## The idea

NodeJAX introduces no new execution model; it gives JAX's existing functional model an immutable compositional object form.

```python
@node
def WindowedFilter(window_size: int) -> Node:
    def param(rng):
        return jax.random.normal(rng.next(), (window_size,)) / jnp.sqrt(window_size)

    def init(param, input):
        return jnp.full_like(param, input)

    def apply(param, state, input):
        state = jnp.roll(state, 1).at[0].set(input)
        output = param @ state
        return state, output

    return Leaf(apply, param=param, init=init)
```

The form of this contract is dictated by the needs of JAX. The separate static argument binding facilitates `jit`; a `grad` of its parameters is unambiguous. `apply` has exactly the transition shape that `scan` requires. The separate roles tell `vmap` what a transform may share or map.

Anything you can do with JAX, you can do within this contract. Composition and valid transforms preserve this contract, so their results remain Nodes and can be composed again.

## See it working

- [`test_train_loop.py`](examples/test_train_loop.py) trains one `nn.Linear` Node and shows an ordinary compiled training loop with host-side logging and early stopping.
- [`actuator.py`](examples/actuator/actuator.py) demonstrates a non-neural physics simulation. NodeJAX is designed to extend beyond neural networks.
- [`test_nn_vit.py`](examples/test_nn_vit.py) builds a small image classifier entirely from stock `nodejax.nn` blocks, then batches and trains it on handwritten digits.
- [`test_kv_cache.py`](examples/test_kv_cache.py) treats a decoder cache as ordinary state, including branched decoding and batched users with shared params.
- [`test_maml_composed.py`](examples/test_maml_composed.py) expresses MAML by placing an adapting trainer inside an outer trainer.
- [`test_gan.py`](examples/test_gan.py) represents both sides of a GAN and their optimizers inside one differentiable program.
- [`test_ppo_pendulum.py`](examples/rl/test_ppo_pendulum.py) trains a recurrent PPO policy and replays each chunk from the policy state recorded during collection.
- [`test_pendulum_shac.py`](examples/rl/test_pendulum_shac.py) composes short-horizon actor-critic training over feed-forward and recurrent policy ensembles.
- [`examples/comparisons/`](examples/comparisons/) contains runnable NodeJAX, JAX, Equinox, Flax, Haiku, and PyTorch formulations of the same problems.

The complete gallery lives in [`examples/`](examples/).

## Read next

- The [cookbook](docs/cookbook.md) builds from a minimal Node to state, composition, transforms, and training.
- The [handbook](docs/handbook.md) is the technical reference for the contract, binding stages, authoring system, and transforms.
- The [design](docs/design.md) describes the implementation architecture and its Node-authoring and transform-authoring interfaces.
- The [design philosophy](docs/philosophy.md) explains why Nodes, explicit lifetimes, value semantics, and transform closure belong together.
- The [framework comparison](docs/comparison.md) examines ecosystem tradeoffs through side-by-side implementations.

## Status

NodeJAX is under active development and its API is evolving. The test suite and example programs are the executable specification. It is licensed under the [MIT License](LICENSE).

## Install

NodeJAX supports Python 3.11 or newer. To install from a source checkout:

```console
git clone https://github.com/EelcoHoogendoorn/nodejax.git
cd nodejax
pip install -e .
```
