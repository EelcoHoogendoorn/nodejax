# NodeJAX

[![tests](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml/badge.svg)](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml)

NodeJAX is a functional object layer for stateful differentiable programs in JAX.

JAX closes pure functions under composition and transformation (`jit`, `grad`, `vmap`, `lax.scan`). NodeJAX gives stateful program units (`Node`s) the same closure across neural layers, physical plants, controllers, caches, replay buffers, and optimizers.

| | JAX functions | NodeJAX Nodes |
| :--- | :--- | :--- |
| Closure under composition | `f = lambda x: sum(sigmoid(x))` | `node = nn.RNN(64) >> nn.relu` |
| Closure under transforms | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

## Example

```python
core = nn.Linear(HIDDEN) >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH) >> nn.Projection()
controller = ensemble(core, n=MEMBERS) >> reduce(jnp.mean)
rollout = batch(scanned(closed_loop(controller >> Motor(DT))))

model = rollout.with_input(references[0]).parameterize(rng=jax.random.PRNGKey(0))
trainer = train_step(model, tracking_loss, optax.adam(0.02))
fitted, aux = split_aux(trained(trainer)(input=references, target=targets))
```

A committee of residual GRU stacks drives a simulated DC motor in closed loop (`examples/test_motor_control.py`). Recurrent hidden states, motor current and angular velocity, active weights, and Adam moments share one immutable state contract and differentiate end to end through the physics rollout.

## The Node contract

An authored `@node` declares up to three pure functions (`param`, `init`, and `apply`) aligned with JAX's transformation boundaries:

```python
@node
def WindowedFilter(window_size: int) -> Node:
    def param(rng) -> jax.Array:
        return jax.random.normal(rng.next(), (window_size,)) / jnp.sqrt(window_size)

    def init(param: jax.Array, input: jax.Array) -> jax.Array:
        return jnp.full_like(param, input)

    def apply(param: jax.Array, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_state = jnp.roll(state, 1).at[0].set(input)
        return next_state, param @ next_state

    return Leaf(apply, param=param, init=init)
```

`window_size` is a compile-time static for `jit` (and stays open in a `Generic` blueprint when omitted). `param` isolates differentiable parameters for `grad`. `apply` has the `(carry, input) -> (next_carry, output)` transition signature expected by `lax.scan`. Separate `param`, `state`, and `input` trees tell `vmap` transforms (`batch`, `ensemble`, `stack`) which axes to share or map, while `rng.next()` draws from a scope-local `KeyStream` seeded by a single boundary key.

## Examples

- [`examples/test_train_loop.py`](examples/test_train_loop.py): linear regression stepped in chunks with host-side logging and early stopping.
- [`examples/test_nn_vit.py`](examples/test_nn_vit.py) and [`examples/test_char_lm.py`](examples/test_char_lm.py): Vision Transformer and tied-embedding Mixture-of-Experts character language model built from `nodejax.nn`.
- [`examples/test_kv_cache.py`](examples/test_kv_cache.py): autoregressive decoder KV cache modeled as ordinary `Node` state across branched and batched decoding.
- [`examples/diffusion.py`](examples/diffusion.py) and [`examples/test_gan.py`](examples/test_gan.py): iterative denoising diffusion and coupled generator-discriminator training inside a single `Node`.
- [`examples/test_maml_composed.py`](examples/test_maml_composed.py) and [`examples/third_order_learning.py`](examples/third_order_learning.py): second-order MAML and third-order test-time training composed from nested `train_step` Nodes.
- [`examples/actuator/`](examples/actuator/), [`examples/pbd/`](examples/pbd/), and [`examples/whip/`](examples/whip/): electromagnetic actuator control, position-based particle/rigid-body dynamics (`cyclic` constraint solvers), and actuated multi-link whip trajectory optimization.
- [`examples/rl/`](examples/rl/): composable feed-forward and recurrent MPPI, SHAC, PPO, and off-policy SAC (with a cyclic replay `Buffer` Node) over shared physical plants.
- [`examples/comparisons/`](examples/comparisons/): side-by-side implementations in NodeJAX, Flax NNX, Flax Linen, Equinox, Haiku, PyTorch, and Keras.

## Documentation

- [Cookbook](docs/cookbook.md) ([`cookbook.ipynb`](docs/cookbook.ipynb)): progressive runnable walkthrough from a minimal `Leaf` to pipelines, transforms, `Generic` templates, and closed-loop control.
- [Authoring guide](docs/authoring.md): where to place `Node` boundaries and how programs decompose across control, physics, neural architectures, nested training, and RL.
- [Handbook](docs/handbook.md): complete technical reference for views, authoring hooks, combinators, transforms, `nodejax.nn`, `nodejax.control`, and PyTree utilities.
- [Architecture & design](docs/design.md): internal `Def` lowering, canonical `Contract` roles, and T3 `@transform` authoring.
- [Philosophy](docs/philosophy.md): why JAX transformations lead to single-`apply` Functional Object-Oriented Programming (FOOP) and unified state contracts.
- [Framework comparison](docs/comparison.md): structural evaluation against Flax, Equinox, Haiku, PyTorch, Keras, and Brax.

## Installation

NodeJAX requires Python 3.11 or newer:

```console
git clone https://github.com/EelcoHoogendoorn/nodejax.git
cd nodejax
pip install -e .
```

NodeJAX is licensed under the [MIT License](LICENSE).
