# nodejax

[![tests](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml/badge.svg)](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml)

Functional objects in JAX.

JAX closes pure functions under composition and transformation. `nodejax` lifts that closure to computation-graph `Node`s, where each `Node` bundles an `apply` transition with immutable `param` and `state` PyTrees so acyclic maps and `cyclic` feedback transitions compose and transform under one algebra. Rather than another neural network library, `nodejax` defines a general object model for JAX computation graphs that serves equally as the foundation for neural architectures (`nodejax.nn`), physical plants, feedback controllers (`nodejax.control`), replay buffers, and gradient-based or evolutionary optimizers.

| | Functions (`jax`) | `Node`s (`nodejax`) |
| :--- | :--- | :--- |
| Compose | `lambda x: jnp.sum(jax.nn.sigmoid(x))` | `nn.RNN(64) >> nn.relu` |
| Transform | `jax.jit(jax.vmap(f))` | `batch(scanned(ensemble(stack(node))))` |

A committee of residual GRU stacks driving a simulated DC motor in closed loop and trained end to end through the physics rollout:

```python
core = nn.Linear(HIDDEN) >> stack(residual(nn.GRU(HIDDEN)), n=DEPTH) >> nn.Projection()
controller = ensemble(core, n=MEMBERS) >> reduce(jnp.mean)       # committee of residual GRUs
rollout = batch(scanned(closed_loop(controller >> Motor(DT))))   # closed loop with a DC motor

model = rollout.with_input(references[0]).parameterize(rng=jax.random.PRNGKey(0))
trainer = train_step(model, tracking_loss, optax.adam(0.02))     # optimizer is another Node
fitted, aux = split_aux(trained(trainer)(input=references, target=targets))
```

Recurrent hidden states, motor current and velocity, active weights, and Adam moments share one immutable state contract. Full runnable example: [`examples/test_motor_control.py`](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/test_motor_control.py).

## Highlights

* **[Actuated whip](https://github.com/EelcoHoogendoorn/nodejax/tree/main/examples/whip)**: two FOC electric actuators with quantized encoders, a shared battery pack, a two-link arm, and a PBD rope optimized via MPPI, PPO, and SHAC.
* **[Third-order meta-learning](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/third_order_learning.py)**: inner test-time training (`next_step_ttt`) inside support-set `finetune` inside outer `train_step`, differentiated to third order across `ensemble`, `tie`, and collective `BatchNorm`.
* **[RL as Node algebra](https://github.com/EelcoHoogendoorn/nodejax/tree/main/examples/rl)**: MPPI, SHAC, PPO, and off-policy SAC (with a cyclic replay `Buffer` `Node`) over interchangeable feed-forward and recurrent controllers.
* **[Position-based dynamics](https://github.com/EelcoHoogendoorn/nodejax/tree/main/examples/pbd)**: particle ropes and 2D rigid-body XPBD constraints solved via `cyclic` and `repeat`.
* **[FOC motor stack](https://github.com/EelcoHoogendoorn/nodejax/tree/main/examples/actuator)**: encoder quantization, Luenberger observer, thermal derating, `DQ`-frame PI current control, and battery droop in one multi-rate `Composite`.
* **Neural & generative**: [tied-embedding MoE language model](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/test_char_lm.py) · [branched KV-cache decoding](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/test_kv_cache.py) · [denoising diffusion](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/diffusion.py) · [single-Node GAN](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/test_gan.py) · [second-order MAML](https://github.com/EelcoHoogendoorn/nodejax/blob/main/examples/test_maml_composed.py).
* **[Framework comparisons](https://github.com/EelcoHoogendoorn/nodejax/tree/main/examples/comparisons)**: nine side-by-side tasks against Flax NNX, Flax Linen, Equinox, Haiku, PyTorch, and Keras.

## Documentation

* **[Cookbook](https://github.com/EelcoHoogendoorn/nodejax/blob/main/docs/cookbook.md)** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/nodejax/blob/main/docs/cookbook.ipynb): runnable tour from a single `Leaf` to pipelines, transforms, `Generic` blueprints, and closed-loop control.
* **[Authoring guide](https://github.com/EelcoHoogendoorn/nodejax/blob/main/docs/authoring.md)**: placing `Node` boundaries across physics, control, neural architectures, and nested training.
* **[Handbook](https://github.com/EelcoHoogendoorn/nodejax/blob/main/docs/handbook.md)**: complete API reference for views, authoring hooks, transforms, `nodejax.nn`, and `nodejax.control`.
* **[Philosophy](https://github.com/EelcoHoogendoorn/nodejax/blob/main/docs/philosophy.md)** · **[Functional objects in JAX (paper)](https://github.com/EelcoHoogendoorn/nodejax/blob/main/paper/functional_objects_in_jax.md)** · **[Architecture & design](https://github.com/EelcoHoogendoorn/nodejax/blob/main/docs/design.md)** · **[Framework comparison](https://github.com/EelcoHoogendoorn/nodejax/blob/main/docs/comparison.md)**

## Install

Requires Python 3.11+:

```sh
pip install nodejax
```

MIT License ([`LICENSE`](https://github.com/EelcoHoogendoorn/nodejax/blob/main/LICENSE)).
