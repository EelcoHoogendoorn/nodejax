# nodejax

[![tests](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml/badge.svg)](https://github.com/EelcoHoogendoorn/nodejax/actions/workflows/test.yml)

```python
core       = up(HIDDEN) >> reconstruction >> ttt(rnn(HIDDEN), mse, 0.01) >> readout(HIDDEN)
controller = serial(norm=running_norm(), committee=ensemble(core, n=MEMBERS), mix=mean)
rollout    = scan(closed_loop(controller >> motor(DT)))
trainer    = train_step(batch(rollout), mse, optax.adam(0.02))

final, losses = jax.jit(trainer.scan)(trainer.init(model=params), reference_batches)
```
Code: [`nodejax/tests/test_motor_control.py`](nodejax/tests/test_motor_control.py).

```
pip install git+https://github.com/EelcoHoogendoorn/nodejax.git
```

Nodejax is purely functional middleware for JAX. The core object is a Node in the compute graph, and every capability above (`ttt`, `ensemble`, `scan`, `closed_loop`, `batch`, `train_step`) is a function from Node to Node, a Node transform. It serves neural network use cases ergonomically, and is designed to generalize to classical control, physical simulation and beyond; [`nodejax/examples/actuator/`](nodejax/examples/actuator/) is a nontrivial example.

Unpacking the above: A committee of recurrent controllers drives a simulated motor in closed loop. Each member carries a test-time-training core: at every control step it adapts its own weights by one reconstruction gradient step at the same time as evolving its recurrent hidden state, while the outer trainer backpropagates through the unrolled physics to learn every member's initial weights and per-weight adaptation rates, as one compiled scan.

Six kinds of state are involved: the motor's, each RNN's carry, the running statistics, the ttt weight updates, the loop's feedback register, and the training state itself. They compose without a line of glue, because all six fill the same slot of the same Node contract.

## The node

```python
def motor(dt):                                 # a physical plant
    def init(ndef):
        return jnp.zeros_like(ndef.input)      # state shaped by the incoming signal
    def apply(state, input):
        omega = state + dt * (KT * input - B * state)
        return omega, omega
    return node_def(apply, init=init, name='motor')

def linear(n_out):                             # a neural layer
    def param(ndef, rng):
        n_in = ndef.apply_input_spec.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (n_in, n_out)) / jnp.sqrt(n_in),
                      b=jnp.zeros(n_out))
    def apply(param, input):
        return input @ param.w + param.b
    return node_def(apply, param=param, name='linear')
```

Code: [`nodejax/tests/test_motor_control.py`](nodejax/tests/test_motor_control.py), [`nodejax/nn.py`](nodejax/nn.py).

Everything in the opening block is, underneath, of this form: up to three pure functions against one rigid contract. It is similar in feel to an equinox or torch Module, but crucially it is a purely functional representation, with distinct slots for static arguments, parameters, state, and input.

```
param : (param_input)           -> params
init  : (params, state_input)   -> state
apply : (params, state, input)  -> (state, output)
```

The contract and its container live in [`nodejax/core.py`](nodejax/core.py). Statics like `n_out` bind at construction, params bind at `parameterize`, state initializes with init and evolves across steps within a run, and input arrives fresh per call. State being first-class in the contract is what the compositionality and the transforms below rest on, and one of the key differentiators from other frameworks.

## Composition

Nodejax aims to make composition and transformations of nodes frictionless.

```python
net   = linear(64) >> gelu >> linear(10)
model = net.with_input(images).parameterize(rng=key)   # one key, split per member

model.param.linear.w.shape          # (784, 64): named trees, plain pytrees
model.apply(images)                 # logits
```
Code: [`nodejax/examples/test_digits_committee.py`](nodejax/examples/test_digits_committee.py)

`>>` chains nodes into a node. The pipe's params and state are trees named by member, each member sized from what its own upstream produces. Both trees are ordinary JAX pytrees: `jax.grad` with respect to a model is simply `jax.grad`, and reading or editing weights is tree access, with no module object in the way.

## Example Transforms

```python
batch(node)                   # vmap over data: params shared, state per element
ensemble(node, n=8)           # vmap over params: independent members, one call
stack(node, n=4)              # scan over depth: layer k feeds layer k+1, vectorized
scan(node)                    # internalize state: a stepper becomes a sequence fn
train_step(node, loss, opt)   # internalize optimization: params become state
finetune(node, loss, opt)     # adaptation as a differentiable function
tie(pipe, src, *aliases)      # parameter sharing as reparameterization
```

Each of these is a reusable standard library component, written in a few dozen lines, generic over any Node. Because a node separates statics, params, state and apply input, every transform knows which axes to act on without `in_axes` bookkeeping. Code: [`nodejax/transforms/`](nodejax/transforms/).

## The trainer is a composable node

```python
population = train_step(ensemble(mlp, n=8), mse, adam)   # 8 models, one program
maml = train_step(batch(finetune(model, loss, sgd(0.1))), loss, adam(1e-2))
targets = enc.apply(state.ema, v2)  # BYOL: an EMA node smoothing a weight subtree
```

`train_step` is itself a Node whose state holds the weights, the optimizer moments, and the model's own state. This makes it easy to implement concepts like meta-learning. Everything a framework would offer as a feature on its trainer object is, here, a transform applied to it or a tree operation on its state. Code: [`nodejax/examples/test_population.py`](nodejax/examples/test_population.py), [`nodejax/transforms/tests/test_finetune.py`](nodejax/transforms/tests/test_finetune.py), [`nodejax/examples/test_byol.py`](nodejax/examples/test_byol.py).

```python
gan = gan_def(adam(d_lr), adam(g_lr))     # one adversarial round; two trainers inside
meta = train_step(replay(gan, rounds=60), sample_quality, adam(0.2))
```

And it nests without limit: the adversarial round holds two trainers, and an outer trainer learns their learning rates through the replayed game. Code: [`nodejax/examples/test_gan.py`](nodejax/examples/test_gan.py).

## Examples

Each is a test; each runs.

- [`nodejax/tests/test_motor_control.py`](nodejax/tests/test_motor_control.py): the opening block; a test-time-training committee drives a motor, trained through the physics.
- [`nodejax/examples/test_gan.py`](nodejax/examples/test_gan.py): a GAN as one node; a population of games via `ensemble`; the learning rates meta-learned through the unrolled game.
- [`nodejax/examples/test_byol.py`](nodejax/examples/test_byol.py): BYOL, where the target network is an EMA filter applied to the encoder's weight subtree.
- [`nodejax/examples/meta_comparison.py`](nodejax/examples/meta_comparison.py): test-time training with a RECURRENT inner model, benchmarked against ttt variants and plain RNNs.
- [`nodejax/examples/test_meta_controller.py`](nodejax/examples/test_meta_controller.py): MAML over controllers: `train_step(batch(metasgd(task)))` adapts to plants unseen at training time.
- [`nodejax/examples/test_char_lm.py`](nodejax/examples/test_char_lm.py): a character LM with mixture-of-experts aux losses, tied embeddings, and a sampling loop as a scanned node.
- [`nodejax/examples/test_digits_committee.py`](nodejax/examples/test_digits_committee.py): deep residual recurrent committee over pixel rows; whitening stats live inside the stack, frozen-read at eval.
- [`nodejax/examples/test_transfer.py`](nodejax/examples/test_transfer.py): pretrain, freeze the trunk, swap the head: the frozen trunk comes out bitwise identical.
- [`nodejax/examples/test_population.py`](nodejax/examples/test_population.py): eight models trained as one program; the champion slices out as an ordinary model.
- [`nodejax/examples/test_train_loop.py`](nodejax/examples/test_train_loop.py): the mundane loop done right: jitted chunks, host-side stats, early stopping, bit-exact reproducibility.
- [`nodejax/tests/test_imu.py`](nodejax/tests/test_imu.py): a drifting, quantized accelerometer as a pipe of four physical components.
- [`nodejax/transforms/tests/test_tie.py`](nodejax/transforms/tests/test_tie.py): tied autoencoders and tied embeddings: one copy in the tree, so the two views cannot drift apart.

## Status

Working prototype: the tests double as documentation. [`docs/cookbook.md`](docs/cookbook.md) builds up from a minimal node, explaining as it goes (runnable in Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/nodejax/blob/main/docs/cookbook.ipynb)); [`docs/handbook.md`](docs/handbook.md) is the reference; [`docs/comparison.md`](docs/comparison.md) compares with equinox, flax, and haiku, with runnable counterparts in [`nodejax/examples/comparisons/`](nodejax/examples/comparisons/).