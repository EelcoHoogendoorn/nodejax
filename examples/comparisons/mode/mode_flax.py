"""The mode switch, the flax nnx side of the comparison.

Self-contained: no nodejax import.

nnx spells the switch as torch does, model.train() and model.eval(),
implemented as a graph walk that rewrites flag attributes on every
mode-aware submodule: Dropout.deterministic and
BatchNorm.use_running_average flip together. The bit lives on the
objects and every forward trusts whoever flipped it last, torch's
discipline cost with a functional substrate underneath: the flags are
part of the module graph, so graph-aware transforms preserve them, and the
dropout's entropy is RngState updated per call while training.

flax is a reference-exhibit dependency: the file runs in any
environment with flax installed.

Run directly:  python -m examples.comparisons.mode.mode_flax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

DIM, HIDDEN, CLASSES = 8, 16, 3
N, STEPS = 128, 200
RATE, MOMENTUM, LR = 0.3, 0.1, 0.02


def make_data(rs):
    centers = 2.0 * rs.standard_normal((CLASSES, DIM))
    labels = rs.randint(CLASSES, size=N)
    xs = centers[labels] + rs.standard_normal((N, DIM))
    return jnp.asarray(xs, jnp.float32), jnp.asarray(labels, jnp.int32)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Net(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.lin = nnx.Linear(DIM, HIDDEN, rngs=rngs)
        self.drop = nnx.Dropout(RATE, rngs=rngs)
        self.bn = nnx.BatchNorm(HIDDEN, momentum=1 - MOMENTUM, rngs=rngs)
        self.head = nnx.Linear(HIDDEN, CLASSES, rngs=rngs)

    def __call__(self, x):
        return self.head(self.bn(self.drop(jax.nn.gelu(self.lin(x)))))


def main() -> None:
    xs, ys = make_data(np.random.RandomState(0))
    model = Net(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    model.train()                      # the graph walk, flags flipped on

    @nnx.jit
    def step(model, optimizer):
        loss, grads = nnx.value_and_grad(lambda m: xent(m(xs), ys))(model)
        optimizer.update(model, grads)
        return loss

    losses = [float(step(model, optimizer)) for _ in range(STEPS)]

    logits_a = model(xs)               # still train mode: dropout draws
    logits_b = model(xs)
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    # THE MODE SWITCH: the walk again, flags flipped off on every
    # mode-aware submodule
    model.eval()
    eval_a = model(xs)
    eval_b = model(xs)
    solo = model(xs[:16])

    print(f'{"flax nnx":10s} train_stochastic={train_stochastic} '
          f'eval_deterministic={bool(jnp.allclose(eval_a, eval_b))} '
          f'eval_isolated={bool(jnp.allclose(solo, eval_a[:16], atol=1e-6))} '
          f'loss {losses[0]:.2f} -> {losses[-1]:.2f}', flush=True)


if __name__ == '__main__':
    main()
