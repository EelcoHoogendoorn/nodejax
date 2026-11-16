"""Train and evaluation mode switching in Flax NNX.

`model.train()` and `model.eval()` walk the NNX module graph and mutate `deterministic` and `use_running_average` attributes on mode-aware submodules in place. Stochastic machinery and mutable statistic variables remain on the shared instance behind flag discipline.

Run directly with `python -m examples.comparisons.mode.mode_nnx`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from examples.comparisons.mode.mode_common import (
    CLASSES, DIM, HIDDEN, LR, MOMENTUM, RATE, STEPS, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Net(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.lin = nnx.Linear(DIM, HIDDEN, rngs=rngs)
        self.drop = nnx.Dropout(RATE, rngs=rngs)
        self.bn = nnx.BatchNorm(HIDDEN, momentum=1 - MOMENTUM, rngs=rngs)
        self.head = nnx.Linear(HIDDEN, CLASSES, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(self.bn(self.drop(jax.nn.gelu(self.lin(x)))))


def main() -> None:
    xs_np, ys_np = make_data(np.random.RandomState(0))
    xs, ys = jnp.asarray(xs_np), jnp.asarray(ys_np)
    model = Net(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    model.train()

    @nnx.jit
    def step(m: Net, opt: nnx.Optimizer) -> jax.Array:
        loss, grads = nnx.value_and_grad(lambda mod: xent(mod(xs), ys))(m)
        opt.update(m, grads)
        return loss

    losses = [float(step(model, optimizer)) for _ in range(STEPS)]

    logits_a = model(xs)
    logits_b = model(xs)
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    model.eval()
    eval_a = model(xs)
    eval_b = model(xs)
    solo = model(xs[:16])

    report(
        'nnx',
        train_stochastic,
        bool(jnp.allclose(eval_a, eval_b)),
        bool(jnp.allclose(solo, eval_a[:16], atol=1e-6)),
        losses[0],
        losses[-1],
    )


if __name__ == '__main__':
    main()
