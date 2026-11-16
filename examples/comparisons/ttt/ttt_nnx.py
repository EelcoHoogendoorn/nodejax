"""Reusable test-time training wrapper in Flax NNX across stateless and recurrent modules.

NNX separates unary `module(x) -> y` modules from `RNNCellBase`-style `(carry, x) -> (carry, y)` modules. Supporting both inside one `TTT` wrapper requires runtime protocol inspection (`hasattr(model, 'initialize_carry')`), cloning `self.model` per sequence so `nnx.scan` can mutate fast weights without corrupting meta-parameters, subclassing `nnx.Param` (`Rate`) so outer and inner parameters stay distinguishable, and shifting `(prev, target)` token pairs outside the module graph or as explicit `nnx.scan` carry state.

Run directly with `python -m examples.comparisons.ttt.ttt_nnx`.
"""

from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from examples.comparisons.ttt.ttt_common import (
    EVAL_TASKS, HIDDEN, META_LR, META_STEPS, QUERY0, TASKS, TTT_LR0, VOCAB,
    make_tasks, query_nll_numpy, report,
)

PyTree = object


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -jax.nn.log_softmax(logits)[target]


def query_xent(preds: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(preds)
    picked = jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[..., QUERY0:])


class Rate(nnx.Param):
    """Tags per-parameter inner learning rates outside the wrapped module."""


def _is_var(x: object) -> bool:
    return isinstance(x, nnx.Variable)


def _strip(tree: PyTree) -> PyTree:
    return jax.tree.map(lambda v: v[...] if _is_var(v) else v, tree, is_leaf=_is_var)


class MLP(nnx.Module):
    """Maps each input token to next-token logits without carried state."""

    def __init__(self, vocab: int, hidden: int, rngs: nnx.Rngs):
        self.embed = nnx.Param(0.3 * jax.random.normal(rngs.params(), (vocab, hidden)))
        self.w = nnx.Param(
            0.3 * jax.random.normal(rngs.params(), (hidden, hidden)) / jnp.sqrt(hidden)
        )
        self.out = nnx.Param(0.1 * jax.random.normal(rngs.params(), (hidden, vocab)))

    def __call__(self, token: jax.Array) -> jax.Array:
        return jnp.tanh(self.embed[...][token] @ self.w[...]) @ self.out[...]


class RNN(nnx.Module):
    """Updates a recurrent hidden vector and predicts next-token logits per step."""

    def __init__(self, vocab: int, hidden: int, rngs: nnx.Rngs):
        self.embed = nnx.Param(0.3 * jax.random.normal(rngs.params(), (vocab, hidden)))
        self.wh = nnx.Param(
            0.3 * jax.random.normal(rngs.params(), (hidden, hidden)) / jnp.sqrt(hidden)
        )
        self.out = nnx.Param(0.1 * jax.random.normal(rngs.params(), (hidden, vocab)))

    def initialize_carry(self) -> jax.Array:
        return jnp.zeros(self.wh[...].shape[0])

    def __call__(self, carry: jax.Array, token: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(self.embed[...][token] + self.wh[...] @ carry)
        return h, h @ self.out[...]


class TTT(nnx.Module):
    """Adapts unary or carry-based NNX modules online by per-token gradient descent."""

    def __init__(self, model: nnx.Module, loss_fn: Callable, lr0: float):
        self.model = model
        self.loss_fn = loss_fn
        self.has_carry = hasattr(model, 'initialize_carry')
        params = nnx.state(model, nnx.Param)
        self.lr = nnx.data(
            jax.tree.map(lambda x: Rate(jnp.full_like(x, lr0)), _strip(params))
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        fast = nnx.clone(self.model)
        lr = _strip(self.lr)
        carry0 = fast.initialize_carry() if self.has_carry else ()
        fast_axes = nnx.StateAxes({...: nnx.Carry})

        @nnx.scan(in_axes=(fast_axes, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step(adapted: nnx.Module, state, token: jax.Array):
            carry, prev = state

            def selfsup(m: nnx.Module):
                if self.has_carry:
                    next_carry, out = m(carry, prev)
                else:
                    next_carry, out = (), m(prev)
                return self.loss_fn(out, token), (next_carry, out)

            grads, (next_carry, out) = nnx.grad(selfsup, has_aux=True)(adapted)
            values = _strip(nnx.state(adapted, nnx.Param))
            next_values = jax.tree.map(
                lambda v, g, r: v - r * g, values, _strip(grads), lr,
            )
            nnx.update(adapted, next_values)
            return (next_carry, token), out

        _, preds = step(fast, (carry0, tokens[0]), tokens)
        return preds


class Batched(nnx.Module):
    def __init__(self, module: nnx.Module):
        self.module = module

    def __call__(self, batch: jax.Array) -> jax.Array:
        return nnx.vmap(lambda m, s: m(s), in_axes=(None, 0), out_axes=0)(self.module, batch)


class MetaTrainer(nnx.Module):
    def __init__(self, model: nnx.Module, loss_fn, tx):
        self.model = model
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        self.loss_fn = loss_fn

    def __call__(self, batch: jax.Array) -> jax.Array:
        def objective(m: nnx.Module):
            return self.loss_fn(m(batch), batch)

        loss, grads = nnx.value_and_grad(objective)(self.model)
        self.optimizer.update(self.model, grads)
        return loss

    def fit(self, batches: jax.Array) -> jax.Array:
        @nnx.jit
        def loop_fn(trainer: MetaTrainer, seq: jax.Array):
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def body(tr: MetaTrainer, batch: jax.Array):
                return tr, tr(batch)

            _, losses = body(trainer, seq)
            return losses

        return loop_fn(self, batches)


def run_variant(variant: str, inner: nnx.Module, tokens: jax.Array,
                held_out: jax.Array) -> None:
    batched = Batched(TTT(inner, xent, TTT_LR0))
    preds0 = batched(held_out)
    trainer = MetaTrainer(batched, query_xent, optax.adam(META_LR))
    losses = trainer.fit(tokens)
    preds1 = batched(held_out)
    n_weights = sum(x.size for x in jax.tree.leaves(nnx.state(batched, nnx.Param)))
    report(
        'nnx', variant, n_weights, bool(jnp.all(jnp.isfinite(losses))),
        query_nll_numpy(preds0, held_out), query_nll_numpy(preds1, held_out),
    )


def main() -> None:
    tokens = jnp.asarray(
        make_tasks(np.random.RandomState(0), META_STEPS * TASKS).reshape(META_STEPS, TASKS, -1)
    )
    held_out = jnp.asarray(make_tasks(np.random.RandomState(99), EVAL_TASKS))
    run_variant('mlp', MLP(VOCAB, HIDDEN, nnx.Rngs(0)), tokens, held_out)
    run_variant('rnn', RNN(VOCAB, HIDDEN, nnx.Rngs(0)), tokens, held_out)


if __name__ == '__main__':
    main()
