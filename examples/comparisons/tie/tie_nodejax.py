"""Tied input/output embeddings in NodeJAX.

`tie(pipe, 'embed', 'unembed')` keeps one embedding array under `embed`, leaves `unembed`'s parameter slot empty in `lm.param`, and routes `embed`'s parameter value into `unembed` during `apply`. The optimizer sees one leaf and gradients from both uses accumulate by the chain rule.

Run directly with `python -m examples.comparisons.tie.tie_nodejax`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import KeyStream, Leaf, Node, Struct, nn, node, scanned, tie, tile, train_step, trained

from examples.comparisons.tie.tie_common import (
    DIM, LR, STEPS, VOCAB, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


@node(name='rnn')
def RNNCell(dim: int) -> Node:
    def param(rng: KeyStream) -> Struct:
        return Struct(wh=0.5 * jax.random.normal(rng.next(), (dim, dim)) / jnp.sqrt(dim))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.wh.shape[0])

    def apply(param: Struct, state: jax.Array,
              input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(input + param.wh @ state)
        return h, h

    return Leaf(apply, init=init, param=param)


def tied_lm() -> Node:
    pipe = nn.Embed(VOCAB, DIM) >> RNNCell(DIM) >> nn.Unembed(VOCAB, DIM)
    return scanned(tie(pipe, 'embed', 'unembed'))


def main() -> None:
    prev_np, cur_np = make_data(np.random.RandomState(0))
    prev, cur = jnp.asarray(prev_np), jnp.asarray(cur_np)

    lm = tied_lm().parameterize(rng=jax.random.PRNGKey(0))
    assert 'unembed' not in lm.param

    trainer = train_step(lm, xent, optax.adam(LR))
    final, aux = trained(trainer).apply(input=tile(prev, STEPS), target=tile(cur, STEPS))

    assert 'unembed' not in final.param
    tables = [leaf for leaf in jax.tree.leaves(final.param) if leaf.shape == (VOCAB, DIM)]
    report('nodejax', len(tables), 0.0, float(aux.loss[0]), float(aux.loss[-1]))


if __name__ == '__main__':
    main()
