"""Tied embeddings, the nodejax side of the sharing comparison.

The sharing is STRUCTURAL: tie(pipe, 'embed', 'unembed') stores one
table under the source member and leaves the alias slot EMPTY, the
copy inserted at the consumer's slot at apply time. There is no second
reference to lose: jit and tree_map cannot desynchronize what exists
once, gradients from both uses accumulate by the chain rule, and the
optimizer sees exactly one table because exactly one is in the tree.
Drift is not prevented here; it is unrepresentable.

Run directly:  python -m nodejax.examples.comparisons.tie.tie_nodejax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import Node, node, nn, Leaf, scanned, tie, train_step, trained, KeyStream
from nodejax.struct import Struct

from nodejax.examples.comparisons.tie.tie_common import (
    VOCAB, DIM, STEPS, LR, make_data, xent, report)


@node(name='rnn')
def RNNCell(dim: int) -> Node:
    """The recurrent middle: hidden state carrying between the tied ends."""
    def param(rng: KeyStream) -> Struct:
        return Struct(wh=0.5 * jax.random.normal(rng.next(), (dim, dim)) / jnp.sqrt(dim))

    def init(param: Struct) -> jax.Array:
        return jnp.zeros(param.wh.shape[0])

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(input + param.wh @ state)
        return h, h

    return Leaf(apply, init=init, param=param)


def tied_lm() -> Node:
    """The tree: embed in, a recurrent cell between, unembed out through
    the SAME table. The two views live in DIFFERENT members, and tie
    rewires the unembed slot to read the embed's across that boundary."""
    pipe = nn.Embed(VOCAB, DIM) >> RNNCell(DIM) >> nn.Unembed(VOCAB, DIM)
    return scanned(tie(pipe, 'embed', 'unembed'))


def main() -> None:
    prev, cur = make_data(np.random.RandomState(0))
    lm = tied_lm().parameterize(rng=jax.random.PRNGKey(0))
    assert 'unembed' not in lm.param             # the alias slot does not exist

    tile = lambda leaf: jnp.broadcast_to(leaf, (STEPS, *leaf.shape))
    trainer = train_step(lm, xent, optax.adam(LR))
    final, aux = trained(trainer).apply(input=tile(prev), target=tile(cur))

    assert 'unembed' not in final.param         # still one copy after training
    tables = [leaf for leaf in jax.tree.leaves(final.param)
              if leaf.shape == (VOCAB, DIM)]
    report('nodejax', len(tables), 0.0, float(aux.loss[0]), float(aux.loss[-1]))


if __name__ == '__main__':
    main()
