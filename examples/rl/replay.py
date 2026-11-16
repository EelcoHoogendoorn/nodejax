"""Implements a cyclic replay buffer as a stateful Node.

Stores a fixed-capacity ring buffer of row pytrees alongside a write cursor and
fill count. Each `apply` call writes a segment of rows at the cursor and wraps
around at capacity, while `sample` draws a uniform minibatch from the filled
rows within the same carried state as optimizers and target networks.
"""

import jax
import jax.numpy as jnp

from nodejax import Leaf, PNode, PyTree, Struct, node, tree_len


@node
def Buffer(capacity: int) -> PNode:
    """Maintains a cyclic replay store of `capacity` rows shaped like one row of its input segment.

    Inserts a leading-axis row segment on `apply`, overwriting the oldest rows once
    full, and gathers `count` filled rows uniformly with replacement via `sample`.
    """

    def init(node):
        return Struct(
            store=jax.tree.map(
                lambda row: jnp.zeros((capacity,) + row.shape[1:], row.dtype),
                node.input,
            ),
            cursor=jnp.asarray(0),
            fill=jnp.asarray(0),
        )

    def apply(state, input):
        length = tree_len(input)
        positions = (state.cursor + jnp.arange(length)) % capacity
        store = jax.tree.map(
            lambda held, segment: held.at[positions].set(segment),
            state.store,
            input,
        )
        next_state = Struct(
            store=store,
            cursor=(state.cursor + length) % capacity,
            fill=jnp.minimum(state.fill + length, capacity),
        )
        return next_state, next_state.fill

    def sample(state, rng, count: int) -> PyTree:
        indices = jax.random.randint(rng.next(), (count,), 0, state.fill)
        return jax.tree.map(lambda held: held[indices], state.store)

    return Leaf(apply, init=init, methods={'sample': sample})