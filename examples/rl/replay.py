"""A cyclic replay buffer as an ordinary stateful Node.

The buffer's state is a tiling of the row pytree it is fed, a write
cursor, and a fill count. Insertion is the apply: one call writes a whole
segment of rows at the cursor and wraps around at capacity. ``sample`` is a
method drawing a uniform minibatch from the filled rows. Because the buffer
is ordinary Node state, it shares one carry with the optimizers and targets
of its program, and one jitted apply owns collection, storage, and updates
alike.
"""

import jax
import jax.numpy as jnp

from nodejax import Leaf, Node, PyTree, Struct, node, tree_len


@node
def Buffer(capacity: int) -> Node:
    """A cyclic store of ``capacity`` rows shaped like one row of what is
    inserted, which the declared input spec says.

    Apply inserts a segment whose leading axis counts rows and overwrites
    the oldest rows once full. ``sample(count, rng=...)`` gathers ``count``
    filled rows uniformly with replacement. Inside one enclosing apply, a
    sample after an insert sees the inserted rows.
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