"""Shared task and data for the tie comparison: WEIGHT SHARING as a
property you can lose.

One embedding table serves two roles in a next-token model: it embeds
the previous token on the way in, and unembeds the hidden state on the
way out. The two roles are ONE object by definition, and each column
measures whether its framework can say so: how the sharing is declared,
how many copies of the table the optimizer sees, and whether the two
views can drift apart under training. The task is next-token prediction
on one Markov chain; scores are sanity checks.

THE MODEL IS BUILT FROM COMPONENTS: an embed block, a recurrent cell,
an unembed head, each its own module, because that is where sharing
lives in practice. The two views belong to DIFFERENT components, so the
tie must cross a composition boundary; a single class holding both
attributes would trivialize exactly the thing being measured.

The frameworks answer differently, and the differences are the point.
nodejax's tie makes sharing STRUCTURAL: one copy in the param tree, the
alias slot empty, inserted at apply time, so drift is unrepresentable.
flax nnx and torch share by OBJECT REFERENCE, which their graph and
registration machinery deduplicate; the sharing is real but lives in
identity, not in the tree. haiku shares by PARAMETER PATH, the two
views reaching one dict entry under one module name, a convention
nothing checks. equinox modules are pytrees, and an aliased array
DUPLICATES on flatten: the optimizer sees two tables and the views
drift, which is the failure the nnx documentation names as its reason
modules must not be pytrees; eqx.nn.Shared is equinox's repair.

The rival files carry their own line-identical copy of the generator on
purpose: each stays runnable as one self-contained file in an
environment without this package.
"""

import numpy as np
import jax
import jax.numpy as jnp

VOCAB, DIM = 8, 6
POSITIONS, STEPS = 256, 200
LR = 0.03
CONCENTRATION = 2.0


def make_data(rs: np.random.RandomState):
    """One Markov chain, one sequence: previous-token inputs and
    current-token targets, flattened positions."""
    logits = CONCENTRATION * rs.standard_normal((VOCAB, VOCAB))
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)
    tokens = np.zeros(POSITIONS + 1, dtype=np.int64)
    state = rs.randint(VOCAB)
    for t in range(POSITIONS + 1):
        tokens[t] = state
        state = (P[state].cumsum() > rs.random()).argmax()
    return jnp.asarray(tokens[:-1], jnp.int32), jnp.asarray(tokens[1:], jnp.int32)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


def report(name: str, table_copies: int, drift: float, first: float, last: float) -> None:
    print(f'{name:14s} table_copies={table_copies} drift={drift:.2e} '
          f'loss {first:.2f} -> {last:.2f}', flush=True)
