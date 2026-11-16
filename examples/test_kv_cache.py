"""Key-Value (KV) cache as ordinary cyclic state.

`KVCache` stores preallocated key and value ring buffers plus a scalar write
index in cyclic `state`. `Decoder` pairs the same four linear projections as
`Attention` with a parameterless `KVCache` member, leaving their sparse `param`
trees congruent so trained attention weights bind onto the decoder directly.
Prefilling scans prompt tokens through `Decoder`, and subsequent single-token
calls continue from the resulting `PSNode` value.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Composite, Leaf, Node, PNode, batch, derive, nn, node, scan, train_step,
    trained,
)
from nodejax.struct import Struct
from examples.util import mse
D, MAX_LEN = 8, 16


@node
def KVCache(max_len: int, dim: int) -> PNode:
    """Maintains `(max_len, dim)` key-value ring buffers and a write index in cyclic state, exposing a validity `mask` method."""
    def init():
        return Struct(
            k=jnp.zeros((max_len, dim)),
            v=jnp.zeros((max_len, dim)),
            idx=0,
        )

    def apply(state, input):
        k_step, v_step = input
        write_pos = state.idx % max_len
        k = state.k.at[write_pos].set(k_step)
        v = state.v.at[write_pos].set(v_step)
        next_state = Struct(k=k, v=v, idx=state.idx + 1)
        return next_state, Struct(k=k, v=v)

    def mask(state):
        num_valid = jnp.minimum(state.idx, max_len)
        return jnp.arange(max_len) < num_valid

    return Leaf(apply, init=init, methods={'mask': mask})


@node(name='attn')
def Attention(d: int, max_len: int) -> Node:
    """Computes full-sequence causal self-attention across four linear projections."""
    members = Composite(q=nn.Linear(d), k=nn.Linear(d), v=nn.Linear(d), o=nn.Linear(d))

    def apply(self, input):
        q = self.q(input)
        k = self.k(input)
        v = self.v(input)
        scores = q @ k.T / jnp.sqrt(d)
        t = input.shape[0]
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))
        attn = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
        return self.o(attn @ v)

    return members(apply)


@node(name='decode')
def Decoder(d: int, max_len: int) -> Node:
    """Computes single-token causal attention against a stateful `KVCache` member."""
    members = Composite(
        q=nn.Linear(d),
        k=nn.Linear(d),
        v=nn.Linear(d),
        o=nn.Linear(d),
        cache=KVCache(max_len, d),
    )

    def apply(self, input):
        q = self.q(input)
        cache = self.cache((self.k(input), self.v(input)))
        scores = cache.k @ q / jnp.sqrt(d)
        valid = self.cache.mask()
        attn = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf))
        return self.o(attn @ cache.v)

    return members(apply)


def test_decode_equals_full_attention_and_prefill_is_decode() -> None:
    """Verifies that scanning a prompt prefix and continuing token-by-token matches full causal attention."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(rng=jax.random.PRNGKey(0))
    dec = Decoder(D, MAX_LEN).bind(model.param)

    xs = jax.random.normal(jax.random.PRNGKey(1), (10, D))
    full = model.apply(xs)
    session = dec.initialize()
    # Prefilling scans the first 8 tokens and leaves the populated cache in the returned `PSNode`.
    session, ys = session.scan(xs[:8])
    assert jnp.allclose(ys, full[:8], atol=1e-5)

    session, y8 = session(xs[8])
    _, y9 = session(xs[9])
    assert jnp.allclose(y8, full[8], atol=1e-5)
    assert jnp.allclose(y9, full[9], atol=1e-5)


def test_caches_batch_like_any_state() -> None:
    """Verifies that `batch` allocates an independent cache row per sequence while broadcasting shared weights."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((6, D))).parameterize(rng=jax.random.PRNGKey(0))
    # Sparse `param` omits parameterless `cache`, so `model.param` binds onto `batch(Decoder(...))` directly.
    bdec = batch(Decoder(D, MAX_LEN), n=4).bind(model.param)

    xs = jax.random.normal(jax.random.PRNGKey(2), (6, 4, D))
    _, ys = bdec.initialize().scan(xs)
    full = jax.vmap(model.apply, in_axes=1, out_axes=1)(xs)
    assert jnp.allclose(ys, full, atol=1e-5)


def test_training_never_meets_the_cache() -> None:
    """Verifies that `Decoder` bound to trained `Attention` weights matches the trained full-sequence model."""
    full = Attention(D, MAX_LEN)
    model = full.with_input(jnp.zeros((50, D))).parameterize(rng=jax.random.PRNGKey(0))

    xs = jax.random.normal(jax.random.PRNGKey(3), (50, 10, D))
    target = jnp.roll(xs, -1, axis=1)
    trainer = train_step(model.initialize(), mse, optax.adam(1e-2))
    final, aux = trained(trainer).apply(input=xs, target=target)
    assert aux.loss[-1] < aux.loss[0]

    dec = Decoder(D, MAX_LEN).bind(final.param)
    seq = jax.random.normal(jax.random.PRNGKey(4), (10, D))
    _, ys = dec.initialize().scan(seq)
    assert jnp.allclose(ys, final.pnode.apply(seq), atol=1e-5)


def test_kv_cache_ring_buffer_wraps() -> None:
    """Verifies that `KVCache` wraps its write index modulo `max_len` when sequence length exceeds buffer capacity."""
    cache = KVCache(max_len=4, dim=D)
    inputs = jax.random.normal(jax.random.PRNGKey(5), (6, D))
    final, _ = cache.initialize().scan((inputs, inputs))
    assert final.state.idx == 6
    # Steps 4 and 5 overwrite buffer slots 0 and 1 modulo 4.
    assert jnp.allclose(final.state.k[0], inputs[4])
    assert jnp.allclose(final.state.k[1], inputs[5])


def test_a_decode_session_is_a_value() -> None:
    """Verifies that a state-bound `Decoder` branches without aliasing, resets to index zero, and exposes bound member methods."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(
        rng=jax.random.PRNGKey(0))
    dec = Decoder(D, MAX_LEN).bind(model.param)
    xs = jax.random.normal(jax.random.PRNGKey(6), (10, D))
    full = model.apply(xs)

    session = dec.initialize()
    outs = []
    for t in range(8):
        session, y = session(xs[t])
        outs.append(y)
    assert jnp.allclose(jnp.stack(outs), full[:8], atol=1e-5)

    # Aliasing a `PSNode` forks two independent continuations from the same cached prefix.
    fork = session
    session, y8 = session(xs[8])
    fork, z8 = fork(xs[9])
    assert jnp.allclose(y8, full[8], atol=1e-5)
    assert not jnp.allclose(y8, z8)
    assert session.state.cache.idx == fork.state.cache.idx == 9

    fresh = session.reset()
    assert fresh.state.cache.idx == 0

    # Member attribute access on a `PSNode` returns a state-bound view whose `mask` method reads live state directly.
    cache = session.cache
    assert cache.state.idx == 9
    assert int(cache.mask().sum()) == 9


def test_a_fleet_is_a_batched_session() -> None:
    """Verifies that `batch(dec, n=2)` shares one parameter copy across two independent cache rows and supports per-row state replacement."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(
        rng=jax.random.PRNGKey(0))
    dec = Decoder(D, MAX_LEN).bind(model.param)
    xs = jax.random.normal(jax.random.PRNGKey(7), (10, D))

    fleet = batch(dec, n=2).initialize()
    assert fleet.param.q.w.shape == (D, D)
    assert fleet.state.cache.idx.shape == (2,)

    for t in range(3):
        fleet, _ = fleet(jnp.stack([xs[t], xs[t]]))
    # Replacing row 1 of the batched `state` tree admits a fresh stream without disturbing row 0.
    fresh = dec.initialize()
    fleet = fleet.pnode.bind(state=jax.tree.map(
        lambda rows, one: rows.at[1].set(one), fleet.state, fresh.state))
    fleet, ys = fleet(jnp.stack([xs[3], xs[0]]))
    assert fleet.state.cache.idx[0] == 4
    assert fleet.state.cache.idx[1] == 1
    assert ys.shape == (2, D)
