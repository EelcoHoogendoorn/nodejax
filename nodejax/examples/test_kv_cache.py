"""Key-Value (KV) cache as ordinary cyclic state.

The KV cache is represented as state (preallocated buffers plus an index counter).
During autoregressive decoding, tokens update the cache step-by-step in `state`.
The decode definition derives from full causal attention (sharing params),
and prefilling is simply scanning tokens through the same decode apply function.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import node_def, derive, batch, train_step, composite, nn
from nodejax.struct import Struct
from nodejax.util import mse
D, MAX_LEN = 8, 16


def KVCache(max_len, dim):
    """Standalone KV-cache node. State stores preallocated buffers + index pointer."""
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

    return node_def(apply, init=init, methods={'mask': mask}, name='kv_cache')


def Attention(d, max_len):
    """Full-sequence training attention using 4 linear sub-nodes."""
    members = dict(q=nn.Linear(d), k=nn.Linear(d), v=nn.Linear(d), o=nn.Linear(d))

    def apply(self, input):
        q = self.q(input)
        k = self.k(input)
        v = self.v(input)
        scores = q @ k.T / jnp.sqrt(d)
        t = input.shape[0]
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))
        attn = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
        return self.o(attn @ v)

    return composite(apply, members=members, name='attn')


def Decoder(d, max_len):
    """Token decode attention using 4 linear sub-nodes + KVCache."""
    members = dict(
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

    return composite(apply, members=members, name='decode')


def test_decode_equals_full_attention_and_prefill_is_decode():
    """The cached decode IS the attention: token-by-token equals the
    full pass, and 'prefill' is nothing but the same scan stopped
    early — the state left behind is the filled cache."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(rng=jax.random.PRNGKey(0))
    p = model.param
    dec = Decoder(D, MAX_LEN).bind(Struct(cache=(), q=p.q, k=p.k, v=p.v, o=p.o))

    xs = jax.random.normal(jax.random.PRNGKey(1), (10, D))
    full = model.apply(xs)
    cache, ys = dec.scan(dec.init(), xs[:8])          # prefill: 8 tokens
    assert jnp.allclose(ys, full[:8], atol=1e-5)

    cache, y8 = dec.apply(cache, xs[8])               # continue decoding
    _, y9 = dec.apply(cache, xs[9])
    assert jnp.allclose(y8, full[8], atol=1e-5)
    assert jnp.allclose(y9, full[9], atol=1e-5)


def test_caches_batch_like_any_state():
    """batch() maps state per element, so every sequence gets its own
    cache — no cache collection, no routing annotation."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((6, D))).parameterize(rng=jax.random.PRNGKey(0))
    p = model.param
    bdec = batch(Decoder(D, MAX_LEN), n=4).bind(Struct(cache=(), q=p.q, k=p.k, v=p.v, o=p.o))

    xs = jax.random.normal(jax.random.PRNGKey(2), (6, 4, D))   # (T, B, d)
    _, ys = bdec.scan(bdec.init(), xs)
    full = jax.vmap(model.apply, in_axes=1, out_axes=1)(xs)
    assert jnp.allclose(ys, full, atol=1e-5)


def test_training_never_meets_the_cache():
    """Train the full def; the derived decode follows the params and
    still agrees — no decode flag, no mutable collections, and the
    optimizer never saw a cache because state is not params."""
    full = Attention(D, MAX_LEN)
    model = full.with_input(jnp.zeros((50, D))).parameterize(rng=jax.random.PRNGKey(0))

    xs = jax.random.normal(jax.random.PRNGKey(3), (50, 10, D))
    target = jnp.roll(xs, -1, axis=1)                 # a next-step-ish objective
    trainer = train_step(full, mse, optax.adam(1e-2))
    final, losses = trainer.scan(trainer.init(model=model.param, input=xs),
                                 Struct(input=xs, target=target))
    assert losses[-1] < losses[0]

    trained = full.bind(final.model)
    p = final.model
    dec = Decoder(D, MAX_LEN).bind(Struct(cache=(), q=p.q, k=p.k, v=p.v, o=p.o))
    seq = jax.random.normal(jax.random.PRNGKey(4), (10, D))
    _, ys = dec.scan(dec.init(), seq)
    assert jnp.allclose(ys, trained.apply(seq), atol=1e-5)


def test_kv_cache_ring_buffer_wraps():
    """KVCache wraps write position modulo max_len for sequences longer than buffer."""
    cache = KVCache(max_len=4, dim=D)
    state = cache.init()
    inputs = jax.random.normal(jax.random.PRNGKey(5), (6, D))  # 6 steps > 4 max_len
    final_state, _ = cache.scan(state, (inputs, inputs))
    assert final_state.idx == 6
    # Write positions 0 and 1 got overwritten by steps 4 and 5 (modulo 4)
    assert jnp.allclose(final_state.k[0], inputs[4])
    assert jnp.allclose(final_state.k[1], inputs[5])
