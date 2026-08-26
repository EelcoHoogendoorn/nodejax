"""Key-Value (KV) cache as ordinary cyclic state.

The KV cache is represented as state (preallocated buffers plus an index counter).
During autoregressive decoding, tokens update the cache step-by-step in `state`.
The decode definition derives from full causal attention (sharing params),
and prefilling is simply scanning tokens through the same decode apply function.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, node, scan, trained, Leaf, derive, batch, train_step, Composite, nn
from nodejax.struct import Struct
from examples.util import mse
D, MAX_LEN = 8, 16


@node
def KVCache(max_len: int, dim: int) -> Node:
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

    return Leaf(apply, init=init, methods={'mask': mask})


@node(name='attn')
def Attention(d, max_len: int) -> Node:
    """Full-sequence training attention using 4 linear sub-nodes."""
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
def Decoder(d, max_len: int) -> Node:
    """Token decode attention using 4 linear sub-nodes + KVCache."""
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


def test_decode_equals_full_attention_and_prefill_is_decode():
    """The cached decode IS the attention: token-by-token equals the
    full pass, and 'prefill' is nothing but the same scan stopped
    early — the state left behind is the filled cache."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(rng=jax.random.PRNGKey(0))
    dec = Decoder(D, MAX_LEN).bind(model.param)   # congruent trees: bind whole

    xs = jax.random.normal(jax.random.PRNGKey(1), (10, D))
    full = model.apply(xs)
    session = dec.initialize()
    session, ys = session.scan(xs[:8])                # prefill: 8 tokens
    assert jnp.allclose(ys, full[:8], atol=1e-5)

    session, y8 = session(xs[8])                      # continue decoding
    _, y9 = session(xs[9])
    assert jnp.allclose(y8, full[8], atol=1e-5)
    assert jnp.allclose(y9, full[9], atol=1e-5)


def test_caches_batch_like_any_state():
    """batch() maps state per element, so every sequence gets its own
    cache — no cache collection, no routing annotation."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((6, D))).parameterize(rng=jax.random.PRNGKey(0))
    # the param trees are congruent (sparse: the cache owns no slot), so
    # the trained attention binds the decoder WHOLE
    bdec = batch(Decoder(D, MAX_LEN), n=4).bind(model.param)

    xs = jax.random.normal(jax.random.PRNGKey(2), (6, 4, D))   # (T, B, d)
    _, ys = bdec.initialize().scan(xs)
    full = jax.vmap(model.apply, in_axes=1, out_axes=1)(xs)
    assert jnp.allclose(ys, full, atol=1e-5)


def test_training_never_meets_the_cache():
    """Train the full attention; the derived decode follows the params and
    still agrees — no decode flag, no mutable collections, and the
    optimizer never saw a cache because state is not params."""
    full = Attention(D, MAX_LEN)
    model = full.with_input(jnp.zeros((50, D))).parameterize(rng=jax.random.PRNGKey(0))

    xs = jax.random.normal(jax.random.PRNGKey(3), (50, 10, D))
    target = jnp.roll(xs, -1, axis=1)                 # a next-step-ish objective
    trainer = train_step(model.initialize(), mse, optax.adam(1e-2))
    final, aux = trained(trainer).apply(input=xs, target=target)
    assert aux.loss[-1] < aux.loss[0]

    dec = Decoder(D, MAX_LEN).bind(final.param)   # congruent trees: bind whole
    seq = jax.random.normal(jax.random.PRNGKey(4), (10, D))
    _, ys = dec.initialize().scan(seq)
    assert jnp.allclose(ys, final.pnode.apply(seq), atol=1e-5)


def test_kv_cache_ring_buffer_wraps():
    """KVCache wraps write position modulo max_len for sequences longer than buffer."""
    cache = KVCache(max_len=4, dim=D)
    inputs = jax.random.normal(jax.random.PRNGKey(5), (6, D))  # 6 steps > 4 max_len
    final, _ = cache.initialize().scan((inputs, inputs))
    assert final.state.idx == 6
    # Write positions 0 and 1 got overwritten by steps 4 and 5 (modulo 4)
    assert jnp.allclose(final.state.k[0], inputs[4])
    assert jnp.allclose(final.state.k[1], inputs[5])


def test_a_decode_session_is_a_value():
    """The session story: the loop lives OUTSIDE jax (a server, a
    notebook), so the cache rides the object and each token advances a
    successor, session, y = session(x). Sessions are values: forking one for a
    second continuation is nothing (the pytree copies), and the fork
    diverges without the original noticing — the beam-search shape that
    mutable caches make into a cloning ceremony."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(
        rng=jax.random.PRNGKey(0))
    dec = Decoder(D, MAX_LEN).bind(model.param)   # congruent trees: bind whole
    xs = jax.random.normal(jax.random.PRNGKey(6), (10, D))
    full = model.apply(xs)

    session = dec.initialize()
    outs = []
    for t in range(8):                       # the harness owns this loop
        session, y = session(xs[t])
        outs.append(y)
    assert jnp.allclose(jnp.stack(outs), full[:8], atol=1e-5)

    fork = session                           # a fork is a binding, nothing more
    session, y8 = session(xs[8])
    fork, z8 = fork(xs[9])                   # the fork continues differently
    assert jnp.allclose(y8, full[8], atol=1e-5)
    assert not jnp.allclose(y8, z8)
    assert session.state.cache.idx == fork.state.cache.idx == 9

    fresh = session.reset()                  # start over on the same bindings
    assert fresh.state.cache.idx == 0

    # the slice door at the state rung: the cache member is a session of
    # its own, and its node's METHODS read the live state it carries, so
    # mask() takes no arguments here where the node-level spelling would
    # thread the state by hand
    cache = session.cache
    assert cache.state.idx == 9
    assert int(cache.mask().sum()) == 9      # nine positions valid, live


def test_a_fleet_is_a_batched_session():
    """Concurrent users are not a new transform: shared params, unique
    state, unique inputs IS batch's axes signature. The fleet is the
    batched decoder staterized, one param copy and a cache row per
    session; joining a user mid-flight is row surgery on the state,
    because sessions are values all the way down."""
    model = Attention(D, MAX_LEN).with_input(jnp.zeros((10, D))).parameterize(
        rng=jax.random.PRNGKey(0))
    dec = Decoder(D, MAX_LEN).bind(model.param)   # congruent trees: bind whole
    xs = jax.random.normal(jax.random.PRNGKey(7), (10, D))

    fleet = batch(dec, n=2).initialize()
    assert fleet.param.q.w.shape == (D, D)        # ONE param copy, broadcast
    assert fleet.state.cache.idx.shape == (2,)    # a clock per session

    for t in range(3):                            # both users decode in step
        fleet, _ = fleet(jnp.stack([xs[t], xs[t]]))
    # user 1 leaves; a fresh session joins as a WRITTEN ROW of the state
    fresh = dec.initialize()
    fleet = fleet.pnode.bind(state=jax.tree.map(
        lambda rows, one: rows.at[1].set(one), fleet.state, fresh.state))
    fleet, ys = fleet(jnp.stack([xs[3], xs[0]]))
    assert fleet.state.cache.idx[0] == 4          # each user their own clock
    assert fleet.state.cache.idx[1] == 1
    assert ys.shape == (2, D)
