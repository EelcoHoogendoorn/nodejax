"""Chunked sequences in haiku: ambient state, ambient entropy, one boundary.

Same task, same numbers, same references as `chunk_nodejax.py`. hk.scan over
chunks of hk.scan over steps, so what the comparison shows is what the
spelling cost.

Haiku is the design the others are not: state and entropy are AMBIENT inside a
transformed function, reached by name rather than passed. `hk.get_state` and
`hk.next_rng_key` take no argument saying where they come from, and the
plumbing that makes them work is entirely in `hk.transform_with_state` and the
lifted `hk.scan`.

WHAT THAT BUYS, and it is the shortest answer in the table on both counts:

1. THE STATISTICS CROSS EVERYTHING BY SAYING NOTHING. There is no annotation
   at either scan, no collection named at any level, no path built by hand.
   hk.scan threads whatever the body touched.

2. THE DRAW NEEDS NO FLAG. `hk.next_rng_key()` inside an hk.scan gives a
   different key per iteration on its own. flax needs `split_rngs={'carry':
   True}` at the lift, three lines from the draw and joined to it by a string,
   and silently repeats the same carry if you forget.

WHAT IT COSTS, and it is one thing said three ways:

1. THE FUNCTION IS NOT A FUNCTION until it is transformed. `step_fn` cannot be
   called, tested, or composed on its own; outside a transform it raises. What
   the other four files hand around as ordinary values, this one can only
   reach inside a context.

2. WHICH STATE IS WHICH IS A STRING. `hk.get_state('mean')` is matched by
   name, under a module path haiku assigns. Nothing checks that the name at
   the read is the name at the write, and nothing relates either to the
   lifetime it is supposed to have.

Run directly:  python -m examples.comparisons.chunk.chunk_haiku
"""

from typing import Callable

import jax
import jax.numpy as jnp
import haiku as hk

from examples.comparisons.chunk import chunk_common as task


def norm(x: jax.Array):
    """Running per-feature normalizer. Its statistics are AMBIENT state,
    reached by name."""
    mean = hk.get_state('mean', (task.W,), init=jnp.zeros)
    var = hk.get_state('var', (task.W,), init=jnp.ones)
    out = task.normalize(x, mean, var)
    new_mean, new_var = task.update_stats(x, mean, var)
    hk.set_state('mean', new_mean)
    hk.set_state('var', new_var)
    return out


def rnn(hidden: int, x: jax.Array):
    """The recurrent cell, with WEIGHTS. Its hidden state is an ORDINARY
    VALUE, in and out, because hk.scan carries it. Two kinds of state in one
    model, and which kind a piece goes in is decided by whether the scan
    carries it.

    The weights are hk.get_parameter, ambient exactly as the statistics are
    and reached the same way, and NOTHING at either scan mentions them. That
    is the same trick as the statistics crossing by saying nothing, applied to
    the other channel: transform_with_state already returns the pair, and the
    lifted scan already threads whatever the body touched."""
    def drawn(i):
        return lambda shape, dtype: task.weights()[i]

    wx = hk.get_parameter('wx', (task.W, task.H), init=drawn(0))
    wh = hk.get_parameter('wh', (task.H, task.H), init=drawn(1))
    b = hk.get_parameter('b', (task.H,), init=drawn(2))
    return task.cell(hidden, x, wx, wh, b)


def step(hidden: int, x: jax.Array):
    """The two composed, the ordinary way: one function calling the other.

    Composing costs nothing visible here, which is haiku's real trick and its
    real cost at once. `norm` reaches its statistics by name under a path
    haiku assigns from the CALL STACK, so nesting it renames its state without
    a line changing: the slot moves because the caller moved."""
    h = rnn(hidden, norm(x))
    return h, h                                   # a scan body: carry, output


def _chunked(seq: jax.Array):
    """chunks of steps. The carry is drawn once, at the top."""
    def chunk(hidden, xs):
        return hk.scan(step, hidden, xs)

    hidden = task.carry_init(hk.next_rng_key())
    _, outs = hk.scan(chunk, hidden, seq.reshape(-1, task.CHUNK, task.W))
    return outs.reshape(-1, task.H)


def _recordings(seq: jax.Array):
    """recordings of chunks of steps, and the boundary is a line in a body.

    The recording boundary is the `carry_init` inside the recording body, and
    it draws a fresh key per recording because hk.next_rng_key inside an
    hk.scan does that without being asked. The statistics cross everything by
    saying nothing, which is ambient state's whole trick. Nothing declares
    the boundary, and nothing says which state it belongs to."""
    def chunk(hidden, xs):
        return hk.scan(step, hidden, xs)

    def recording(_, chunks):
        hidden = task.carry_init(hk.next_rng_key())      # the RECORDING boundary
        _, outs = hk.scan(chunk, hidden, chunks)
        return None, outs

    _, outs = hk.scan(recording, None, task.recordings(seq))
    return outs.reshape(-1, task.H)


def _run(fn: Callable, *args):
    """init to build the state, then apply. Both take a key, because entropy
    enters the transform and nothing inside it names one."""
    f = hk.transform_with_state(fn)
    params, state = f.init(task.INIT_KEY, *args)
    out, _ = f.apply(params, state, task.INIT_KEY, *args)
    return out


def run(seq: jax.Array=task.SEQ):
    return _run(_chunked, seq)


def run_recordings(seq: jax.Array=task.SEQ):
    return _run(_recordings, seq)


def train(rollout=_chunked, seq: jax.Array=task.SEQ):
    """FOURTH question: train the weights through the chunked rollout.

    Haiku's cleanest moment, and for the same reason as its awkward ones.
    transform_with_state hands back (params, state) as two separate trees, so
    differentiating with respect to the first cannot reach the second and
    nothing has to say which is which. The split that made the model a
    non-function is the split that makes this free.

    The optimizer state is a third thing the caller holds, OUTSIDE the
    transform. Ambient buys nothing here: hk.get_state is for what the model
    owns, and the optimizer is not the model's.

    FIFTH AND SIXTH: the same loop over a different rollout. The lifetimes are
    lines in bodies, so training them costs nothing extra here, which is the
    other side of them costing a line each to write in the first place."""
    f = hk.transform_with_state(rollout)
    params, state = f.init(task.INIT_KEY, seq)

    def loss_fn(params, state):
        outs, _ = f.apply(params, state, task.INIT_KEY, seq)
        return task.loss_of(outs)

    # the loop is a lax.scan, because that is what a user of a pure-functional
    # framework writes: the training state is a carry like any other, and a
    # Python loop over jax code is not something a real user would put up with
    def step(params, _):
        loss, grads = jax.value_and_grad(loss_fn)(params, state)
        # HAND-ROLLED SGD: these frameworks own no optimizer, optax being
        # the ecosystem's, and this loop owns its own arithmetic instead
        return jax.tree.map(lambda w, g: w - task.LR * g, params, grads), loss

    _, losses = jax.lax.scan(step, params, None, length=task.TRAIN_STEPS)
    return losses



def train_sessions():
    """FIFTH question: recording and session in one scan nest.

    The caller owns all three loops as lax.scans, and each lifetime is which
    scan's top its reset sits at: the cold state enters at the session scan,
    the cold hidden at the recording scan, and the params thread through
    everything. haiku contributes the chunk forward and nothing else; the
    training structure is plain jax, outside the transform."""
    def one_chunk(hidden, xs):
        return hk.scan(step, hidden, xs)

    f = hk.transform_with_state(one_chunk)
    hidden0 = task.carry_init(task.INIT_KEY)
    params0, state0 = f.init(task.INIT_KEY, hidden0, task.SESS_SEQ[0, 0, 0])

    def over_chunks(carry, xt):
        params, vel, state, hidden = carry
        x, t = xt

        def loss_fn(p):
            (h2, outs), s2 = f.apply(p, state, None, hidden, x)
            return jnp.mean((outs - t) ** 2), (h2, s2)

        (loss, (h2, s2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        # HAND-ROLLED MOMENTUM, same note: the velocity is the caller's
        # pytree because nothing in the framework owns the training loop
        vel = jax.tree.map(lambda v, g: task.MU * v + g, vel, grads)
        params = jax.tree.map(lambda w, v: w - task.LR * v, params, vel)
        return (params, vel, s2, h2), loss

    def over_recordings(carry, xt):
        params, vel, state = carry
        (params, vel, state, _), losses = jax.lax.scan(
            over_chunks, (params, vel, state, hidden0), xt)   # RECORDING: hidden dies
        return (params, vel, state), losses

    def over_sessions(params, xt):
        vel0 = jax.tree.map(jnp.zeros_like, params)   # SESSION: momentum restarts
        (params, _, _), losses = jax.lax.scan(
            over_recordings, (params, vel0, state0), xt)       # SESSION: recalibrated
        return params, losses

    _, losses = jax.lax.scan(over_sessions, params0,
                             (task.SESS_SEQ, task.SESS_TARGET))
    return losses.reshape(-1)


def main() -> None:
    task.report('haiku',
                live_ok=bool(jnp.allclose(run(), task.reference_live(), atol=1e-5)),
                two_ok=bool(jnp.allclose(run_recordings(),
                                         task.reference_recordings(), atol=1e-5)),
                train_ok=bool(jnp.allclose(train(), task.reference_trained(), atol=1e-4)),
                train_two_ok=bool(jnp.allclose(
                    train(_recordings), task.reference_trained_recordings(),
                    atol=1e-4)),
                train_tags_ok=bool(jnp.allclose(
                    train_sessions(), task.reference_trained_sessions(),
                    atol=1e-4)),
                cost={
                    'hidden across chunks:': 'the scan carry, nothing declared',
                    'stats across chunks:': 'nothing: ambient state, threaded by hk.scan',
                    'model edited:': 'no',
                    'slots named outside:': "none, but 'mean'/'var' are strings inside",
                    'carry re-inits per recording:': 'a line in the recording body',
                    'the carry is DRAWN:': 'hk.next_rng_key; no flag, no threading',
                    'the cell has WEIGHTS:': 'hk.get_parameter; scans unchanged',
                    'stats cross both:': 'by saying nothing, at any level',
                    'training state lives:': 'a third tree, outside the transform',
                    'params kept out of the gradient:': 'nothing: init returns them apart',
                    'lifetimes under training:': 'the same loop, a different rollout',
                    'recording and session, trained:': 'three nested lax.scans; each reset a constant at a scan top',
                })


if __name__ == '__main__':
    main()
