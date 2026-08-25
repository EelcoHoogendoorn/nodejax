"""Chunked sequences in equinox, the whole run compiled as one program.

Same task, same numbers, same references as `chunk_nodejax.py`. lax.scan over
chunks of lax.scan over steps.

Equinox comes out of this well and the reason deserves saying plainly: its
state is an ordinary pytree value, so both scans carry it the way they carry
anything else. Nothing is lifted, nothing is annotated, and there is no
framework notion of a boundary to learn, because carrying is what happens when
you pass the value along and restarting is what happens when you do not.

What it costs is that nothing is declared, so nothing is checkable: the
boundary is a line in a loop body, written in terms of what the model happens
to keep, and moving it one loop up or down leaves a program that runs and is
wrong.

Mixed lifetimes are where it thins out. Here both carried pieces have the same
lifetime, so threading them is uniform. If the hidden state had to restart
while the statistics carried, the caller would be re-initing one subtree from
scratch and keeping the other, per boundary, by hand.

Run directly:  python -m nodejax.examples.comparisons.chunk.chunk_equinox
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from nodejax.examples.comparisons.chunk import chunk_common as task
from nodejax.types import PyTree


class Norm(eqx.Module):
    """Running per-feature normalizer. Its statistics live in the State,
    under an index it owns."""

    index: eqx.nn.StateIndex

    def __init__(self):
        _, mean, var = task.cold()
        self.index = eqx.nn.StateIndex((mean, var))

    def __call__(self, x, state):
        mean, var = state.get(self.index)
        return task.normalize(x, mean, var), state.set(
            self.index, task.update_stats(x, mean, var))


class RNN(eqx.Module):
    """The recurrent cell, with WEIGHTS. Its hidden state is an ORDINARY
    VALUE, in and out, because the caller's scan carries it. Two kinds of
    state in one model, and which kind a piece goes in is decided by whether
    the scan carries it.

    The weights are FIELDS, which is to say the module IS its parameters: an
    eqx.Module is a pytree of them, and the key arrives at the constructor,
    which is the equinox convention. Adding them costs nothing at the scans,
    because the module was already closed over rather than threaded. It costs
    later instead, when training needs the parameters separated from
    everything else in the same pytree by a filter."""

    wx: jnp.ndarray
    wh: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, key):
        self.wx, self.wh, self.b = task.weights(key)

    def __call__(self, hidden, x):
        return task.cell(hidden, x, self.wx, self.wh, self.b)


class Step(eqx.Module):
    """The two composed, the ordinary way: submodules as fields.

    Composing costs a level of PATH. Everything that reaches for the
    statistics from outside now goes through `model.norm.index` rather than
    `model.norm.index`, and the boundary lines are all outside."""

    norm: Norm
    rnn: RNN

    def __init__(self, key):
        # the key is threaded down by hand, one constructor to the next
        self.norm, self.rnn = Norm(), RNN(key)

    def __call__(self, hidden, x, state):
        normalized, state = self.norm(x, state)
        return self.rnn(hidden, normalized), state


def run(seq: jax.Array=task.SEQ):
    """The whole chunked run as one program: lax.scan over chunks of lax.scan
    over steps.

    Nothing is lifted and nothing is annotated, because equinox state is an
    ordinary pytree value and both scans carry it the way they carry anything
    else. The boundary is the top of the outer body, where the snapshot is
    read; the caller writes that, reaching into the model's own slot."""
    model, state = eqx.nn.make_with_state(Step)(task.PARAM_KEY)
    hidden = task.carry_init(task.INIT_KEY)     # the caller draws it, by hand

    def chunk(carry, xs):
        hidden, state = carry
        def step(carry, x):
            hidden, state = carry
            h, state = model(hidden, x, state)
            return (h, state), h

        return jax.lax.scan(step, (hidden, state), xs)

    _, outs = jax.lax.scan(chunk, (hidden, state), seq.reshape(-1, task.CHUNK, task.W))
    return outs.reshape(-1, task.H)


def run_recordings(seq: jax.Array=task.SEQ):
    """TWO boundaries, and a piece of state answering each.

    Nothing declares anything, so the caller writes both, in both places, and
    the two do not look alike. The chunk boundary is the snapshot read at the
    top of the chunk body, reaching into the model's own state slot by index.
    The recording boundary is `hidden0` passed instead of the threaded carry
    at the top of the recording body. Same idea, two spellings, neither
    checkable: nothing here says which slots belong to which lifetime, and
    both bodies would run identically if the two were swapped.

    The draw makes the asymmetry plainer. The carry is an ordinary value, so
    its key is an ordinary value too: split at the top, threaded through the
    outer scan as a second xs beside the data, indexed by nothing but position.
    Nothing relates the key to the carry it initializes; the pairing is that
    both happen to be written on the same line.

    It works this cleanly only because the two lifetimes live in different
    KINDS of thing: `hidden` is a plain value the caller already owns, and the
    statistics are in the State. Give a real model two lifetimes inside the
    State and the recording body becomes pytree surgery, re-initing some
    subtrees from init and keeping others, at every boundary, with nothing
    checking that the split is right."""
    model, state0 = eqx.nn.make_with_state(Step)(task.PARAM_KEY)
    # one key, split by the caller into one per recording, because nothing
    # else will: the carry is an ordinary value and its draw is an ordinary
    # line, so the keys are threaded through the scan as data alongside it
    keys = jax.random.split(task.INIT_KEY, task.RECORDINGS)

    def chunk(carry, xs):
        def step(carry, x):
            hidden, state = carry
            h, state = model(hidden, x, state)
            return (h, state), h

        return jax.lax.scan(step, carry, xs)

    def recording(state, xs):
        chunks, key = xs                                 # the key rides the xs
        # the RECORDING boundary: carry drawn afresh, statistics kept
        (_, state), outs = jax.lax.scan(chunk, (task.carry_init(key), state), chunks)
        return state, outs

    _, outs = jax.lax.scan(recording, state0, (task.recordings(seq), keys))
    return outs.reshape(-1, task.H)


def train(recordings: PyTree=False, seq: jax.Array=task.SEQ):
    """FOURTH question: train the weights through the chunked rollout.

    Equinox's one pytree comes due here. A module IS its parameters, and it is
    also everything else it holds, so differentiating means splitting the tree
    first: eqx.partition on a PREDICATE, then eqx.combine inside the loss.

    The predicate is `is_inexact_array`, which is a guess that happens to be
    right for this model. It says nothing about parameters; it says "float
    arrays", and anything float the module held that was not a weight would be
    trained too, silently. Nothing here declares a parameter, so nothing can
    check the guess.

    The statistics escape it only because they live in the State rather than
    in the module, which is the same accident that made the recording boundary
    a clean line. Put them in the module and the predicate reaches them."""
    model, state0 = eqx.nn.make_with_state(Step)(task.PARAM_KEY)
    hidden = task.carry_init(task.INIT_KEY)
    chunks = seq.reshape(-1, task.CHUNK, task.W)

    def rollout(m, state):
        """The lifetimes are the caller's lines, so they are written again
        here: a fresh carry at the top of the recording body when there are
        recordings. Nothing declared it the first time and nothing carries it
        over, which is the cost this column pays twice."""
        def chunk(carry, xs):
            def step(c, x):
                h, st = c
                out, st = m.norm(x, st)
                return (m.rnn(h, out), st), m.rnn(h, out)

            return jax.lax.scan(step, carry, xs)

        if not recordings:
            _, outs = jax.lax.scan(chunk, (hidden, state), chunks)
            return outs.reshape(-1, task.H)

        keys = jax.random.split(task.INIT_KEY, task.RECORDINGS)

        def recording(st, xs):
            chunk_data, key = xs
            # the RECORDING boundary: carry drawn afresh, statistics kept
            (_, st), outs = jax.lax.scan(chunk, (task.carry_init(key), st),
                                         chunk_data)
            return st, outs

        _, outs = jax.lax.scan(recording, state, (task.recordings(seq), keys))
        return outs.reshape(-1, task.H)

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params):
        return task.loss_of(rollout(eqx.combine(params, static), state0))

    # the loop is a lax.scan, because that is what a user of a pure-functional
    # framework writes: the training state is a carry like any other, and a
    # Python loop over jax code is not something a real user would put up with
    def step(params, _):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        # HAND-ROLLED SGD: these frameworks own no optimizer, optax being
        # the ecosystem's, and this loop owns its own arithmetic instead
        return jax.tree.map(lambda w, g: w - task.LR * g, params, grads), loss

    _, losses = jax.lax.scan(step, params, None, length=task.TRAIN_STEPS)
    return losses



def train_sessions():
    """FIFTH question: recording and session in one scan nest.

    The same shape as haiku's: three lax.scans the caller owns, each lifetime
    the constant entering at its scan's top. equinox contributes the module
    and the partition; the training structure is plain jax around it."""
    model, state0 = eqx.nn.make_with_state(Step)(task.PARAM_KEY)
    params0, static = eqx.partition(model, eqx.is_inexact_array)
    hidden0 = task.carry_init(task.INIT_KEY)

    def forward(p, state, hidden, x):
        m = eqx.combine(p, static)

        def stp(c, xi):
            h, st = c
            h2, st = m(h, xi, st)
            return (h2, st), h2

        (h2, st), outs = jax.lax.scan(stp, (hidden, state), x)
        return h2, st, outs

    def over_chunks(carry, xt):
        p, vel, state, hidden = carry
        x, t = xt

        def loss_fn(pp):
            h2, s2, outs = forward(pp, state, hidden, x)
            return jnp.mean((outs - t) ** 2), (h2, s2)

        (loss, (h2, s2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        # HAND-ROLLED MOMENTUM, same note: the velocity is the caller's
        # pytree because nothing in the framework owns the training loop
        vel = jax.tree.map(lambda v, g: task.MU * v + g, vel, grads)
        p = jax.tree.map(lambda w, v: w - task.LR * v, p, vel)
        return (p, vel, s2, h2), loss

    def over_recordings(carry, xt):
        p, vel, state = carry
        (p, vel, state, _), losses = jax.lax.scan(
            over_chunks, (p, vel, state, hidden0), xt)        # RECORDING: hidden dies
        return (p, vel, state), losses

    def over_sessions(p, xt):
        vel0 = jax.tree.map(jnp.zeros_like, p)   # SESSION: momentum restarts
        (p, _, _), losses = jax.lax.scan(
            over_recordings, (p, vel0, state0), xt)            # SESSION: recalibrated
        return p, losses

    _, losses = jax.lax.scan(over_sessions, params0,
                             (task.SESS_SEQ, task.SESS_TARGET))
    return losses.reshape(-1)


def main() -> None:
    task.report('equinox',
                live_ok=bool(jnp.allclose(run(), task.reference_live(), atol=1e-5)),
                two_ok=bool(jnp.allclose(run_recordings(),
                                         task.reference_recordings(), atol=1e-5)),
                train_ok=bool(jnp.allclose(train(), task.reference_trained(), atol=1e-4)),
                train_two_ok=bool(jnp.allclose(train(recordings=True),
                                               task.reference_trained_recordings(),
                                               atol=1e-4)),
                train_tags_ok=bool(jnp.allclose(
                    train_sessions(), task.reference_trained_sessions(),
                    atol=1e-4)),
                cost={
                    'hidden across chunks:': 'the scan carry, nothing declared',
                    'stats across chunks:': 'the scan carry, nothing declared',
                    'model edited:': 'no',
                    'slots named outside:': 'model.norm.index, at every boundary',
                    'carry re-inits per recording:': 'the caller re-inits it, per level',
                    'the carry is DRAWN:': 'a key split by the caller, threaded as xs',
                    'the cell has WEIGHTS:': 'fields; a key threaded ctor to ctor',
                    'stats cross both:': 'by not touching them, at both levels',
                    'training state lives:': 'a tree beside the partitioned module',
                    'lifetimes under training:': 'the caller writes them again, in the loss',
                    'recording and session, trained:': 'three nested scans; resets are which constant enters where',
                    'params kept out of the gradient:': 'a PREDICATE over the one pytree',
                })


if __name__ == '__main__':
    main()
