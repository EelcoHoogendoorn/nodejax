"""Shared task for the chunked-sequence comparisons: what happens to state
between calls.

A sequence too long to process at once is fed in chunks, one chunk per call.
Chunks group into recordings, so there are two kinds of boundary, and a piece
of state can answer to either or neither:

                    recording 0        ||        recording 1
                chunk 0     chunk 1    ||    chunk 0     chunk 1
               --------------------------------------------------
   hidden      h>h>h>h>h | h>h>h>h>h   || * h>h>h>h>h | h>h>h>h>h
   stats       s>s>s>s>s | s>s>s>s>s   ||   s>s>s>s>s | s>s>s>s>s

     >  advances a step      |  chunk boundary      ||  recording boundary
     *  rebuilt from init

   hidden      crosses chunks, re-inits per recording. A new recording is a
               new signal, but the chunked run of ONE recording still has to
               equal the unchunked run of it.
   stats       crosses both. Statistics describe the sensor, not the chunk
               they were measured in and not the recording.

Two rows, and only ONE of them has a `*` in it. That is the shape of the
answer: a carry crossing a boundary is the ordinary case, and the departure
is what a framework has to give you a way to say. Both pieces sit under both
loops and nothing about either piece says which loop it answers to, so
position distinguishes nothing: only a NAME can bind the departure to the
boundary it belongs to.

The cell also holds WEIGHTS, initialized identically everywhere. They are
here because param plumbing is supposed to be orthogonal to state plumbing,
and in some columns it is not: the same lifts that annotate where state goes
have to be told where params go as well.

The questions, each with an answer here in plain jax to be checked against:

    LIVE       reference_live                the whole sequence in chunks;
                                             everything carries
    TWO        reference_recordings          recordings of chunks; the carry
                                             re-inits per recording, the
                                             statistics cross both
    TRAINED    reference_trained             the weights train through the
                                             chunked rollout
    ...TWO     reference_trained_recordings  both lifetimes, while training
    TAGS       reference_trained_sessions    recording AND session in one
                                             tree, both crossing the trainer:
                                             the hidden dies per recording,
                                             the calibration per session, and
                                             the weights couple everything

A question rides along unasserted, answered by reading the files: the carry
is DRAWN, and each column draws it the way that framework would. nodejax
names rng in the node's init and says nothing more, the machinery routing one
key per re-init wherever those turn out to be. Linen routes entropy through
named RNG collections and `split_rngs` at lifted scans. NNX carries typed RNG
state through its graph-aware scans. Equinox threads a key as an ordinary
value. Haiku draws from a key owned by its transformed function. Torch calls
its global generator and threads no key through the model call.

The weights are fixed constants shared by every file, so the six columns are
not merely similar: they must produce the same numbers as each other and as
these references, to floating point. What the files record is what each
spelling COST, not what it could manage.

flax appears TWICE on purpose. nnx is the current API and linen the lifted
one it largely replaces, and the two files differ by more than the other
columns differ from each other. Comparing against linen alone would have
flattered everyone else.

WHERE THE LOOP LIVES is the structural difference underneath the rows. In the
NodeJAX column, `scan(node)` returns another component, so the scans belong to
the composition before data flows. Linen lifts a Module definition and states
how each variable collection crosses each scan. NNX's graph-aware scan accepts
Modules directly and preserves reference updates; `StateAxes` states which
Variable kinds map, carry, or broadcast at that transform site. Equinox places
the loop around its Module PyTree and explicit state. Haiku places it inside a
transformed function. The checked PyTorch formulation keeps it in its runner.

The defaults differ, and that is most of what the comparison shows. Every
column carries state across a boundary unless something says otherwise, and
they differ in whether that is a property of the framework or of the loop you
happened to write. Torch's buffers mutate in place, so carrying is what the
objects do and a forgotten line leaks silently; chunk_torch hoists one line
to show it, and executes the mistake rather than describing it. The jax
columns carry because a scan threads its carry, so the equivalent slip is a
re-init written in the wrong loop body, which nothing checks either.
"""

import functools

from typing import Callable

import jax
import jax.numpy as jnp
from nodejax.types import PyTree


def _once(fn: Callable):
    """A reference is a constant. Six columns each asking for it is six times
    the work for one answer, so the no-argument call is remembered. Called
    with a sequence of its own it computes afresh, since that is a question
    about different data."""
    cache = {}

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if args or kwargs:
            return fn(*args, **kwargs)
        if 'value' not in cache:
            cache['value'] = fn()
        return cache['value']

    return wrapped

W, H = 4, 6                     # signal width, hidden width
T, CHUNK = 24, 6                # 24 steps fed as four chunks of six
MOMENTUM = 0.2
EPS = 1e-5

PARAM_KEY = jax.random.PRNGKey(0)


def weights(key: jax.Array=None):
    """The cell's weights. Takes a key and IGNORES it, drawing from a fixed
    one instead.

    Ignoring it is the same device as INIT_SCALE, on the other channel. The
    SIGNATURE is the point: every column has to obtain a key and get it to a
    parameter initializer, which is the plumbing worth comparing, and five
    derivations cannot agree on values so the values are not left to them.
    Honour the key and the columns diverge, which would be a fact about
    generators rather than about any framework.

    Where the key comes from is what differs, and it is the whole exercise:
    flax hands one to a param initializer, haiku to hk.get_parameter, equinox
    to a constructor, nodejax to the param function. torch is passed nothing,
    because torch has nothing to pass, which is its answer here as it was for
    the carry."""
    del key                                      # deliberately, see above
    kx, kh = jax.random.split(PARAM_KEY)
    return (jax.random.normal(kx, (W, H)) / jnp.sqrt(W),
            jax.random.normal(kh, (H, H)) / jnp.sqrt(H),
            jnp.zeros(H))


WX, WH, B = weights(PARAM_KEY)
_k3 = jax.random.split(PARAM_KEY, 3)[2]

SEQ = jnp.sin(jax.random.normal(_k3, (T, W)) * 2.0) * 1.5 + 0.3


def chunks(seq: jax.Array=SEQ):
    """The sequence in CHUNK-sized pieces, in order."""
    return [seq[i:i + CHUNK] for i in range(0, len(seq), CHUNK)]


def cell(hidden: int, x_normalized, wx=WX, wh=WH, b: jax.Array=B):
    """The recurrent step, identical everywhere.

    The weights default to the shared constants so the references below stay
    plain functions, and each framework passes its OWN copies instead: the
    cell has parameters now, which is the point. Every column initializes them
    from these same values, so the numbers do not move and what the comparison
    shows is what carrying a param alongside state cost."""
    return jnp.tanh(x_normalized @ wx + hidden @ wh + b)


def normalize(x: jax.Array, mean, var):
    return (x - mean) / jnp.sqrt(var + EPS)


def update_stats(x: jax.Array, mean, var):
    mean = (1 - MOMENTUM) * mean + MOMENTUM * x
    var = (1 - MOMENTUM) * var + MOMENTUM * (x - mean) ** 2
    return mean, var


# The carry is DRAWN, which is the second thing this task measures. Each file
# draws it the way that framework would, so what the comparison records is who
# holds a key, who threads it, and who has to remember to advance it.
#
# INIT_SCALE is how four different generators still agree on the numbers. At
# zero the draw contributes nothing to the arithmetic while the plumbing that
# would carry it is written out in full, so the references stay comparable to
# the floating-point digit. Raise it and the columns diverge, which is the
# honest consequence of four generators rather than a fault in any of them.
INIT_SCALE = 0.0
INIT_KEY = jax.random.PRNGKey(7)


def carry_init(key: jax.Array):
    """The carry's cold value: a fresh draw, scaled by INIT_SCALE."""
    return jax.random.normal(key, (H,)) * INIT_SCALE


def cold(key: jax.Array=INIT_KEY):
    """The cold state: hidden, running mean, running var.

    The hidden part is drawn, so a file has to have a key to build it. The
    statistics keep their natural cold values: a normalizer that begins
    believing the signal is standard is the honest start, and a random one
    would be noise for its own sake."""
    return carry_init(key), jnp.zeros(W), jnp.ones(W)


@_once
def reference_live(seq: jax.Array=SEQ, params: PyTree=None):
    """The whole sequence in one pass, every step reading the statistics as
    they stand. The answer a chunked run has to reproduce.

    `params` is explicit so this can be DIFFERENTIATED. The weights default to
    the drawn ones, which is what every non-training check passes, and the
    training reference below passes its current ones instead."""
    wx, wh, b = weights() if params is None else params

    def step(carry, x):
        hidden, mean, var = carry
        hidden = cell(hidden, normalize(x, mean, var), wx, wh, b)
        mean, var = update_stats(x, mean, var)
        return (hidden, mean, var), hidden

    _, out = jax.lax.scan(step, cold(), seq)
    return out


RECORDINGS, PER_RECORDING = 2, 2        # 24 steps as 2 recordings of 2 chunks


def recordings(seq: jax.Array=SEQ):
    """The same sequence as RECORDINGS recordings of PER_RECORDING chunks."""
    return seq.reshape(RECORDINGS, PER_RECORDING, CHUNK, W)


@_once
def reference_recordings(seq: jax.Array=SEQ, params: PyTree=None):
    """TWO nested loops, and the name picks which one a lifetime answers to.

    A recording is fed in chunks. The carry crosses chunks and RE-INITS per
    recording, because a new recording is a new signal; the statistics cross
    everything, because they describe the sensor and not any one recording.
    Both pieces sit under both loops, and nothing about either piece says
    which loop it answers to: re-init the carry at the chunk instead and the
    same tree computes something else."""
    wx, wh, b = weights() if params is None else params
    _, mean, var = cold()                       # the statistics cross everything
    keys = jax.random.split(INIT_KEY, RECORDINGS)
    outs = []
    for key, recording in zip(keys, recordings(seq)):
        hidden = carry_init(key)                # a FRESH draw, once per recording
        for chunk in recording:
            def step(carry, x):
                hidden, mean, var = carry
                hidden = cell(hidden, normalize(x, mean, var), wx, wh, b)
                mean, var = update_stats(x, mean, var)
                return (hidden, mean, var), hidden

            (hidden, mean, var), out = jax.lax.scan(step, (hidden, mean, var), chunk)
            outs.append(out)
    return jnp.concatenate(outs)


TARGET = jnp.tanh(jax.random.normal(jax.random.split(PARAM_KEY, 4)[3], (T, H)))
LR, TRAIN_STEPS = 0.5, 40

# a corpus of SESSIONS, each recorded under its own calibration: distinct data
# per session, visited once, in the same recipes as SEQ and TARGET
SESSIONS = TRAIN_STEPS // (RECORDINGS * PER_RECORDING)
_sk1, _sk2 = jax.random.split(jax.random.PRNGKey(11))
SESS_SEQ = jnp.sin(jax.random.normal(
    _sk1, (SESSIONS, RECORDINGS, PER_RECORDING, CHUNK, W)) * 2.0) * 1.5 + 0.3
SESS_TARGET = jnp.tanh(jax.random.normal(
    _sk2, (SESSIONS, RECORDINGS, PER_RECORDING, CHUNK, H)))
# the sessions question trains with MOMENTUM, so the optimizer carries state
# of its own: a velocity per weight, the third KIND of state in the tree
MU = 0.9


def loss_of(outs) -> jax.Array:
    """One scalar from a whole rollout: how far the trajectory is from a fixed
    target. Arbitrary as an objective and deliberately so, since what is being
    compared is where the gradient has to travel, not what it is asked to
    learn."""
    return jnp.mean((outs - TARGET) ** 2)


@_once
def reference_trained(seq: jax.Array=SEQ):
    """FOURTH question: train the cell's weights through the chunked rollout.

    Plain SGD, a fixed rate, a fixed number of steps, so every column can
    reproduce the loss trace exactly rather than merely converging. optax is
    not used here for the same reason jnp is not used for the model: the
    reference owes the others no dependency.

    What it exercises is the thing the other three questions do not. A
    gradient has to reach the weights THROUGH the state: the normalizer's
    statistics are updated inside the same scan and are not trained, so every
    framework has to say which of the things it carries are parameters and
    which are not, and say it to the differentiation rather than to the scan."""
    return _trained(reference_live, seq)


def _trained(rollout, seq: jax.Array):
    """Plain SGD over whichever rollout, so the three lifetimes and the
    training loop can be checked together rather than one at a time."""
    step = jax.jit(jax.value_and_grad(lambda p: loss_of(rollout(seq, p))))
    params = weights()
    losses = []
    for _ in range(TRAIN_STEPS):
        loss, grads = step(params)
        losses.append(loss)
        params = tuple(w - LR * g for w, g in zip(params, grads))
    return jnp.stack(losses)


@_once
def reference_trained_recordings(seq: jax.Array=SEQ):
    """SIXTH question: both boundaries, while the weights train.

    The hardest cell of the table. Three lifetimes, two boundaries, a carry
    drawn afresh per recording, and a gradient travelling through all of it to
    reach the weights."""
    return _trained(reference_recordings, seq)


@_once
def reference_trained_sessions():
    """FIFTH question: RECORDING and SESSION in one tree, both crossing the
    trainer, and THREE kinds of state dying at two boundaries. The corpus is
    sessions, each recorded under its own calibration, visited once. The
    statistics re-measure per session; the optimizer's momentum restarts with
    them, stale velocity being a fact about the previous session's data; the
    hidden dies per recording; the weights cross everything, coupling the
    sessions into one computation.

                           session 0             |||             session 1
                    rec 0              rec 1              rec 0              rec 1
                -----------------------------------------------------------------------
       hidden   h>h>h | h>h>h || * h>h>h | h>h>h ||| * h>h>h | h>h>h || * h>h>h | h>h>h
       stats    s>s>s | s>s>s ||   s>s>s | s>s>s ||| * s>s>s | s>s>s ||   s>s>s | s>s>s
       moments  ==>== | ==>== ||   ==>== | ==>== ||| * ==>== | ==>== ||   ==>== | ==>==
       weights  ==>== | ==>== ||   ==>== | ==>== |||   ==>== | ==>== ||   ==>== | ==>==

         >  advances a step        |   chunk boundary       *  rebuilt from init
        ||  recording boundary   |||   session boundary

       hidden    dies per recording: a new recording is a new signal
       stats     die per session: each session carries its own calibration
       moments   one > per chunk, like the weights they steer, and they die
                 with the calibration: OPTIMIZER state, not model state
       weights   cross every boundary; they are what couples the sessions

    Truncated BPTT falls out rather than being chosen: the hidden enters each
    step as a carried value, so one step's gradient reaches back exactly to
    the start of its own chunk."""
    def step(params, vel, carry, xchunk, tchunk):
        def loss_fn(p):
            wx, wh, b = p

            def one(c, x):
                h, m, v = c
                h = cell(h, normalize(x, m, v), wx, wh, b)
                m, v = update_stats(x, m, v)
                return (h, m, v), h

            c2, outs = jax.lax.scan(one, carry, xchunk)
            return jnp.mean((outs - tchunk) ** 2), c2

        (loss, c2), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        vel = tuple(MU * v + g for v, g in zip(vel, grads))
        params = tuple(w - LR * v for w, v in zip(params, vel))
        return params, vel, c2, loss

    step = jax.jit(step)
    params = weights()
    losses = []
    for si in range(SESSIONS):
        mean, var = jnp.zeros(W), jnp.ones(W)          # SESSION: recalibrated
        vel = tuple(jnp.zeros_like(w) for w in params)  # SESSION: momentum restarts
        for r in range(RECORDINGS):
            carry = (carry_init(INIT_KEY), mean, var)  # RECORDING: the hidden dies
            for c in range(PER_RECORDING):
                params, vel, carry, loss = step(params, vel, carry,
                                                SESS_SEQ[si, r, c],
                                                SESS_TARGET[si, r, c])
                losses.append(loss)
            _, mean, var = carry                       # the calibration carries on
    return jnp.stack(losses)


def report(name: str, live_ok: bool, two_ok: bool,
           cost: dict[str, str], train_ok: bool | None = None,
           train_two_ok: bool | None = None,
           train_tags_ok: bool | None = None) -> None:
    """One framework's verdict: does it reproduce every reference, and what did
    each lifetime cost to express."""
    print(f'\n[{name}]')
    print(f'  live reference reproduced:           {live_ok}')
    print(f'  two boundaries reproduced:           {two_ok}')
    if train_ok is not None:
        print(f'  training trace reproduced:           {train_ok}')
    if train_two_ok is not None:
        print(f'  ...with both boundaries:             {train_two_ok}')
    if train_tags_ok is not None:
        print(f'  ...two tags, recording and session:  {train_tags_ok}')
    for question, answer in cost.items():
        print(f'  {question:<24} {answer}')

    # printed AND asserted, because printing alone is not a check. Making Norm
    # a submodule silently broke two of flax's three, its collection path
    # returning defaults rather than raising, and the run said False in a line
    # nobody was reading.
    checks = [('live', live_ok), ('two boundaries', two_ok)]
    for label, ok in (('training', train_ok),
                      ('training two boundaries', train_two_ok),
                      ('training with two tags', train_tags_ok)):
        if ok is not None:
            checks.append((label, ok))
    failed = [q for q, ok in checks if not ok]
    if failed:
        raise AssertionError(f'{name} does not reproduce: {", ".join(failed)}')
