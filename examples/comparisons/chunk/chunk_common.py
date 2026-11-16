"""Shared task for the chunked-sequence comparisons: what happens to state
between calls.

A sequence too long to process in one call is fed in chunks, one chunk per call.
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
               new signal, while the chunked run of one recording still has to
               equal the unchunked run of it.
   stats       crosses both. Statistics describe the sensor, not the chunk
               they were measured in and not the recording.

Two rows, and only one of them has a `*` in it. That is the shape of the
answer: a carry crossing a boundary is the ordinary case, and the departure
is what a framework has to give you a way to say. Both pieces sit under both
loops and nothing about either piece says which loop it answers to, so
position distinguishes nothing: only a name can bind the departure to the
boundary it belongs to.

The cell also holds weights, initialized identically everywhere. They are
here because param plumbing is supposed to be orthogonal to state plumbing,
and in some columns it is not: the same lifts that annotate where state goes
have to be told where params go as well.

The questions, each with an answer here in plain jax to be checked against:

    live       reference_live                the whole sequence in chunks;
                                             everything carries
    two        reference_recordings          recordings of chunks; the carry
                                             re-inits per recording, the
                                             statistics cross both
    trained    reference_trained             the weights train through the
                                             chunked rollout
    ...two     reference_trained_recordings  both lifetimes, while training
    tags       reference_trained_sessions    recording and session in one
                                             tree, both crossing the trainer:
                                             the hidden dies per recording,
                                             the calibration per session, and
                                             the weights couple everything

A question rides along unasserted, answered by reading the files: the carry
is drawn, and each column draws it the way that framework would. nodejax
names rng in the node's init and says nothing more, the machinery routing one
key per re-init wherever those turn out to be. Linen routes entropy through
named RNG collections and `split_rngs` at lifted scans. NNX carries typed RNG
state through its graph-aware scans. Equinox threads a key as an ordinary
value. Haiku draws from a key owned by its transformed function. Torch calls
its global generator and threads no key through the model call.

The weights are fixed constants shared by every file, so the six columns are
not merely similar: they must produce the same numbers as each other and as
these references, to floating point. What the files record is what each
formulation cost to write, not what it could manage.

Flax appears twice on purpose. NNX is the current API and Linen the lifted
one it largely replaces, and the two files differ by more than the other
columns differ from each other. Comparing against Linen alone would have
flattered everyone else.

Where the loop lives is the structural difference underneath the rows. In the
NodeJAX column, `scan(node)` returns another Node, so the scans belong to
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

PyTree = object


def _once(fn: Callable):
    """Caches the zero-argument reference result across columns while
    recomputing fresh calls when explicit sequence or parameter arguments
    are passed."""
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


def weights(key: jax.Array | None = None):
    """Returns the shared cell weights, ignoring `key` and drawing from
    `PARAM_KEY` instead.

    Ignoring `key` is the same device as `INIT_SCALE`, on the parameter role.
    The signature is the point: every column has to obtain a key and get it to a
    parameter initializer, which is the plumbing worth comparing, and five
    derivations cannot agree on values so the values are not left to them.
    Honour the key and the columns diverge, which would be a fact about
    generators rather than about any framework.

    Where the key comes from is what differs, and it is the whole exercise:
    flax hands one to a param initializer, haiku to hk.get_parameter, equinox
    to a constructor, nodejax to the param function. torch is passed nothing,
    because torch has nothing to pass, which is its answer here as it was for
    the carry."""
    kx, kh = jax.random.split(PARAM_KEY)
    return (jax.random.normal(kx, (W, H)) / jnp.sqrt(W),
            jax.random.normal(kh, (H, H)) / jnp.sqrt(H),
            jnp.zeros(H))


WX, WH, B = weights(PARAM_KEY)
_k3 = jax.random.split(PARAM_KEY, 3)[2]

SEQ = jnp.sin(jax.random.normal(_k3, (T, W)) * 2.0) * 1.5 + 0.3


def chunks(seq: jax.Array = SEQ):
    """Slices `seq` into `CHUNK`-sized pieces in order."""
    return [seq[i:i + CHUNK] for i in range(0, len(seq), CHUNK)]


def cell(hidden: jax.Array, x_normalized: jax.Array, wx=WX, wh=WH, b: jax.Array = B):
    """Computes one recurrent `tanh` step.

    The weights default to the shared constants so the references below stay
    plain functions, and each framework passes its own copies instead: the
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


# The carry is drawn, which is the second thing this task measures. Each file
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
    """Draws the initial hidden carry scaled by `INIT_SCALE`."""
    return jax.random.normal(key, (H,)) * INIT_SCALE


def cold(key: jax.Array = INIT_KEY):
    """Builds the initial `(hidden, mean, var)` tuple from `key`.

    The hidden part is drawn, so a file has to have a key to build it. The
    statistics keep their natural cold values: a normalizer that begins
    believing the signal is standard is the honest start, and a random one
    would be noise for its own sake."""
    return carry_init(key), jnp.zeros(W), jnp.ones(W)


@_once
def reference_live(seq: jax.Array = SEQ, params: PyTree = None):
    """Runs `seq` in one pass with live running statistics as the reference
    that chunked rollouts must reproduce.

    `params` is explicit so this can be differentiated. The weights default to
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


def recordings(seq: jax.Array = SEQ):
    """Reshapes `seq` into `(RECORDINGS, PER_RECORDING, CHUNK, W)`."""
    return seq.reshape(RECORDINGS, PER_RECORDING, CHUNK, W)


@_once
def reference_recordings(seq: jax.Array = SEQ, params: PyTree = None):
    """Runs two nested loops where hidden state resets per recording while
    normalization statistics persist across recordings.

    A recording is fed in chunks. The carry crosses chunks and re-inits per
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
        hidden = carry_init(key)                # a fresh draw, once per recording
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

# a corpus of `SESSIONS` calibration sessions, each with distinct data visited
# once in the same shape recipe as SEQ and TARGET
SESSIONS = TRAIN_STEPS // (RECORDINGS * PER_RECORDING)
_sk1, _sk2 = jax.random.split(jax.random.PRNGKey(11))
SESS_SEQ = jnp.sin(jax.random.normal(
    _sk1, (SESSIONS, RECORDINGS, PER_RECORDING, CHUNK, W)) * 2.0) * 1.5 + 0.3
SESS_TARGET = jnp.tanh(jax.random.normal(
    _sk2, (SESSIONS, RECORDINGS, PER_RECORDING, CHUNK, H)))
# the sessions question trains with `MOMENTUM`, so the optimizer carries state
# of its own: a velocity per weight, the third kind of state in the tree
MU = 0.9


def loss_of(outs) -> jax.Array:
    """Computes mean squared error between `outs` and `TARGET`. Arbitrary as an
    objective and deliberately so, since what is being
    compared is where the gradient has to travel, not what it is asked to
    learn."""
    return jnp.mean((outs - TARGET) ** 2)


@_once
def reference_trained(seq: jax.Array = SEQ):
    """Trains the cell weights by SGD through the chunked live rollout.

    Plain SGD, a fixed rate, a fixed number of steps, so every column can
    reproduce the loss trace exactly rather than merely converging. optax is
    not used here for the same reason jnp is not used for the model: the
    reference owes the others no dependency.

    What it exercises is the thing the other three questions do not. A
    gradient has to reach the weights through the state: the normalizer's
    statistics are updated inside the same scan and are not trained, so every
    framework has to say which of the things it carries are parameters and
    which are not, and say it to the differentiation rather than to the scan."""
    return _trained(reference_live, seq)


def _trained(rollout, seq: jax.Array):
    """Runs plain SGD over `rollout` for `TRAIN_STEPS` steps and returns the
    loss trace."""
    step = jax.jit(jax.value_and_grad(lambda p: loss_of(rollout(seq, p))))
    params = weights()
    losses = []
    for _ in range(TRAIN_STEPS):
        loss, grads = step(params)
        losses.append(loss)
        params = tuple(w - LR * g for w, g in zip(params, grads))
    return jnp.stack(losses)


@_once
def reference_trained_recordings(seq: jax.Array = SEQ):
    """Trains the cell weights by SGD across chunk and recording boundaries.

    Three lifetimes, two boundaries, a carry
    drawn afresh per recording, and a gradient travelling through all of it to
    reach the weights."""
    return _trained(reference_recordings, seq)


@_once
def reference_trained_sessions():
    """Trains across recording and session boundaries with three state
    lifetimes in one loop nest. The corpus is
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
                 with the calibration: optimizer state, not model state
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
        mean, var = jnp.zeros(W), jnp.ones(W)          # session: recalibrated
        vel = tuple(jnp.zeros_like(w) for w in params)  # session: momentum restarts
        for r in range(RECORDINGS):
            carry = (carry_init(INIT_KEY), mean, var)  # recording: the hidden dies
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
    """Prints and asserts one framework's parity checks and cost summary."""
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
