"""Chunked sequences in nodejax: lifetimes as declarations.

The model is a running normalizer piped into a recurrent cell, and the whole
comparison is about how much each framework makes you write to say what
survives which boundary.

The whole run compiles as one program: a scan over chunks of a scan over
steps, the same shape as the other files.

What changes between the behaviours is what wraps the model:

    scan(norm >> rnn)                                    everything survives
    scan(scan(norm >> state_reinit(rnn, 'recording')), boundary='recording')
                                                         the carry dies per recording

A carry carries, which is why the first line is bare. `state_reinit` says
this state re-inits at the named boundary, and it wraps a node written once
with no idea that chunking exists. A scan claims a name only when something
beneath answers to it: claiming a name nothing declares is an error rather
than a no-op.

The nesting uses both halves of the pair, and which goes where is forced.
Every scan but the outermost is `scan`: it changes the step grain and hands back a
cyclic node, which is the only reason the level above has something to run.
The outermost is `scanned`, because when the whole sequence is done nobody
holds the carry. `scanned` takes no boundary, and needs none: there is no
carry left for one to act on.

Run directly:  python -m examples.comparisons.chunk.chunk_nodejax
"""

import jax
import jax.numpy as jnp

import optax

from nodejax import Node, PNode, node, Leaf, scan, scanned, trained, state_reinit, train_step
from nodejax.transforms.learning import optimizer, opt_reinit
from nodejax import tile
from nodejax.struct import Struct
from examples.comparisons.chunk import chunk_common as task


@node
def Norm() -> PNode:
    """Maintains per-feature running mean and variance across steps without
    referencing chunk or recording boundaries.

    It names `signal` because it sits first in the pipe and the sequence is a
    named Struct. Entropy rides beside that data in the framework RNG frame;
    naming `rng` in an authored signature declares the separate plan rather
    than another sequence field. Only the first member names the data field:
    everything downstream is fed whatever the member before it returned."""
    def init():
        _, mean, var = task.cold()
        return Struct(mean=mean, var=var)

    def apply(state, signal):
        out = task.normalize(signal, state.mean, state.var)
        mean, var = task.update_stats(signal, state.mean, state.var)
        return Struct(mean=mean, var=var), out

    return Leaf(apply, init=init)


@node
def RNN() -> Node:
    """Carries a recurrent hidden vector drawn in `init` alongside trainable
    cell weights.

    Naming `rng` in init is the entire declaration. The node says it needs
    entropy to begin; it does not say where from, how often, or which of the
    enclosing scans will ask. The public key becomes an execution frame, and
    the machinery routes one stream here per re-init.

    It has weights too, drawn at construction, and they change nothing above.
    `param` is its own role with its own function and its own key, so the
    scans that thread state say not one word about params.

    Naming `rng` twice, in `param` and in `init`, declares two distinct roles:
    entropy to parameterize with and entropy to initialize state from. The
    machinery splits a separate stream toward each."""
    def param(rng):
        wx, wh, b = task.weights(rng.next())
        return Struct(wx=wx, wh=wh, b=b)

    def init(rng):
        return task.carry_init(rng.next())

    def apply(param, state, input):
        hidden = task.cell(state, input, param.wx, param.wh, param.b)
        return hidden, hidden

    return Leaf(apply, param=param, init=init)


def sequence(data):
    """Wraps `data` in `Struct(signal=data)` while RNG stays on `apply(rng=...)`."""
    return Struct(signal=data)


def run(inner: Node, seq: jax.Array = task.SEQ):
    """Reshapes `seq` into chunks, parameterizes `inner`, and runs it.

    Nothing that decides what survives lives here. That is all in the
    expression that built `inner`, which is the point of the file; this is
    data reshaping plus `parameterize`.

    Models without parameters skip `parameterize` directly."""
    return (inner.parameterize(rng=task.PARAM_KEY)
            .apply(bundle=sequence(seq.reshape(-1, task.CHUNK, task.W)),
                   rng=task.INIT_KEY).reshape(-1, task.H))


def run_recordings(inner: Node, seq: jax.Array = task.SEQ):
    """Parameterizes and runs `inner` over recordings of chunks."""
    return (inner.parameterize(rng=task.PARAM_KEY)
            .apply(bundle=sequence(task.recordings(seq)),
                   rng=task.INIT_KEY).reshape(-1, task.H))


def two_boundaries() -> Node:
    """Builds two nested scans where the `'recording'` boundary name resets
    only `RNN` while `Norm` statistics carry across both loops.

    The statistics cross everything, so they say nothing: that is what a
    carry does. The recurrent carry re-inits per recording, so `state_reinit`
    names the outer scan, and only the name can: both pieces sit under both
    scans, and neither one's code says a word about chunks or recordings.
    Claim 'recording' at the chunk scan instead and the same tree computes
    something else."""
    model = Norm() >> state_reinit(RNN(), boundary='recording')
    chunk = scan(model)                                       # steps -> one chunk
    recording = scan(chunk, boundary='recording')             # chunks -> one recording
    return scanned(recording)                                 # recordings -> the lot


def train(model, data):
    """Trains `model` weights through the chunked rollout while preserving
    every declared state lifetime.

    The model is already a node, so training is a transform over it and the
    loop is a scan of that. `train_step` moves the params into state, which is
    the same state slot everything else uses, so nothing new appears: no
    second mechanism, no third object to hold, no annotation anywhere saying
    which of the things being carried are the trainable ones.

    Where the gradient must not go, it does not go, and nothing had to say so.
    The normalizer's statistics update inside the same scan the gradient
    travels through, and they are state, a separate role from `param`, so
    differentiating with respect to `param` cannot reach them.

    The sequence ships `(input, target)` pairs, placing each target beside
    the data it scores rather than closing over it elsewhere.
    This task's target happens to be the same every step, so the sequence
    repeats it, and the loss is an ordinary two-argument one.

    `model` is a parameter here, which answers the multi-boundary training rows.
    The same expressions that answered the forward checks are passed in
    unchanged, wrapped in a trainer, and the boundaries go on meaning what
    they meant. Training is another transform over a node, so it composes with
    the tags rather than negotiating with them: nothing in `train_step` knows a
    boundary exists, and nothing in the tags knows an optimizer does.

    Note which way that composition runs. The tags are claimed inside the
    model, by the scans that own the grain they name. Nothing claims a
    boundary across `train_step`, and nothing should: a tag names a structure in
    the data, and a training step is not part of the data's structure. A model
    whose state lifetimes changed because it was under an optimizer would be a
    train/eval discrepancy by construction."""
    # a bound model in, a bound trainer out, taking those weights as where
    # training starts. scanned is the loop, so the whole of it is one
    # expression: a step, a loop over it, a sequence
    def loss(outs, target):
        return jnp.mean((outs.reshape(-1, task.H) - target) ** 2)

    trainer = train_step(model.parameterize(rng=task.PARAM_KEY), loss,
                         optax.sgd(task.LR))
    # `trained` outputs the final model and emits per-step metrics in aux, so
    # both halves come out of one construct. This row reads the loss trace
    _, aux = trained(trainer).apply(
        rng=task.INIT_KEY,
        signal=tile(data, task.TRAIN_STEPS),         # the same rollout each step
        target=tile(task.TARGET, task.TRAIN_STEPS))  # and the same target
    return aux.loss


def train_two_tags():
    """Trains across `'recording'` and `'session'` boundaries in one tree with
    three distinct state lifetimes. One pass over
    a corpus of sessions: the calibration dies per session and the
    optimizer's momentum restarts with it; the hidden dies per recording; the
    weights couple everything, crossing every boundary because a carry
    carries.

    The momentum is optimizer state, not model state, and the tree says so:
    its lifetime is declared by `opt_reinit` on the optimizer node, the
    partner-state counterpart of `state_reinit`, and the same `'session'`
    claim fires both it and the calibration's tag.

    Each tag is declared on the state it describes; each scan claims the one
    boundary it owns; the trainer sits between declaration and claim saying
    nothing. Swap the two claims and the same tree computes something else,
    which is what a name is for."""
    def loss(outs, target):
        return jnp.mean((outs - target) ** 2)

    model = scan(state_reinit(Norm(), 'session') >> state_reinit(RNN(), 'recording'))
    trainer = train_step(model.parameterize(rng=task.PARAM_KEY), loss,
                         opt_reinit(optimizer(optax.sgd(task.LR, momentum=task.MU)),
                                       'session'))
    run = scanned(scan(scan(trainer, boundary='recording'), boundary='session'))

    _, aux = run.apply(rng=task.INIT_KEY,
                       signal=task.SESS_SEQ,
                       target=task.SESS_TARGET)
    return aux.loss.reshape(-1)


def main() -> None:
    # the whole comparison, one expression each. What wraps the model IS the
    # difference between the two behaviours; the model is the same both times
    live = run(scanned(scan(Norm() >> RNN())))
    two = run_recordings(two_boundaries())

    # the SAME three expressions, wrapped in a trainer. Not one character of
    # any of them changes, which is the fifth and sixth questions in one line
    chunked = task.SEQ.reshape(-1, task.CHUNK, task.W)
    recs = task.recordings(task.SEQ)
    t_live = train(scanned(scan(Norm() >> RNN())), chunked)
    t_two = train(two_boundaries(), recs)

    task.report('nodejax',
                live_ok=bool(jnp.allclose(live, task.reference_live(), atol=1e-5)),
                two_ok=bool(jnp.allclose(two, task.reference_recordings(), atol=1e-5)),
                train_ok=bool(jnp.allclose(t_live, task.reference_trained(), atol=1e-4)),
                train_two_ok=bool(jnp.allclose(t_two, task.reference_trained_recordings(),
                                               atol=1e-4)),
                train_tags_ok=bool(jnp.allclose(
                    train_two_tags(), task.reference_trained_sessions(),
                    atol=1e-4)),
                cost={
                    'hidden across chunks:': 'nothing: a carry carries',
                    'stats across chunks:': 'nothing: a carry carries',
                    'model edited:': 'no',
                    'slots named outside:': 'none',
                    'carry re-inits per recording:': "state_reinit(RNN(), 'recording')",
                    'the carry is DRAWN:': "rng named in RNN's init, nothing else",
                    'the cell has WEIGHTS:': 'a param fn with its own rng; scans unchanged',
                    'stats cross both:': 'nothing to say',
                    'training state lives:': 'in the state slot, like everything else',
                    'params kept out of the gradient:': 'nothing: param is not state',
                    'lifetimes under training:': 'unchanged: the same expressions, wrapped',
                    'recording and session, trained:': 'two tags, two claims, the trainer between them silent',
                    'the momentum restarts per session:': "opt_reinit(optimizer(...), 'session'); the same claim",
                })


if __name__ == '__main__':
    main()
