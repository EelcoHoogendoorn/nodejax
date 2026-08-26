"""Chunked sequences in torch: persistence is the default, and that is the
whole difference.

Same task, same numbers, same references as `chunk_nodejax.py`. Torch is the
interesting column because its default is the INVERSE of every other file here.
Buffers are attributes that mutate, so state survives a call unless something
re-inits it. Nothing is declared, nothing is threaded, and carrying the running
statistics across a chunk costs zero lines.

That makes the easy question free and moves the cost onto the other one:

    stats across chunks     nothing. Buffers persist because they mutate.
    hidden across chunks    threaded by the caller, as in equinox
    carry restarts per      the caller must REMEMBER to re-init it. There is
    recording               no declaration to get wrong, only a line to forget

The failure modes invert with the default. Everywhere else in this comparison,
forgetting something means state re-inits when it should have carried, and you
notice because the chunked run disagrees with the whole run. Here, forgetting
means state carries when it should have re-inited: episode two starts with episode
one's statistics, nothing raises, and the model silently trains on a leak. The
`two boundaries` case below is one `torch.zeros` away from being silently wrong.

Two more torch-specific costs that have no analogue in the jax files:

1. `detach`. The hidden state carries the autograd graph with it, so a chunked
   training loop must detach at each boundary or the graph grows without bound.
   That is the truncated-BPTT idiom, and it is a THIRD thing the boundary means
   here, alongside what re-inits and what is snapshotted.

2. Buffers belong to a module INSTANCE, so several independent sequences need
   either a batch dimension threaded through every buffer by hand or one module
   per sequence. There is no vmap over a module's buffers, which is what
   `batch(scan(...))` does in one line in the nodejax file.

Run directly:  python -m examples.comparisons.chunk.chunk_torch
"""

import jax
import os

# jax and torch each bring their own OpenMP, and importing both in one process
# aborts on macOS without this. Documented as unsupported; universally used.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn as nn

from examples.comparisons.chunk import chunk_common as task
from nodejax.core.types import PyTree


def _t(a: jax.Array):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


WX, WH, B, SEQ = _t(task.WX), _t(task.WH), _t(task.B), _t(task.SEQ)


def carry_init():
    """The drawn carry, torch's way: no key in sight.

    torch.randn reads the global generator and advances it, so nothing is
    threaded, nothing is split, and no argument anywhere mentions entropy.
    That is the shortest column in the table by a distance, and it is the same
    property as the leak in run_recordings_forgetting_the_reset: the state
    that makes it convenient is the state nobody passed.

    The scale matches the other three, so at INIT_SCALE=0 this agrees with
    them; above zero it cannot, torch's generator not being jax's."""
    return torch.randn(task.H) * task.INIT_SCALE


class Norm(nn.Module):
    """Running per-feature normalizer.

    The statistics are BUFFERS, so they update in place and survive every call
    without anyone saying so."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(task.W))
        self.register_buffer('var', torch.ones(task.W))

    def forward(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + task.EPS)
        m = task.MOMENTUM
        self.mean = (1 - m) * self.mean + m * x
        self.var = (1 - m) * self.var + m * (x - self.mean) ** 2
        return out


class RNN(nn.Module):
    """The recurrent cell, with WEIGHTS.

    Its hidden state is NOT a buffer and could not be: the loop carries it, so
    it is an argument in and a value out. Two kinds of state in one model, and
    which kind a piece goes in is decided by whether the loop happens to carry
    it, not by anything about what it means.

    The weights ARE the buffers' neighbours: nn.Parameter and register_buffer
    put things in the same state_dict, differing in requires_grad. So adding
    weights costs nothing here, and the reason is the same one that makes the
    forgotten re-init silent: everything the module owns is in one bag that
    travels with it.

    No key reaches this constructor, because there is none to reach it. Every
    other column routes one to a parameter initializer and that routing is
    what their rows record; torch initializes from its own generator, or here
    from the shared values, and says nothing about entropy at all."""

    def __init__(self):
        super().__init__()
        wx, wh, b = task.weights()          # no key: torch has none to pass
        self.wx = nn.Parameter(_t(wx))
        self.wh = nn.Parameter(_t(wh))
        self.b = nn.Parameter(_t(b))

    def forward(self, hidden, x):
        return torch.tanh(x @ self.wx + hidden @ self.wh + self.b)


class Step(nn.Module):
    """The two composed, the ordinary way: submodules on a parent.

    Composing costs a level of PATH. The statistics were model.mean; they are
    model.norm.mean now, and every line that reaches for them from outside
    grows to match."""

    def __init__(self):
        super().__init__()
        self.norm = Norm()
        self.rnn = RNN()

    def forward(self, hidden, x):
        return self.rnn(hidden, self.norm(x))


def run():
    """One recording in chunks. The statistics need no help to cross a chunk
    boundary; the hidden state is threaded because it is an ordinary value."""
    model, hidden, outs = Step(), carry_init(), []
    for chunk in SEQ.reshape(-1, task.CHUNK, task.W):
        for x in chunk:
            hidden = model(hidden, x)
            outs.append(hidden)
        hidden = hidden.detach()          # truncated BPTT: the graph stops here
    return torch.stack(outs).detach().numpy()


def run_recordings():
    """TWO nested loops, and what says which lifetime belongs to which: one
    line in one of them. The carry re-inits because `carry_init()` is at the
    top of the recording loop. The statistics cross everything because no
    line re-inits them, a fact stated by absence. Neither is checkable, and
    moving the line one loop up or down leaves a program that runs and is
    wrong."""
    model, outs = Step(), []
    shape = (task.RECORDINGS, task.PER_RECORDING, task.CHUNK, task.W)
    for recording in SEQ.reshape(shape):
        hidden = carry_init()                     # RECORDING: the hidden dies
        for chunk in recording:
            for x in chunk:
                hidden = model(hidden, x)
                outs.append(hidden)
            hidden = hidden.detach()
    return torch.stack(outs).detach().numpy()


def run_recordings_forgetting_the_reset():
    """The same function with the carry re-init hoisted one loop out, which is
    the one-line mistake this default invites.

    It runs. It produces the right shape. It silently computes the LIVE
    answer, the whole sequence as one recording: another row of this very
    table, arrived at by moving a line. Checked in main(), because a claim
    about a failure mode is worth executing."""
    model, outs = Step(), []
    hidden = carry_init()                                     # hoisted: the bug
    shape = (task.RECORDINGS, task.PER_RECORDING, task.CHUNK, task.W)
    for recording in SEQ.reshape(shape):
        for chunk in recording:
            for x in chunk:
                hidden = model(hidden, x)
                outs.append(hidden)
            hidden = hidden.detach()
    return torch.stack(outs).detach().numpy()


def train(recordings: PyTree=False):
    """FOURTH question: train the weights through the chunked rollout.

    Torch's shortest row and its most implicit. Nothing here names a
    parameter: nn.Parameter set requires_grad when the module was built,
    backward follows it, and the optimizer was handed model.parameters() once
    and holds references by identity ever after. Three separate mechanisms
    agreeing without a line saying they agree.

    The buffers have to go back to cold BY HAND, and this is the third time
    the same property has cost the same thing. They are not trained, but they
    do carry, so a step that does not reset them starts where the last one
    finished. It does not raise, the loss still falls, and only the reference
    says otherwise."""
    model = Step()
    opt = torch.optim.SGD(model.parameters(), lr=float(task.LR))
    target = _t(task.TARGET)
    cold_hidden = _t(np.asarray(task.carry_init(task.INIT_KEY)))
    losses = []
    for _ in range(task.TRAIN_STEPS):
        model.norm.mean = _t(np.zeros(task.W))      # not trained, but carried
        model.norm.var = _t(np.ones(task.W))
        hidden, outs = cold_hidden, []

        def play(chunks):
            """one recording's worth: the CHUNK boundary is the clone at the
            top of this loop, and nothing relates it to the identical line in
            run(). Training the lifetimes means writing them again"""
            nonlocal hidden
            for chunk in chunks:
                for x in chunk:
                    hidden = model(hidden, x)
                    outs.append(hidden)

        if recordings:
            shape = (task.RECORDINGS, task.PER_RECORDING, task.CHUNK, task.W)
            for recording in SEQ.reshape(shape):
                hidden = cold_hidden                    # the RECORDING boundary
                play(recording)
        else:
            play(SEQ.reshape(-1, task.CHUNK, task.W))
        loss = ((torch.stack(outs) - target) ** 2).mean()
        losses.append(loss.detach().numpy())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return np.stack(losses)



def train_sessions():
    """FIFTH question: recording and session in one loop nest.

    Torch's spelling of two lifetimes is two reset lines in two loops, and
    the third lifetime, the weights, is the absence of a third line. The
    detach at the chunk boundary applies to EVERY carried tensor, the hidden
    and the statistics alike, because whatever carries carries its graph:
    forget the statistics' detach and backward reaches into earlier chunks,
    with nothing raised."""
    model = Step()
    xs, ts = _t(task.SESS_SEQ), _t(task.SESS_TARGET)
    cold_hidden = _t(np.asarray(task.carry_init(task.INIT_KEY)))
    losses = []
    for si in range(task.SESSIONS):
        model.norm.mean = _t(np.zeros(task.W))          # SESSION: recalibrated
        model.norm.var = _t(np.ones(task.W))
        # SESSION: the momentum restarts, spelled as a FRESH OPTIMIZER, since
        # the buffers live inside the object and there is no smaller thing to
        # replace than the object itself
        opt = torch.optim.SGD(model.parameters(), lr=float(task.LR),
                              momentum=float(task.MU))
        for r in range(task.RECORDINGS):
            hidden = cold_hidden                        # RECORDING: the hidden dies
            for c in range(task.PER_RECORDING):
                outs = []
                for x in xs[si, r, c]:
                    hidden = model(hidden, x)
                    outs.append(hidden)
                loss = ((torch.stack(outs) - ts[si, r, c]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.detach().numpy())
                # CHUNK: the graph truncates here; EVERY carried tensor
                # detaches, the statistics as much as the hidden, or backward
                # reaches into earlier chunks with nothing raised
                hidden = hidden.detach()
                model.norm.mean = model.norm.mean.detach()
                model.norm.var = model.norm.var.detach()
    return np.stack(losses)


def main() -> None:
    ref = lambda r: np.asarray(r)
    task.report('torch',
                live_ok=bool(np.allclose(run(), ref(task.reference_live()), atol=1e-5)),
                two_ok=bool(np.allclose(run_recordings(),
                                        ref(task.reference_recordings()), atol=1e-5)),
                train_ok=bool(np.allclose(train(), ref(task.reference_trained()),
                                          atol=1e-4)),
                train_two_ok=bool(np.allclose(
                    train(recordings=True),
                    ref(task.reference_trained_recordings()), atol=1e-4)),
                train_tags_ok=bool(np.allclose(
                    train_sessions(),
                    ref(task.reference_trained_sessions()), atol=1e-4)),
                cost={
                    'hidden across chunks:': 'threaded by the caller, plus a detach',
                    'stats across chunks:': 'nothing: buffers mutate and survive',
                    'model edited:': 'no',
                    'slots named outside:': 'model.norm.mean/.var, at every boundary',
                    'carry re-inits per recording:': 'a line the caller must remember',
                    'the carry is DRAWN:': 'torch.randn; nothing threaded at all',
                    'the cell has WEIGHTS:': 'nn.Parameter, beside the buffers',
                    'stats cross both:': 'by default, whether you meant it or not',
                    'training state lives:': 'an optimizer object, keyed by identity',
                    'lifetimes under training:': 'the caller writes both loops again',
                    'params kept out of the gradient:': 'requires_grad, set when built',
                    'state reset between steps:': 'BY HAND; it carries, and silently',
                    'recording and session, trained:': 'two resets in two loops; detach EVERY carried tensor',
                    'the momentum restarts per session:': 'a fresh optimizer object each session',
                })

    # the failure mode this default invites, executed rather than asserted
    try:
        leaked, raised = run_recordings_forgetting_the_reset(), False
    except Exception:
        leaked, raised = None, True
    leaks = leaked is not None and bool(
        np.allclose(leaked, np.asarray(task.reference_live()), atol=1e-5))
    print(f'  hoisting the re-init raises:         {raised}')
    print(f'  ...and silently answers the LIVE question: {leaks}')


if __name__ == '__main__':
    main()
