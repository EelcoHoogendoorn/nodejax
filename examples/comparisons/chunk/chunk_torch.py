"""Chunked sequences in torch: persistence is the default, and that is the
whole difference.

Same task, same numbers, same references as `chunk_nodejax.py`. Torch is the
interesting column because its default is the inverse of every other file here.
Buffers are attributes that mutate, so state survives a call unless something
re-inits it. Nothing is declared, nothing is threaded, and carrying the running
statistics across a chunk costs zero lines.

That makes the easy question free and moves the cost onto the other one:

    stats across chunks     nothing. Buffers persist because they mutate.
    hidden across chunks    threaded by the caller, as in equinox
    carry restarts per      the caller must remember to re-init it. There is
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
   That is the truncated-BPTT idiom, and it is a third thing the boundary means
   here, alongside what re-inits and what is snapshotted.

2. Buffers belong to a module instance, so several independent sequences need
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


def _t(a: jax.Array):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


WX, WH, B, SEQ = _t(task.WX), _t(task.WH), _t(task.B), _t(task.SEQ)


def carry_init():
    """Draws the initial hidden carry from PyTorch's global RNG without an
    explicit key argument.

    `torch.randn` reads the global generator and advances it, so nothing is
    threaded, nothing is split, and no argument anywhere mentions entropy.
    That is the shortest column in the table by a distance, and it is the same
    property as the leak in `run_recordings_forgetting_the_reset`: the state
    that makes it convenient is the state nobody passed.

    The scale matches the other columns, so at `INIT_SCALE=0` this agrees with
    them; above zero it cannot, torch's generator not being jax's."""
    return torch.randn(task.H) * task.INIT_SCALE


class Norm(nn.Module):
    """Normalizes features using registered buffers that mutate in place and
    persist across calls automatically."""

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
    """Applies the recurrent cell with `nn.Parameter` weights while taking
    `hidden` as an explicit argument and return value.

    Its hidden state is not a buffer and could not be: the loop carries it, so
    it is an argument in and a value out. Two kinds of state in one model, and
    which kind a piece goes in is decided by whether the loop happens to carry
    it, not by anything about what it means.

    The weights sit beside the buffers in the same `state_dict`, differing in
    `requires_grad`. Adding weights costs nothing extra at the loop, for the
    same reason a forgotten re-init stays silent: everything the module owns
    travels on one mutable instance.

    No key reaches this constructor, because there is none to reach it. Every
    other column routes one to a parameter initializer and that routing is
    what their rows record; torch initializes from its own generator, or here
    from the shared values, and says nothing about entropy."""

    def __init__(self):
        super().__init__()
        wx, wh, b = task.weights()          # no key: torch has none to pass
        self.wx = nn.Parameter(_t(wx))
        self.wh = nn.Parameter(_t(wh))
        self.b = nn.Parameter(_t(b))

    def forward(self, hidden, x):
        return torch.tanh(x @ self.wx + hidden @ self.wh + self.b)


class Step(nn.Module):
    """Composes `Norm` and `RNN` as submodules on one parent `nn.Module`.

    Composing adds a level of attribute path: `model.mean` becomes
    `model.norm.mean`, and every external reset line grows to match."""

    def __init__(self):
        super().__init__()
        self.norm = Norm()
        self.rnn = RNN()

    def forward(self, hidden, x):
        return self.rnn(hidden, self.norm(x))


def run():
    """Runs one recording in chunks, detaching `hidden` at each chunk boundary
    while `Norm` buffers persist in place."""
    model, hidden, outs = Step(), carry_init(), []
    for chunk in SEQ.reshape(-1, task.CHUNK, task.W):
        for x in chunk:
            hidden = model(hidden, x)
            outs.append(hidden)
        hidden = hidden.detach()          # truncated BPTT: the graph stops here
    return torch.stack(outs).detach().numpy()


def run_recordings():
    """Runs nested recording and chunk loops where `carry_init()` at the top of
    the outer loop resets `hidden` per recording while `Norm` buffers persist
    by absence of a reset line. Neither lifetime is checkable, and
    moving the line one loop up or down leaves a program that runs and is
    wrong."""
    model, outs = Step(), []
    shape = (task.RECORDINGS, task.PER_RECORDING, task.CHUNK, task.W)
    for recording in SEQ.reshape(shape):
        hidden = carry_init()                     # recording: the hidden dies
        for chunk in recording:
            for x in chunk:
                hidden = model(hidden, x)
                outs.append(hidden)
            hidden = hidden.detach()
    return torch.stack(outs).detach().numpy()


def run_recordings_forgetting_the_reset():
    """Hoists `carry_init()` outside the recording loop to demonstrate the
    silent state leak invited by mutable persistence.

    It runs, produces the expected output shape, and silently computes the
    single-recording `reference_live` trajectory instead of resetting per
    recording. Checked in `main()`."""
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


def train(recordings: bool = False):
    """Trains the weights through the chunked rollout with `torch.optim.SGD`.

    `nn.Parameter` sets `requires_grad` at construction, `backward()` follows
    it, and the optimizer holds parameter references by object identity.

    The normalization buffers must be reset manually on every training step:
    they are not trained, yet they persist across calls, so omitting the reset
    starts each rollout from the previous step's statistics without raising."""
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
            """Runs one recording's chunks without detaching inside the full-
            rollout training objective, duplicating the step loop from `run`."""
            nonlocal hidden
            for chunk in chunks:
                for x in chunk:
                    hidden = model(hidden, x)
                    outs.append(hidden)

        if recordings:
            shape = (task.RECORDINGS, task.PER_RECORDING, task.CHUNK, task.W)
            for recording in SEQ.reshape(shape):
                hidden = cold_hidden                    # the recording boundary
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
    """Trains across recording and session boundaries in three nested Python loops.

    Two lifetimes are written as manual reset statements in two loops, while
    the weights persist by omitting a third reset. At the chunk boundary,
    every carried tensor (`hidden`, `mean`, and `var`) must call `.detach()`,
    or `backward()` traverses into earlier chunks without raising."""
    model = Step()
    xs, ts = _t(task.SESS_SEQ), _t(task.SESS_TARGET)
    cold_hidden = _t(np.asarray(task.carry_init(task.INIT_KEY)))
    losses = []
    for si in range(task.SESSIONS):
        model.norm.mean = _t(np.zeros(task.W))          # session: recalibrated
        model.norm.var = _t(np.ones(task.W))
        # session: the momentum restarts by constructing a fresh optimizer,
        # since momentum buffers live inside the optimizer instance
        opt = torch.optim.SGD(model.parameters(), lr=float(task.LR),
                              momentum=float(task.MU))
        for r in range(task.RECORDINGS):
            hidden = cold_hidden                        # recording: the hidden dies
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
                # chunk: truncate the autograd graph on every carried tensor
                # (`hidden`, `mean`, and `var`) so backward stops at this chunk
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
