"""Multi-horizon state lifetimes in Flax NNX.

Compared against `chunk_linen.py`, NNX avoids Linen's hand-built initialization dictionary and string collection names by storing running statistics on `Norm` as `nnx.Variable` attributes and annotating `nnx.scan` via `nnx.StateAxes`. Three structural costs remain:

1. Every `nnx.scan` site repeats the `StateAxes` annotation so parameters broadcast while variables carry.
2. Freezing normalization statistics across a chunk requires the outer chunk body to reach into `model.norm.mean` and `model.norm.var` and pass snapshots back down.
3. Recurrent hidden state rides the functional `nnx.scan` carry while normalization statistics mutate on the module, so changing `Norm`'s reset horizon to `'recording'` requires two independent edits across the inference and training rollout loops.

Run directly with `python -m examples.comparisons.chunk.chunk_nnx`.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from examples.comparisons.chunk import chunk_common as task

# Params broadcast, everything else carries. One annotation by Variable type,
# reused across the rollout scans.
AXES = nnx.StateAxes({nnx.Param: None, nnx.Variable: nnx.Carry})


class Norm(nnx.Module):
    """Normalizes features using mutable `nnx.Variable` running statistics that
    carry across scans when `AXES` maps `nnx.Variable` to `nnx.Carry`."""

    def __init__(self):
        _, mean, var = task.cold()
        self.mean, self.var = nnx.Variable(mean), nnx.Variable(var)

    def __call__(self, x):
        out = task.normalize(x, self.mean[...], self.var[...])
        self.mean[...], self.var[...] = task.update_stats(
            x, self.mean[...], self.var[...])
        return out


class RNN(nnx.Module):
    """Applies the recurrent cell with `nnx.Param` weights broadcast by `AXES`
    and initialized via `nnx.Rngs` at construction.

    Its hidden state is not module state: the scan carries it in and out.
    Two kinds of state live in one model, and which kind a piece goes in is
    decided by whether the scan carries it."""

    def __init__(self, rngs: nnx.Rngs):
        wx, wh, b = task.weights(rngs.params())
        self.wx, self.wh, self.b = nnx.Param(wx), nnx.Param(wh), nnx.Param(b)

    def __call__(self, hidden, x):
        return task.cell(hidden, x, self.wx[...], self.wh[...], self.b[...])


class Step(nnx.Module):
    """Composes `Norm` and `RNN` as submodule attributes on one step module.

    Composing requires no manual dictionary nesting here, unlike `chunk_linen`:
    there, making `Norm` a submodule added a level to two dict literals in
    other functions and broke a path fetch that returned defaults instead of
    raising."""

    def __init__(self, rngs: nnx.Rngs):
        self.norm, self.rnn = Norm(), RNN(rngs)

    def __call__(self, hidden, x):
        return self.rnn(hidden, self.norm(x))


def _model():
    return Step(nnx.Rngs(params=task.PARAM_KEY))


def run(seq: jax.Array = task.SEQ):
    """Runs the chunked sequence as nested `nnx.scan` over chunks and steps."""
    model = _model()

    @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
    def over_steps(hidden, m, x):
        h = m(hidden, x)
        return h, h

    @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
    def over_chunks(hidden, m, xs):
        return over_steps(hidden, m, xs)

    hidden = task.carry_init(task.INIT_KEY)
    _, outs = over_chunks(hidden, model, seq.reshape(-1, task.CHUNK, task.W))
    return outs.reshape(-1, task.H)


def run_recordings(seq: jax.Array = task.SEQ):
    """Runs nested `nnx.scan` loops over recordings, chunks, and steps.

    Both boundaries are lines inside loop bodies. The recording body rebuilds
    the carry while `Norm` statistics carry via `AXES`; nothing declares either
    lifetime on the module itself."""
    model = _model()

    @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
    def over_steps(hidden, m, x):
        h = m(hidden, x)
        return h, h

    @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
    def over_chunks(hidden, m, xs):
        return over_steps(hidden, m, xs)

    @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
    def over_recordings(_, m, chunks):
        hidden = task.carry_init(task.INIT_KEY)         # the recording boundary
        _, outs = over_chunks(hidden, m, chunks)
        return None, outs

    _, outs = over_recordings(None, model, task.recordings(seq))
    return outs.reshape(-1, task.H)


def train(recordings: bool = False, seq: jax.Array = task.SEQ):
    """Trains the weights through the chunked rollout with `nnx.Optimizer`.

    Two NNX properties show up here:

    1. `nnx.value_and_grad` differentiates with respect to `nnx.Param` by
       Variable type, leaving `nnx.Variable` statistics on the same module
       untouched without manual filtering.

    2. Because the module is mutable, its running statistics persist from one
       training step to the next unless reset manually. Each training step here
       is an independent rollout from cold state, and forgetting the manual
       reset does not raise: the loss still falls, to `0.3092` instead of
       `0.3114`, leaking state across steps via reference mutation."""
    model = _model()
    _, cold_mean, cold_var = task.cold()
    chunks = seq.reshape(-1, task.CHUNK, task.W)

    def rollout(m):
        @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
        def over_steps(hidden, mm, x):
            h = mm(hidden, x)
            return h, h

        @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
        def over_chunks(hidden, mm, xs):
            return over_steps(hidden, mm, xs)

        if not recordings:
            _, outs = over_chunks(task.carry_init(task.INIT_KEY), m, chunks)
            return outs.reshape(-1, task.H)

        @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
        def over_recordings(_, mm, groups):
            hidden = task.carry_init(task.INIT_KEY)      # the recording boundary
            _, outs = over_chunks(hidden, mm, groups)
            return None, outs

        _, outs = over_recordings(None, m, task.recordings(seq))
        return outs.reshape(-1, task.H)

    @nnx.value_and_grad
    def loss_fn(m):
        return task.loss_of(rollout(m))

    # the framework's own optimizer, under the framework's own scan: model
    # and nnx.Optimizer ride as two carried objects, since the optimizer
    # stopped holding the model in flax 0.11. During training the params must
    # carry, so this lift's StateAxes says everything does, where the
    # rollout scans broadcast them
    optimizer = nnx.Optimizer(model, optax.sgd(task.LR), wrt=nnx.Param)
    train_axes = nnx.StateAxes({...: nnx.Carry})

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def train_loop(m, opt, _):
        # each step is an independent rollout from cold, so the stats
        # reset here; forget it and the loss still falls, to the wrong number
        m.norm.mean[...], m.norm.var[...] = cold_mean, cold_var
        loss, grads = loss_fn(m)
        opt.update(m, grads)
        return loss

    return train_loop(model, optimizer, jnp.arange(task.TRAIN_STEPS))


def train_sessions():
    """Trains across recording and session boundaries in three nested scans.

    Three nested `nnx.scan` loops carry the model and `nnx.Optimizer` as two
    carried objects. The lifetime resets are in-body lines, and the optimizer
    momentum reset selects `OptState` by Variable type and zeroes it."""
    from flax.nnx.training.optimizer import OptState

    model = _model()
    optimizer = nnx.Optimizer(model, optax.sgd(task.LR, momentum=task.MU),
                              wrt=nnx.Param)
    _, cold_mean, cold_var = task.cold()
    hidden0 = task.carry_init(task.INIT_KEY)
    train_axes = nnx.StateAxes({...: nnx.Carry})

    @nnx.scan(in_axes=(nnx.Carry, train_axes, train_axes, 0),
              out_axes=(nnx.Carry, 0))
    def over_chunks(hidden, m, opt, xt):
        x, t = xt

        @nnx.scan(in_axes=(nnx.Carry, AXES, 0), out_axes=(nnx.Carry, 0))
        def over_steps(h, mm, xi):
            h2 = mm(h, xi)
            return h2, h2

        def loss_fn(mm):
            h2, outs = over_steps(hidden, mm, x)
            return jnp.mean((outs - t) ** 2), h2

        (loss, h2), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
        opt.update(m, grads)
        return h2, loss

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def over_recordings(m, opt, xt):
        # recording: the hidden dies; a new recording is a new signal
        _, losses = over_chunks(hidden0, m, opt, xt)
        return losses

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def over_sessions(m, opt, xt):
        # session: the calibration re-measures from cold
        m.norm.mean[...], m.norm.var[...] = cold_mean, cold_var
        # session: the optimizer state restarts, selected by type and zeroed
        nnx.update(opt, jax.tree.map(jnp.zeros_like, nnx.state(opt, OptState)))
        return over_recordings(m, opt, xt)

    losses = over_sessions(model, optimizer, (task.SESS_SEQ, task.SESS_TARGET))
    return losses.reshape(-1)


def main() -> None:
    task.report('flax nnx',
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
                    'hidden across chunks:': 'the scan carry, at both levels',
                    'stats across chunks:': 'nnx.Variable: nnx.Carry, once per lift',
                    'model edited:': 'no',
                    'slots named outside:': 'm.norm.mean/.var, at the boundary',
                    'carry re-inits per recording:': 'a line in the recording body',
                    'the carry is DRAWN:': 'a plain call; the key is the caller\'s',
                    'the cell has WEIGHTS:': 'nnx.Param, broadcast by the same StateAxes',
                    'stats cross both:': 'the same StateAxes at every lift',
                    'training state lives:': 'on the module, mutated in place',
                    'lifetimes under training:': 'both boundaries written a second time',
                    'recording and session, trained:': 'three nested nnx.scans, resets as in-body lines',
                    'the momentum restarts per session:': 'nnx.Optimizer state, selected by TYPE and zeroed',
                    'params kept out of the gradient:': 'nothing: nnx.value_and_grad picks by type',
                    'state reset between steps:': 'BY HAND; it persists, silently',
                })


if __name__ == '__main__':
    main()
