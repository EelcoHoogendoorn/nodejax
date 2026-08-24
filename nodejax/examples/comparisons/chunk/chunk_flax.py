"""Chunked sequences in flax nnx: state on the object, annotated by TYPE.

Same task, same numbers, same references as `chunk_nodejax.py`. nnx.scan over
chunks of nnx.scan over steps, so the comparison is about what the spelling
cost.

Read beside `chunk_flax_linen.py`, which is the same task in flax's older
lifted-transform API. Most of what that file spends its lines on is gone here,
and the difference is worth as much as the difference from the other
frameworks:

  NO HAND-BUILT VARIABLE TREE. The module owns its state and is constructed
  the ordinary way. Linen could not init through the nested scans at all, so
  the caller wrote out a dict mirroring the module nesting, at a path nothing
  checked, and every submodule added a level to a literal in another function.

  ANNOTATED BY TYPE, NOT BY NAME. One `nnx.StateAxes` says what happens to
  nnx.Param and what happens to nnx.Variable. Linen named string collections
  at every lift, four of them, and a missed one failed with a path rather than
  a missing annotation.

WHAT IT STILL COSTS, and this is the part the API change does not touch:

  THE ANNOTATION IS PER LIFT. Both scans repeat the same StateAxes, and each
  one has to be told again that params broadcast and state carries. It is one
  line rather than three, and it is still a line per lift.

  THE BOUNDARY REACHES IN. Holding the statistics for a chunk means reading
  `model.norm.mean` from the chunk body and passing it down, because a
  snapshot is a value and values are not what the module mechanism carries.
  The parent knows where its child keeps its state, exactly as before.

  THE TWO LIFETIMES ARE STILL TWO KINDS OF THING. The statistics are module
  state; the hidden carry is the scan's carry, an ordinary value. Which one a
  piece goes in is decided by whether the scan carries it, and separating the
  recording lifetime from the chunk lifetime is a line in a body either way.

Run directly:  python -m nodejax.examples.comparisons.chunk.chunk_flax
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nodejax.examples.comparisons.chunk import chunk_common as task

# Params broadcast, everything else carries. One annotation, by TYPE, and the
# only thing either scan has to be told.
AXES = nnx.StateAxes({nnx.Param: None, nnx.Variable: nnx.Carry})


class Norm(nnx.Module):
    """Running per-feature normalizer. Its statistics are nnx.Variable, which
    is module state: mutated in place, and crossing a scan because the lift
    says nnx.Variable carries."""

    def __init__(self):
        _, mean, var = task.cold()
        self.mean, self.var = nnx.Variable(mean), nnx.Variable(var)

    def __call__(self, x):
        out = task.normalize(x, self.mean[...], self.var[...])
        self.mean[...], self.var[...] = task.update_stats(
            x, self.mean[...], self.var[...])
        return out


class RNN(nnx.Module):
    """The recurrent cell, with WEIGHTS. They are nnx.Param, which is what the
    lift broadcasts rather than carries, and the key arrives through nnx.Rngs
    at construction.

    Its hidden state is NOT module state: the scan carries it, in and out.
    Two kinds of state in one model, and which kind a piece goes in is decided
    by whether the scan carries it."""

    def __init__(self, rngs: nnx.Rngs):
        wx, wh, b = task.weights(rngs.params())
        self.wx, self.wh, self.b = nnx.Param(wx), nnx.Param(wh), nnx.Param(b)

    def __call__(self, hidden, x):
        return task.cell(hidden, x, self.wx[...], self.wh[...], self.b[...])


class Step(nnx.Module):
    """The two composed, the ordinary way: submodules as attributes.

    Composing costs nothing here, which is the clearest single difference from
    the linen file: there, making Norm a submodule added a level to two dict
    literals in other functions and broke a path fetch that returned defaults
    instead of raising."""

    def __init__(self, rngs: nnx.Rngs):
        self.norm, self.rnn = Norm(), RNN(rngs)

    def __call__(self, hidden, x):
        return self.rnn(hidden, self.norm(x))


def _model():
    return Step(nnx.Rngs(params=task.PARAM_KEY))


def run(seq: jax.Array=task.SEQ):
    """The whole chunked run: nnx.scan over chunks of nnx.scan over steps."""
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


def run_recordings(seq: jax.Array=task.SEQ):
    """TWO boundaries, and a piece of state answering each.

    Both are lines in bodies. The recording one rebuilds the carry, the chunk
    one reads the child's statistics; nothing declares either, and nothing
    relates them to the lifetimes they are supposed to have."""
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
        hidden = task.carry_init(task.INIT_KEY)         # the RECORDING boundary
        _, outs = over_chunks(hidden, m, chunks)
        return None, outs

    _, outs = over_recordings(None, model, task.recordings(seq))
    return outs.reshape(-1, task.H)


def train(recordings: PyTree=False, seq: jax.Array=task.SEQ):
    """FOURTH question: train the weights through the chunked rollout.

    Two things nnx does differently here, one good and one sharp.

    THE GOOD: nnx.value_and_grad differentiates with respect to nnx.Param and
    nothing else, by TYPE, so the statistics sitting in the same object are
    not reached and nothing had to say so. Same annotation vocabulary as the
    scans, doing the same job for a different transform.

    THE SHARP: the module is MUTABLE, so its statistics survive from one
    training step to the next. The reference defines a step as an independent
    rollout from cold, so this needs the reset written out, and forgetting it
    does not raise: the loss still falls, to 0.3092 instead of 0.3114. That is
    torch's leak, in a jax framework, and it arrives with reference semantics
    rather than with buffers."""
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
            hidden = task.carry_init(task.INIT_KEY)      # the RECORDING boundary
            _, outs = over_chunks(hidden, mm, groups)
            return None, outs

        _, outs = over_recordings(None, m, task.recordings(seq))
        return outs.reshape(-1, task.H)

    @nnx.value_and_grad
    def loss_fn(m):
        return task.loss_of(rollout(m))

    # the framework's own optimizer, under the framework's own scan: model
    # and nnx.Optimizer ride as two carried objects, since the optimizer
    # stopped holding the model in flax 0.11. For TRAINING the params must
    # CARRY, so this lift's StateAxes says everything does, where the
    # rollout scans broadcast them
    optimizer = nnx.Optimizer(model, optax.sgd(task.LR), wrt=nnx.Param)
    train_axes = nnx.StateAxes({...: nnx.Carry})

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def train_loop(m, opt, _):
        # STEP: each step is an independent rollout from cold, so the stats
        # reset here; forget it and the loss still falls, to the wrong number
        m.norm.mean[...], m.norm.var[...] = cold_mean, cold_var
        loss, grads = loss_fn(m)
        opt.update(m, grads)
        return loss

    return train_loop(model, optimizer, jnp.arange(task.TRAIN_STEPS))



def train_sessions():
    """FIFTH question: recording and session in one scan nest.

    Three nested nnx.scans, the same shape as everyone else's, with the model
    and the framework's own nnx.Optimizer riding as two carried objects. The
    lifetimes are in-body lines, and the optimizer's restart is nnx at its
    most nnx: its state selected BY TYPE and zeroed, the same vocabulary that
    picks what trains and what carries everywhere else in this file."""
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
        # RECORDING: the hidden dies; a new recording is a new signal
        _, losses = over_chunks(hidden0, m, opt, xt)
        return losses

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def over_sessions(m, opt, xt):
        # SESSION: the calibration re-measures from cold
        m.norm.mean[...], m.norm.var[...] = cold_mean, cold_var
        # SESSION: the optimizer state restarts, selected by TYPE and zeroed
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
