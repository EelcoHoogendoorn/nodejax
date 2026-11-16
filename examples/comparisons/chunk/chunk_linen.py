"""Multi-horizon state lifetimes in Flax Linen.

Kept alongside `chunk_nnx.py` to record the four failure modes of Linen's lifted `nn.scan` API on nested stateful horizons:

1. Every `nn.scan` call site restates `variable_carry='stats'` and `variable_broadcast` policies rather than reading lifetime rules off `Norm`.
2. `nn.scan` crashes with a PyTree carry mismatch if a state collection is created lazily on step 1 rather than pre-populated before scanning.
3. `Module.init` executes one step of the body, advancing running statistics during initialization unless the caller constructs the initial `'stats'` dictionary manually.
4. Freezing statistics per chunk forces the enclosing module to read `get_variable('stats', 'step')['norm']` by unchecked string path, silently returning default zeros and ones if the submodule path changes.

Run directly with `python -m examples.comparisons.chunk.chunk_linen`.
"""

import jax
import jax.numpy as jnp
import flax.linen as fnn

from examples.comparisons.chunk import chunk_common as task


class Norm(fnn.Module):
    """Normalizes features using running statistics stored in a Linen `'stats'`
    collection so lifted scans can carry them across boundaries."""

    @fnn.compact
    def __call__(self, x):
        mean = self.variable('stats', 'mean', lambda: jnp.zeros(task.W))
        var = self.variable('stats', 'var', lambda: jnp.ones(task.W))
        out = task.normalize(x, mean.value, var.value)
        mean.value, var.value = task.update_stats(x, mean.value, var.value)
        return out


class RNN(fnn.Module):
    """Applies the recurrent cell with `'params'` weights while taking `hidden`
    as an explicit scan carry. Two kinds of state
    live in one model, and which kind a piece goes in is decided by whether the
    scan carries it.

    `self.param` hands the initializer a key, which flax derives from the one
    given to init and the module's path. The weights land in the `params`
    collection, and a collection is a thing every lift has to be told about:
    see the scans below, which now say `variable_broadcast` as well as
    `variable_carry`."""

    @fnn.compact
    def __call__(self, hidden, x):
        wx = self.param('wx', lambda k: task.weights(k)[0])
        wh = self.param('wh', lambda k: task.weights(k)[1])
        b = self.param('b', lambda k: task.weights(k)[2])
        return task.cell(hidden, x, wx, wh, b)


class Step(fnn.Module):
    """Composes `Norm` and `RNN` inside a `@nn.compact` step body.

    Composing adds a level of string path that the caller has to
    write down. The hand-built variables below mirror the module nesting
    exactly, so a submodule adds a level to a dict literal three functions
    away."""

    @fnn.compact
    def __call__(self, hidden, x):
        h = RNN(name='rnn')(hidden, Norm(name='norm')(x))
        return h, h                               # a scan body: carry, output


# the inner scan, over the steps of one chunk. The annotation says how the
# stats collection crosses the time axis: carried
Inner = fnn.scan(Step, variable_carry='stats', variable_broadcast='params',
                 split_rngs={'params': False},
                 in_axes=0, out_axes=0)


class Chunk(fnn.Module):
    """Wraps the inner step scan in a named module so outer scans can lift it."""

    @fnn.compact
    def __call__(self, hidden, chunk):
        return Inner(name='step')(hidden, chunk)


def run(seq: jax.Array = task.SEQ):
    """Runs nested `nn.scan` over chunks and steps with policies restated at
    both levels.

    The variables are built by hand. `init` cannot produce them: a collection
    cannot be carried before it exists, so init through the nested scans fails
    on a carry-structure mismatch, and `init` also executes the module, so it
    would hand back statistics already advanced by a step."""
    outer = fnn.scan(Chunk, variable_carry='stats', variable_broadcast='params',
                     split_rngs={'params': False},
                     in_axes=0, out_axes=0)()
    hidden, mean, var = task.cold()
    # the path gained a level when Norm became a submodule, and this literal
    # is three functions away from the line that added it
    wx, wh, b = task.weights()
    variables = {'stats': {'step': {'norm': {'mean': mean, 'var': var}}},
                 'params': {'step': {'rnn': {'wx': wx, 'wh': wh, 'b': b}}}}
    (_, outs), _ = outer.apply(variables, hidden,
                               seq.reshape(-1, task.CHUNK, task.W), mutable=['stats'])
    return outs.reshape(-1, task.H)


# a recording is the chunk-scan again, one level up
OverChunks = fnn.scan(Chunk, variable_carry='stats', variable_broadcast='params',
                      split_rngs={'params': False},
                      in_axes=0, out_axes=0)


class Recording(fnn.Module):
    """Runs one recording while re-initializing `hidden` inside the module body
    via `make_rng('carry')` rather than declaring a lifetime on `RNN`.

    `make_rng('carry')` is the draw, and it is only different per recording
    because the scan above was told `split_rngs={'carry': True}`. That flag
    lives at the lift, three lines away from the draw and in a different
    object, and getting it wrong gives every recording the same carry with
    nothing raised."""

    @fnn.compact
    def __call__(self, _, chunks):
        hidden = task.carry_init(self.make_rng('carry'))   # the recording boundary
        _, outs = OverChunks(name='rec')(hidden, chunks)
        return None, outs


def run_recordings(seq: jax.Array = task.SEQ):
    """Runs nested lifted scans with `split_rngs={'carry': True}` at the outer
    lift and an in-body reset inside `Recording`.

    Three lifted scans each restate `variable_carry='stats'`, plus a
    third annotation on the outer one: `split_rngs={'carry': True}`, which is
    what makes the draw inside the module body differ per recording. The flag
    is at the lift and the draw is in the body, and nothing connects them but
    the string 'carry'. Drop the flag and every recording starts identically,
    silently.

    The recording re-init is a local variable in a module body. Nothing
    declares it, and nothing says which lifetime it belongs to: move the line
    one module up or down and the program runs and is wrong.

    The hand-built variables must mirror the module nesting exactly, so the
    caller writes a path through a tree it does not own. Nothing checks that
    path: name a level wrong and it is silently a different collection, which
    is how several attempts at this failed before the names were read off the
    module structure rather than guessed."""
    outer = fnn.scan(Recording, variable_carry='stats',
                     variable_broadcast='params',         # and now params too
                     split_rngs={'carry': True,           # a key per recording
                                 'params': False},        # with one set of weights
                     in_axes=0, out_axes=0)()
    _, mean, var = task.cold()
    wx, wh, b = task.weights()
    variables = {'stats': {'rec': {'step': {'norm': {'mean': mean, 'var': var}}}},
                 'params': {'rec': {'step': {'rnn': {'wx': wx, 'wh': wh, 'b': b}}}}}
    (_, outs), _ = outer.apply(variables, None, task.recordings(seq),
                               rngs={'carry': task.INIT_KEY}, mutable=['stats'])
    return outs.reshape(-1, task.H)


def train(recordings: bool = False, seq: jax.Array = task.SEQ):
    """Trains the weights through the chunked rollout with isolated `'params'`
    and `'stats'` variable collections.

    `params` and `stats` are separate trees, so differentiating with respect to
    `params` cannot reach `stats` and nothing extra has to filter them apart.

    The `mutable=['stats']` argument on `apply` serves both the state updates
    and the gradient boundary."""
    hidden, mean, var = task.cold()
    wx, wh, b = task.weights()
    weights = {'rnn': {'wx': wx, 'wh': wh, 'b': b}}
    moments = {'norm': {'mean': mean, 'var': var}}

    # the whole setup changes with the lifetime, because each is a different
    # lifted scan over a different module with a differently nested variable
    # tree the caller writes by hand. Training them is training three
    # arrangements, not one arrangement three ways
    if recordings:
        outer = fnn.scan(Recording, variable_carry='stats',
                         variable_broadcast='params',
                         split_rngs={'carry': True, 'params': False},
                         in_axes=0, out_axes=0)()
        params = {'rec': {'step': weights}}
        stats = {'rec': {'step': moments}}
        carry0, data = None, task.recordings(seq)
        rngs = {'carry': task.INIT_KEY}
    else:
        outer = fnn.scan(Chunk, variable_carry='stats',
                         variable_broadcast='params',
                         split_rngs={'params': False},
                         in_axes=0, out_axes=0)()
        params = {'step': weights}
        stats = {'step': moments}
        carry0, data = hidden, seq.reshape(-1, task.CHUNK, task.W)
        rngs = {}

    def loss_fn(params):
        (_, outs), _ = outer.apply({'params': params, 'stats': stats}, carry0,
                                   data, rngs=rngs, mutable=['stats'])
        return task.loss_of(outs.reshape(-1, task.H))

    # the loop is a lax.scan, because that is what a user of a pure-functional
    # framework writes: the training state is a carry like any other, and a
    # Python loop over jax code is not something a real user would put up with
    def step(params, _):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        # hand-rolled SGD: these frameworks own no optimizer, optax being
        # the ecosystem's, and this loop owns its own arithmetic instead
        return jax.tree.map(lambda w, g: w - task.LR * g, params, grads), loss

    _, losses = jax.lax.scan(step, params, None, length=task.TRAIN_STEPS)
    return losses


def train_sessions():
    """Trains across recording and session boundaries in three nested scans.

    Three lax.scans the caller owns, as in haiku and equinox, plus linen's
    own tax: the variable trees are threaded by hand through every level,
    and the paths in them mirror a module nesting the caller does not own."""
    wx, wh, b = task.weights()
    params0 = {'step': {'rnn': {'wx': wx, 'wh': wh, 'b': b}}}
    _, mean, var = task.cold()
    stats0 = {'step': {'norm': {'mean': mean, 'var': var}}}
    hidden0, chunk_mod = task.carry_init(task.INIT_KEY), Chunk()

    def over_chunks(carry, xt):
        params, vel, stats, hidden = carry
        x, t = xt

        def loss_fn(p):
            (h2, outs), mut = chunk_mod.apply(
                {'params': p, 'stats': stats}, hidden, x, mutable=['stats'])
            return jnp.mean((outs - t) ** 2), (h2, mut['stats'])

        (loss, (h2, s2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        # hand-rolled momentum, same note: the velocity is the caller's
        # pytree because nothing in the framework owns the training loop
        vel = jax.tree.map(lambda v, g: task.MU * v + g, vel, grads)
        params = jax.tree.map(lambda w, v: w - task.LR * v, params, vel)
        return (params, vel, s2, h2), loss

    def over_recordings(carry, xt):
        params, vel, stats = carry
        (params, vel, stats, _), losses = jax.lax.scan(
            over_chunks, (params, vel, stats, hidden0), xt)   # recording: hidden dies
        return (params, vel, stats), losses

    def over_sessions(params, xt):
        vel0 = jax.tree.map(jnp.zeros_like, params)   # session: momentum restarts
        (params, _, _), losses = jax.lax.scan(
            over_recordings, (params, vel0, stats0), xt)       # session: recalibrated
        return params, losses

    _, losses = jax.lax.scan(over_sessions, params0,
                             (task.SESS_SEQ, task.SESS_TARGET))
    return losses.reshape(-1)


def main() -> None:
    task.report('flax linen',
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
                    'stats across chunks:': "variable_carry='stats', restated per level",
                    'model edited:': 'no',
                    'slots named outside:': "the whole collection path, built by hand",
                    'carry re-inits per recording:': 'a re-init inside a module body',
                    'the carry is DRAWN:': "split_rngs at the lift, make_rng in the body",
                    'the cell has WEIGHTS:': 'variable_broadcast at ALL FOUR lifts',
                    'stats cross both:': "variable_carry='stats' at all three levels",
                    'training state lives:': 'a tree the caller holds, beside the rest',
                    'lifetimes under training:': 'a different lifted scan and variable tree each',
                    'recording and session, trained:': 'three nested scans, variable trees threaded by hand',
                    'params kept out of the gradient:': 'nothing: they are their own collection',
                })


if __name__ == '__main__':
    main()
