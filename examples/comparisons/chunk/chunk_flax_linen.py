"""Chunked sequences in flax LINEN, the lifted-transform API.

Kept beside the nnx version rather than replaced, because the contrast
between them is worth as much as the contrast with the other frameworks:
nnx exists in large part to remove what this file spends its lines on.

Original opening follows.

Chunked sequences in flax linen, the whole run compiled as one program.

Same task, same numbers, same references as `chunk_nodejax.py`. nn.scan over
chunks of nn.scan over steps, so the comparison is about what the spelling
costs rather than about python loop overhead.

Four things found by making this run, none of them stylistic:

1. THE ROUTING IS RESTATED AT EVERY LIFTED SITE. Each `nn.scan` must be told
   how each kind of state crosses its axis: `variable_carry='stats'` at both
   levels, `fnn.broadcast` for the snapshot at the inner one. The model does
   not carry that knowledge; the transform sites do, and they are the places
   that must change when the model gains a new kind of state.

2. A COLLECTION CANNOT BE CARRIED BEFORE IT EXISTS. Running the scan straight
   from `init` fails with a carry pytree-structure mismatch, because the body
   creates `stats` on its first step and the carry going in has no such key.

3. `init` EXECUTES THE MODULE, so initializing through it would hand back
   statistics already advanced by one step. Between this and (2) the cold
   variables are built by hand, which means writing out the shape, the dtype,
   AND the collection path of state the model owns: `{'stats': {'step': ...}}`.

4. THE BOUNDARY REACHES INTO A GRANDCHILD. The snapshot lives in the
   normalizer's own collection, so the enclosing chunk fetches it by path,
   `get_variable('stats', 'step')['norm']`, and passes it back down as an
   argument. The parent has to know where its child's child keeps its state,
   and a wrong path returns the DEFAULT rather than raising: making Norm a
   submodule left this read silently producing zeros and ones until the
   reference disagreed.

Run directly:  python -m examples.comparisons.chunk.chunk_flax_linen
"""

import jax
import jax.numpy as jnp
import flax.linen as fnn

from examples.comparisons.chunk import chunk_common as task
from nodejax.core.types import PyTree


class Norm(fnn.Module):
    """Running per-feature normalizer. Its statistics live in a COLLECTION,
    which is what makes them cross a boundary."""

    @fnn.compact
    def __call__(self, x):
        mean = self.variable('stats', 'mean', lambda: jnp.zeros(task.W))
        var = self.variable('stats', 'var', lambda: jnp.ones(task.W))
        out = task.normalize(x, mean.value, var.value)
        mean.value, var.value = task.update_stats(x, mean.value, var.value)
        return out


class RNN(fnn.Module):
    """The recurrent cell, with WEIGHTS. Its hidden state is an ORDINARY
    VALUE, in and out, because the lifted scan carries it. Two kinds of state
    in one model, and which kind a piece goes in is decided by whether the
    scan carries it.

    `self.param` hands the initializer a key, which flax derives from the one
    given to init and the module's path. The weights land in the `params`
    collection, and a collection is a thing every lift has to be told about:
    see the scans below, which now say variable_broadcast as well as
    variable_carry."""

    @fnn.compact
    def __call__(self, hidden, x):
        wx = self.param('wx', lambda k: task.weights(k)[0])
        wh = self.param('wh', lambda k: task.weights(k)[1])
        b = self.param('b', lambda k: task.weights(k)[2])
        return task.cell(hidden, x, wx, wh, b)


class Step(fnn.Module):
    """The two composed, the ordinary way: submodules in a compact body.

    Composing costs a level of PATH, and here that path is one a caller has to
    write down. The hand-built variables below mirror the module nesting
    exactly, so a submodule adds a level to a dict literal three functions
    away."""

    @fnn.compact
    def __call__(self, hidden, x):
        h = RNN(name='rnn')(hidden, Norm(name='norm')(x))
        return h, h                               # a scan body: carry, output


# the inner scan, over the steps of one chunk. The annotation says how the
# stats collection crosses the time axis: CARRIED
Inner = fnn.scan(Step, variable_carry='stats', variable_broadcast='params',
                 split_rngs={'params': False},
                 in_axes=0, out_axes=0)


class Chunk(fnn.Module):
    """One chunk: the inner scan, named so the levels above can lift it."""

    @fnn.compact
    def __call__(self, hidden, chunk):
        return Inner(name='step')(hidden, chunk)


def run(seq: jax.Array=task.SEQ):
    """The whole chunked run as one program: nn.scan over chunks of nn.scan
    over steps, with the routing restated at both levels.

    The variables are built by hand. `init` cannot produce them: a collection
    cannot be carried before it exists, so init through the nested scans fails
    on a carry-structure mismatch, and init also EXECUTES the module, so it
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
    """One recording. The carry RE-INITS here, drawn from flax's own rng
    machinery, written inside a module body rather than declared by anything
    that owns it.

    `make_rng('carry')` is the draw, and it is only different per recording
    because the scan above was told `split_rngs={'carry': True}`. That flag
    lives at the lift, three lines away from the draw and in a different
    object, and getting it wrong gives every recording the same carry with
    nothing raised."""

    @fnn.compact
    def __call__(self, _, chunks):
        hidden = task.carry_init(self.make_rng('carry'))   # the RECORDING boundary
        _, outs = OverChunks(name='rec')(hidden, chunks)
        return None, outs


def run_recordings(seq: jax.Array=task.SEQ):
    """TWO nested lifted scans, and a line in a body saying what dies where.

    Three lifted scans, each restating `variable_carry='stats'`, and now a
    THIRD annotation on the outer one: `split_rngs={'carry': True}`, which is
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
                     split_rngs={'carry': True,           # a key PER recording
                                 'params': False},        # but ONE set of weights
                     in_axes=0, out_axes=0)()
    _, mean, var = task.cold()
    wx, wh, b = task.weights()
    variables = {'stats': {'rec': {'step': {'norm': {'mean': mean, 'var': var}}}},
                 'params': {'rec': {'step': {'rnn': {'wx': wx, 'wh': wh, 'b': b}}}}}
    (_, outs), _ = outer.apply(variables, None, task.recordings(seq),
                               rngs={'carry': task.INIT_KEY}, mutable=['stats'])
    return outs.reshape(-1, task.H)


def train(recordings: PyTree=False, seq: jax.Array=task.SEQ):
    """FOURTH question: train the weights through the chunked rollout.

    Linen's best row, and worth reading after all the others. The collection
    split that cost it everywhere else is exactly right here: `params` and
    `stats` are two trees, so differentiating with respect to one cannot reach
    the other and nothing has to say which is which.

    The `mutable=['stats']` on the apply is doing that work, and it was
    already there for the state questions. One argument, saying the same thing
    to the gradient and to the scan."""
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
        # HAND-ROLLED SGD: these frameworks own no optimizer, optax being
        # the ecosystem's, and this loop owns its own arithmetic instead
        return jax.tree.map(lambda w, g: w - task.LR * g, params, grads), loss

    _, losses = jax.lax.scan(step, params, None, length=task.TRAIN_STEPS)
    return losses



def train_sessions():
    """FIFTH question: recording and session in one scan nest.

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
        # HAND-ROLLED MOMENTUM, same note: the velocity is the caller's
        # pytree because nothing in the framework owns the training loop
        vel = jax.tree.map(lambda v, g: task.MU * v + g, vel, grads)
        params = jax.tree.map(lambda w, v: w - task.LR * v, params, vel)
        return (params, vel, s2, h2), loss

    def over_recordings(carry, xt):
        params, vel, stats = carry
        (params, vel, stats, _), losses = jax.lax.scan(
            over_chunks, (params, vel, stats, hidden0), xt)   # RECORDING: hidden dies
        return (params, vel, stats), losses

    def over_sessions(params, xt):
        vel0 = jax.tree.map(jnp.zeros_like, params)   # SESSION: momentum restarts
        (params, _, _), losses = jax.lax.scan(
            over_recordings, (params, vel0, stats0), xt)       # SESSION: recalibrated
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
