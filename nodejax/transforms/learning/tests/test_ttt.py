"""ttt: the wrapped node's params as gradient-adapted state, and the
marker that names that reading of a trainer in the graph.

Every pin here runs the inner loop BARE: no train-time training anywhere,
the init and rates drawn once at parameterize and never optimized, which
is classic online learning, prequential gradient descent from a random
start. The same cells become the inner loop of meta-learning when an
outer train_step wraps them (the flagship row is exactly that), and no
line in them changes: nothing in a cell knows which level it serves."""

import jax
import jax.numpy as jnp
import pytest

from nodejax.transforms.learning import learned_sgd
from nodejax import (Node, train_step, Leaf, scan, scanned, stack, KeyStream,
                     supervised_ttt, next_step_ttt, reconstruction_ttt)
from nodejax.transforms.learning import ttt
from nodejax.struct import Struct
from nodejax import tile


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


def Scale() -> Node:
    def param(scale=0.0) -> Struct:
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    return Leaf(apply, param=param, name='g')


def sample(x: jax.Array):
    """Reconstruction self-supervision, assembled as data: the target
    IS the input."""
    return Struct(input=jnp.asarray(x), target=jnp.asarray(x))


def test_ttt_adapts_toward_self_supervision():
    """Reconstruction drives the wrapped gain toward identity, one
    step per sample, weights carried as state."""
    trainer = train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.1))
    assert trainer.param.objective.model.scale == 0.0
    assert trainer.param.opt.model.scale == 0.1           # per-leaf learned rates

    _, (outs, aux) = scan(trainer, record=True).apply(bundle=tile(sample(1.0), 50))
    scales = aux.state.opt.params.model.scale            # the recorded trajectory
    assert outs[0] == 0.0                                # first prediction: untrained
    assert scales[0] > 0.0                               # ...and the update landed
    assert bool(jnp.all(jnp.diff(scales) >= 0))
    assert abs(float(scales[-1]) - 1.0) < 1e-2           # converged to reconstruction


def test_ttt_predicts_then_updates():
    """Prequential order: the output comes from the incoming weights
    — a prediction those weights never trained on — and the update
    lands after, ready for the next step."""
    trainer = train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.5))
    successor, (out, _) = trainer.apply(bundle=sample(1.0))
    assert jnp.allclose(out, 0.0)                        # predicted at scale 0
    # -lr * d/ds (s-1)^2 at s=0
    assert jnp.allclose(successor.state.opt.params.model.scale, 1.0)


def test_ttt_under_scan_resets_per_sequence():
    """The binding is the anchor: a run advances a SUCCESSOR and never
    the trainer itself, so each call starts from the meta-init.
    Adaptation is per sequence, carried within."""
    rollout = scan(train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.1)))
    sequence = tile(sample(1.0), 20)
    _, (ys1, _) = rollout.apply(bundle=sequence)
    _, (ys2, _) = rollout.apply(bundle=sequence)
    assert jnp.allclose(ys1, ys2)                        # fresh start both times
    assert ys1[-1] > ys1[0]                              # adapted within the sequence


def test_ttt_marker_is_the_trainer_named():
    """The marker changes nothing: same predictions, same aux; the name and
    the trainer-shape check are the whole of it."""
    trainer = train_step(Scale().parameterize().initialize(), mse, learned_sgd(0.1))
    sequence = tile(sample(1.0), 8)
    _, (preds_a, aux_a) = scan(trainer).apply(bundle=sequence)
    _, (preds_b, aux_b) = scan(supervised_ttt(trainer)).apply(bundle=sequence)
    assert jnp.allclose(preds_a, preds_b)
    assert jnp.allclose(aux_a.loss, aux_b.loss)
    assert ttt(trainer).name == f'ttt({trainer.name})'
    assert ttt(trainer).cyclic         # the fast weights ARE carried state


def test_ttt_marker_refuses_a_bare_model():
    with pytest.raises(TypeError, match='product of train_step'):
        ttt(Scale())


def test_ttt_interior_to_a_pipe():
    """No final-layer assumption: reconstruction assembles the flowing
    signal into the element pair, so a ttt layer sits mid-pipe, its fast
    weights riding the pipe's state like any member state.
    test_motor_control.py::test_ttt_in_the_loop runs the full-size
    version, inside a closed loop, an ensemble and an outer trainer."""
    layer = reconstruction_ttt(train_step(Scale().parameterize().initialize(),
                                          mse, learned_sgd(0.1)))
    model = Leaf(lambda input: 2.0 * input, name='pre') >> layer
    rollout = scan(model)

    _, (ys, aux) = rollout.apply(jnp.ones(20))
    losses = jax.tree.leaves(aux)[0]
    assert ys[0] == 0.0                      # first prediction: untrained
    assert losses[-1] < losses[0]            # the interior learner adapts
    assert abs(ys[-1] - 2.0) < abs(ys[0] - 2.0)   # toward reconstructing the signal


def Lin() -> Node:
    """A stackable scalar cell: rng-built weight, input-shaped output."""
    def param(rng: KeyStream) -> Struct:
        return Struct(w=0.5 + 0.1 * jax.random.normal(rng.next(), ()))

    def apply(param, input):
        return param.w * input

    return Leaf(apply, param=param, name='lin')


def RNNCell() -> Node:
    """A stackable recurrent cell: hidden state carrying beneath whatever
    adapts above it, scalar signal in and out."""
    def param(rng: KeyStream) -> Struct:
        return Struct(wx=0.5 * jax.random.normal(rng.next(), (4,)),
                      wh=0.3 * jax.random.normal(rng.next(), (4, 4)) / 2.0,
                      wo=0.5 * jax.random.normal(rng.next(), (4,)))

    def init(param):
        return jnp.zeros(param.wx.shape)

    def apply(param, state, input):
        h = jnp.tanh(param.wx * input + param.wh @ state)
        return h, param.wo @ h

    return Leaf(apply, init=init, param=param, name='rnn')


def test_both_deep_compositions_just_work():
    """The deep stress test with a REAL RNN, both readings one line each:
    every layer carries two memories at two speeds, fast weights adapting
    per step and hidden state carrying beneath them, and the stack routes
    both.

    Both readings are the SAME flavor on the same signal, differing by one
    placement of stack. ONE HEAD LOSS THROUGH THE STACK: the whole deep
    net is one next-step learner, every layer's weights per-step state
    updated by backprop through the layers above. INDEPENDENT WEIGHT
    LAYERS: each layer its own next-step learner on its own signal, the
    losses arriving per (step, layer). And the fused phrase
    ttt(stack(...)) is refused, which is the trainer check forcing
    exactly this choice."""
    # Leave enough online steps for independently initialized deeper cells to
    # settle; their early target moves while the preceding cell is learning.
    data = (jnp.arange(200) % 2).astype(jnp.float32)

    # one head loss, backpropped through the whole stack
    deep = stack(RNNCell(), n=2).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    node = next_step_ttt(train_step(deep, mse, learned_sgd(0.1)))
    node = node.initialize(input=data[0])
    state, out = scan(node.pnode).apply(node.state, data)
    _, aux = out
    losses = aux.ttt_train_step_stack_rnn.loss
    assert losses[-1] < 0.3 * losses[1]                       # the head learns the shift
    fast = state.ttt_train_step_stack_rnn
    drift = jnp.max(jnp.abs(
        fast.opt.params.model.wx
        - node.param.ttt_train_step_stack_rnn.objective.model.wx), axis=1)
    assert drift.shape == (2,) and bool(jnp.all(drift > 0))   # every layer's weights moved
    assert fast.objective.model.shape == (2, 4)               # ...the hidden carrying beneath
    assert bool(jnp.any(fast.objective.model != 0))

    # independent weight layers, each predicting its own signal's next
    # value: construct the cell unbound so stack parameterizes every layer
    # independently. Composing a bound trainer would deliberately capture it.
    cell = next_step_ttt(train_step(RNNCell(), mse, learned_sgd(0.1)))
    rollout = scanned(stack(cell, n=2)).parameterize(
        rng=jax.random.PRNGKey(0))
    ys, aux = rollout.apply(data)
    losses = jax.tree.leaves(aux)[0]
    assert ys.shape == (200,) and losses.shape == (200, 2)    # one loss per (step, layer)
    # EVERY layer converges, the deeper one included: its signal stabilizes
    # as the layer beneath learns, and its task turns stationary
    assert bool(jnp.all(jnp.mean(losses[-15:], axis=0) < 0.01))


def test_next_step_ttt_learns_the_shift():
    """The temporal-shift flavor is nondegenerate: on an alternating signal
    the learner must map each value to its NEGATION, so the fast weight
    leaves identity and heads for -1, a task same-step reconstruction can
    never pose. The register primes from the first value, so the first
    pair is reconstruction-shaped and the shift engages at the second.

    A STATELESS learner under ttt is nonetheless a sequence model, which
    is the TTT-layers thesis: the weight is the recurrent state, each
    prediction depending on the whole history through it. Here the
    memoryless scale REMEMBERS the alternation rule as w -> -1."""
    cell = next_step_ttt(train_step(Scale().parameterize().initialize(),
                                    mse, learned_sgd(0.2)))
    signal = jnp.tile(jnp.array([-1.0, 1.0]), 10)
    bound = cell.initialize(input=signal[0])
    final, _ = scan(bound).apply(signal)
    assert float(final.state.ttt_train_step_g.opt.params.model.scale) < -0.9


def test_next_step_deep_compositions_just_work():
    """next_step's register is cyclic, so the deep forms exercise the
    walked init too: under stack, every layer's register primes from the
    signal as it arrives at that depth. On an alternating signal, ONE DEEP
    NET under the head loss splits the negation across its layers however
    backprop lands it, the PRODUCT of weights reaching -1; a STACK OF
    LOCAL LEARNERS drives every weight to -1 individually, each layer
    negating its own signal."""
    signal = jnp.tile(jnp.array([-1.0, 1.0]), 30)

    deep = next_step_ttt(train_step(
        stack(Lin(), n=3).parameterize(rng=jax.random.PRNGKey(0)).initialize(),
        mse, learned_sgd(0.2)))
    bound = deep.initialize(input=signal[0])
    final, _ = scan(bound).apply(signal)
    assert abs(float(jnp.prod(
        final.state.ttt_train_step_stack_lin.opt.params.model.w)) + 1.0) < 1e-2

    cell = next_step_ttt(train_step(Lin(), mse, learned_sgd(0.2)))
    layered = stack(cell, n=3)
    bound = layered.parameterize(rng=jax.random.PRNGKey(0)).initialize(input=signal[0])
    final, _ = scan(bound).apply(signal)
    assert bool(jnp.all(
        final.state.ttt_train_step_lin.opt.params.model.w < -0.9))
