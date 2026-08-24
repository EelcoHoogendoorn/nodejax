"""train_step: internalize optimization. Param is what training starts
from, state is where it has got to."""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import (batch, scan, scanned, state_reinit, Leaf, PSNode, Wrapper,
                     train_step, trained, serial, nn, map_members, tree_detach)
from nodejax.struct import Struct
from nodejax.control import Gain
from nodejax.transforms.train_step import optimizer, opt_reinit


def test_train_step_convergence():
    """The model arrives FULLY BOUND and those weights are the
    initialization; the trainer is itself a PSNode, immediately
    scannable, and the run advances its state."""
    gain = Gain().parameterize(scale=jnp.array(1.0)).initialize()
    trainer = train_step(gain, lambda pred, target: (pred - target) ** 2, optax.sgd(0.01))
    assert type(trainer) is PSNode and trainer.cyclic

    final, (_, aux) = trainer.scan(input=jnp.full(500, 2.0),
                                   target=jnp.full(500, 6.0))

    assert jnp.allclose(final.state.opt.params.scale, 3.0, atol=0.01)
    assert aux.loss[-1] < 1e-3


def test_train_step_rebuilds_through_its_members():
    """A trainer is a COMPOSITE of two members, `opt` and `inner`, so a rewrite
    of the model reaches it through the member it is under. It used to be a
    Wrapper, whose state was supposed to BE its inner's and was not, which is
    what left every structural walk reading the wrong tree."""
    l1 = nn.Linear(4)
    l2 = Wrapper(inner=nn.Linear(4))(name='target')
    model = serial(l1=l1, l2=l2).with_input(jnp.zeros(4)).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, lambda pred, target: jnp.sum((pred - target) ** 2), optax.sgd(0.01))

    # Replace l2 member inside the trainer; a rewrite reshapes state
    # layouts, so it works the param-only view
    new_l2 = Wrapper(inner=nn.Linear(4))(name='replacement')
    rebuilt = map_members(
        trainer.node, lambda m: new_l2 if m.name == 'target' else m)
    assert rebuilt.members.model.members.l2.name == 'replacement'
    assert l2.name == 'target'

    # tree_detach inside trainer by key
    detached = tree_detach(trainer.pnode, 'l2')
    assert detached.members.model.members.l2.name == 'detach(target)'


def test_train_step_does_not_adopt_apply_entropy_for_shape_only_init():
    model = batch(
        nn.Linear(2) >> nn.Dropout(0.2) >> nn.BatchNorm(0.9)
        >> nn.Linear(1)
    ).with_input(jnp.zeros((3, 4)))
    trainer = train_step(
        model,
        lambda pred, target: jnp.mean((pred - target) ** 2),
        optax.sgd(0.01),
    ).with_input(Struct(
        input=jnp.zeros((3, 4)), target=jnp.zeros((3, 1))))

    assert not bool(model.contract.init_takes_rng)
    assert not bool(trainer.contract.init_takes_rng)
    bound = trainer.parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    assert type(bound) is PSNode


def test_trained_rebuilds_its_current_run():
    def signed_gain(sign, name):
        def param(scale):
            return Struct(scale=jnp.asarray(scale))

        def apply(param, input):
            return sign * param.scale * input

        return Leaf(apply, param=param, name=name)

    sequence = Struct(
        input=jnp.ones(3), target=jnp.full(3, 5.0))
    run = trained(train_step(
        signed_gain(1.0, 'target').parameterize(
            scale=jnp.asarray(1.0)),
        lambda pred, target: (pred - target) ** 2,
        optax.sgd(0.1)))

    done, aux = run.apply(bundle=sequence)
    assert jnp.allclose(done.param.scale, 2.952, atol=1e-5)
    assert aux.loss.shape == (3,)

    replacement = signed_gain(-1.0, 'replacement')
    rebuilt_node = map_members(
        run.node, lambda member: replacement
        if member.name == 'target' else member)
    rebuilt = rebuilt_node.bind(run.param)
    rebuilt_done, _ = rebuilt.apply(bundle=sequence)
    _, original_output = done.apply(jnp.asarray(1.0))
    _, rebuilt_output = rebuilt_done.apply(jnp.asarray(1.0))

    assert rebuilt_done.name == 'replacement'
    assert not jnp.allclose(rebuilt_output, original_output)


def test_train_step_input_is_the_named_model_wire():
    """The trainer's `input` field contains the model's computed wire.

    A model may name that wire something else in its own call. The trainer
    feeds the value into that call form once, without nesting a formed model
    bundle under the model field again.
    """
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, signal):
        return param.scale * signal

    model = Leaf(apply, param=param, name='named_input').parameterize(
        scale=jnp.asarray(1.0))
    run = trained(train_step(
        model,
        lambda output, target: jnp.mean((output - target) ** 2),
        optax.sgd(0.1),
    ))

    done, aux = run.apply(
        input=jnp.ones(3), target=jnp.full(3, 2.0))

    assert jnp.allclose(done.param.scale, 1.488)
    assert aux.loss.shape == (3,)


def test_train_step_uses_the_empty_call_of_a_zero_input_model():
    """The uniform trainer input does not become a model wire."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init():
        return jnp.zeros(())

    def apply(param, state):
        state = state + 1
        return state, param.scale * state

    model = Leaf(apply, param=param, init=init, name='source')
    trainer = train_step(
        model,
        lambda output, target: (output - target) ** 2,
        optax.sgd(0.1),
    ).parameterize(scale=jnp.asarray(1.0)).initialize()

    successor, (output, aux) = trainer.apply(
        input=Struct(), target=jnp.asarray(2.0))

    assert output == 1.0
    assert aux.loss == 1.0
    assert successor.state.model == 1.0
    assert jnp.allclose(successor.state.opt.params.scale, 1.2)
    with pytest.raises(TypeError, match='formed call bundle'):
        trainer.apply(input=jnp.asarray(0.0), target=jnp.asarray(2.0))


def test_trained_routes_model_rng_only_from_the_outer_channel():
    def param(scale=1.0):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input, rng):
        # `input.rng` is ordinary domain data. Only the separate outer key
        # satisfies the authored RNG channel.
        return (param.scale * input.value + input.rng
                + jax.random.normal(rng.next(), ()))

    model = Leaf(apply, param=param, name='drawing_model').parameterize()
    run = trained(train_step(
        model, lambda pred, target: (pred - target) ** 2,
        optax.sgd(0.01)))
    nested = Struct(
        value=jnp.ones(3),
        rng=jnp.zeros(3),
    )

    key = jax.random.PRNGKey(0)
    with pytest.raises(TypeError, match='requires rng'):
        run.apply(input=nested, target=jnp.zeros(3))

    done, aux = run.apply(
        input=nested, target=jnp.zeros(3), rng=key)

    assert done.param.scale.shape == ()
    assert aux.loss.shape == (3,)


def test_a_boundary_claim_crosses_the_trainer():
    """Stateful training: the hidden carries from batch to batch and dies at
    the epoch, THROUGH the trainer. The tag is declared inside the model, the
    scan over an epoch's batches claims it, and train_step sits between the
    two saying nothing: the weights and the optimizer's state cross every
    boundary because a carry carries.

    Truncated BPTT falls out rather than being said: the hidden enters each
    step as carried state, so a step's gradient reaches back exactly to the
    start of its own batch.

    Pinned against a hand-rolled loop from both sides: the claimed run
    matches the loop that resets per epoch, the unclaimed run matches the
    loop that never resets, and the two disagree."""
    EPOCHS, BATCHES, LR = 2, 3, 0.1
    xs = jax.random.normal(jax.random.PRNGKey(0), (EPOCHS, BATCHES, 4))
    ts = jax.random.normal(jax.random.PRNGKey(1), (EPOCHS, BATCHES, 4))

    def Cell():
        # the hidden is weight-independent on purpose, so the reference's
        # state advance is unambiguous
        def param():
            return jnp.ones(())

        def init():
            return jnp.zeros(())

        def apply(param, state, input):
            h = state + input
            return h, h * param

        return Leaf(apply, param=param, init=init, name='cell')

    trainer = train_step(
        scan(state_reinit(Cell(), 'epoch')).parameterize().initialize(),
        lambda out, t: jnp.mean((out - t) ** 2),
        optax.sgd(LR))

    def reference(reset_per_epoch):
        w, h, losses = jnp.ones(()), jnp.zeros(()), []
        for e in range(EPOCHS):
            if reset_per_epoch:
                h = jnp.zeros(())
            for b in range(BATCHES):
                def loss_fn(w_):
                    outs = (h + jnp.cumsum(xs[e, b])) * w_
                    return jnp.mean((outs - ts[e, b]) ** 2)

                loss, g = jax.value_and_grad(loss_fn)(w)
                w, h = w - LR * g, h + jnp.sum(xs[e, b])
                losses.append(loss)
        return jnp.stack(losses).reshape(EPOCHS, BATCHES)

    _, (_, aux) = scan(scan(trainer, boundary='epoch')).apply(input=xs, target=ts)
    assert jnp.allclose(aux.loss, reference(True), atol=1e-5)

    _, (_, plain) = scan(scan(trainer)).apply(input=xs, target=ts)
    assert jnp.allclose(plain.loss, reference(False), atol=1e-5)
    assert not jnp.allclose(aux.loss, plain.loss)


def test_two_tags_cross_the_trainer():
    """Two tags in one tree, both claimed above the trainer.

    The drift dies at the outer boundary, the accumulator at the inner, and
    the weights
    cross everything because a carry carries: they are what couples the
    inner groups, which is why two reinit tags that would fall apart
    untrained compose under training. The name picks the scan even through a trainer:
    swap the two claims and the same tree computes something else."""
    EP, REC, K, LR = 2, 2, 2, 0.1
    xs = jax.random.normal(jax.random.PRNGKey(0), (EP, REC, K, 3))
    ts = jax.random.normal(jax.random.PRNGKey(1), (EP, REC, K, 3))

    def Drift():
        def init():
            return jnp.zeros(())

        def apply(state, input):
            a = state + input
            return a, a

        return Leaf(apply, init=init, name='drift').node

    def Cell():
        def param():
            return jnp.ones(())

        def init():
            return jnp.zeros(())

        def apply(param, state, input):
            h = state + input
            return h, h * param

        return Leaf(apply, param=param, init=init, name='cell').node

    model = scan(state_reinit(Drift(), 'ep') >> state_reinit(Cell(), 'rec'))
    trainer = train_step(model.parameterize().initialize(),
                         lambda out, t: jnp.mean((out - t) ** 2),
                         optax.sgd(LR))

    def reference():
        w, losses = jnp.ones(()), []
        for e in range(EP):
            a = jnp.zeros(())                      # EPOCH: the drift dies
            for r in range(REC):
                b = jnp.zeros(())                  # RECORDING: the accumulator dies
                for k in range(K):
                    aa = a + jnp.cumsum(xs[e, r, k])
                    bb = b + jnp.cumsum(aa)

                    def loss_fn(w_):
                        return jnp.mean((bb * w_ - ts[e, r, k]) ** 2)

                    loss, g = jax.value_and_grad(loss_fn)(w)
                    w, a, b = w - LR * g, aa[-1], bb[-1]
                    losses.append(loss)
        return jnp.stack(losses).reshape(EP, REC, K)

    run = scan(scan(scan(trainer, boundary='rec'), boundary='ep'))
    _, (_, aux) = run.apply(input=xs, target=ts)
    assert jnp.allclose(aux.loss, reference(), atol=1e-5)

    swapped = scan(scan(trainer, boundary='ep'), boundary='rec')
    _, (_, other) = scan(swapped).apply(input=xs, target=ts)
    assert not jnp.allclose(aux.loss, other.loss)


def test_a_lifetime_on_the_optimizer_state():
    """Warm restarts: at the cycle boundary the MOMENTS rebuild and the
    weights carry, which is SGDR's move. The optimizer owns its weights, so a
    whole-subtree state_reinit would restart training outright; splitting the
    slot takes an action over the leaf's own layout.

    Spelled as a WRAPPER over the stock optimizer, which is state_reinit's
    construction with a different action, and legitimate for the same reason:
    a transparent wrapper's state IS its inner's, so the layout the action
    reads is its own. That reasoning stops at leaves whose layout is the
    family's contract; over a composite the slots are member names and an
    outer action would override what members decided, which is why no generic
    at_boundary(node, tag, action) wrapper ships."""
    CYCLES, BATCHES, LR = 3, 4, 0.3
    xs = jax.random.normal(jax.random.PRNGKey(0), (CYCLES, BATCHES, 8))
    ts = 3.0 * xs
    mse = lambda p, t: jnp.mean((p - t) ** 2)

    def Gain():
        def param():
            return jnp.zeros(())

        def apply(param, state, input):
            return state, input * param

        return Leaf(apply, param=param, init=lambda: jnp.zeros(()), name='gain')

    trainer = train_step(Gain().parameterize().initialize(), mse,
                         opt_reinit(optimizer(optax.adam(LR)), 'cycle'))
    _, (_, aux) = scan(scan(trainer, boundary='cycle')).apply(input=xs, target=ts)

    tx = optax.adam(LR)
    w, ref = jnp.zeros(()), []
    for c in range(CYCLES):
        opt = tx.init(w)                                     # the restart
        for b in range(BATCHES):
            loss, g = jax.value_and_grad(lambda w_: mse(xs[c, b] * w_, ts[c, b]))(w)
            up, opt = tx.update(g, opt, w)
            w = optax.apply_updates(w, up)
            ref.append(loss)
    assert jnp.allclose(aux.loss, jnp.stack(ref).reshape(CYCLES, BATCHES), atol=1e-5)

    _, (_, plain) = scan(scan(trainer)).apply(input=xs, target=ts)
    assert not jnp.allclose(aux.loss, plain.loss)
