"""Verifies member slicing on `PNode` and `PSNode` compositions.

Attribute lookup on a bound node resolves real attributes first, then node
methods, then child members as bound slices, and finally parameter fields as
values. Slicing extracts a member with its bound parameters (and state on
`PSNode`) so it can run standalone or re-embed into a new composition.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, batch, nn, PNode
from nodejax.control import Integrator
from nodejax.struct import Struct


def model() -> Node:
    pipe = (nn.Linear(5) >> nn.gelu >> nn.Linear(3)).with_input(jnp.zeros(4))
    return pipe.parameterize(rng=jax.random.PRNGKey(0))


def test_a_member_slices_out_as_a_bound_node():
    net = model()
    encoder = net.linear
    assert type(encoder) is PNode
    assert encoder.param.w.shape == (4, 5)
    out = encoder.apply(jnp.ones(4))
    assert out.shape == (5,)
    # the slice runs on the same weights the composite holds
    assert jnp.allclose(out, jnp.ones(4) @ net.param.linear.w + net.param.linear.b)


def test_a_nonparametric_member_slices_too():
    act = model().gelu
    assert type(act) is PNode and act.param == ()
    assert jnp.allclose(act.apply(jnp.ones(2)), jax.nn.gelu(jnp.ones(2)))


def test_attribute_chains_thread_through_slices():
    net = model()
    # a member hop slices; the value is then spelled through .param
    assert jnp.allclose(net.linear.param.w, net.param.linear.w)


def test_a_wrapper_slices_to_its_declared_member():
    """Verifies `batch(net).sample` slices out the unbatched member on identical weights."""
    net = model()
    b = batch(net)
    sample = b.sample
    assert jnp.allclose(sample.param.linear.w, net.param.linear.w)
    assert sample.apply(jnp.ones(4)).shape == (3,)


def test_a_slice_reembeds_all_bound():
    """Verifies composing a sliced `PNode` with bound members yields a `PNode` directly."""
    donor = model()
    grafted = donor.linear >> nn.tanh
    assert grafted.bound
    assert jnp.allclose(grafted.param.linear.w, donor.param.linear.w)


def test_a_slice_reembeds_as_a_finished_capture_and_owes_no_key():
    """A bound member may be captured explicitly by a new composition.

    Its finished parameter value does not reopen the constructor, so the
    enclosing parameter plan is empty and parameterization draws nothing.
    Re-entering tree or static binding would discard this capture.
    """
    donor = model()
    grafted = (donor.linear >> nn.gelu.node).parameterize()
    assert jnp.allclose(grafted.param.linear.w, donor.param.linear.w)


def test_a_tied_alias_refuses_to_slice():
    """Verifies slicing a `tie` alias raises `TypeError` because it owns no parameter slot."""
    from nodejax import tie
    lm = tie(nn.Embed(7, 4) >> nn.Unembed(7, 4), 'embed', 'unembed')
    node = lm.parameterize(rng=jax.random.PRNGKey(0))
    with pytest.raises(TypeError, match='no slot'):
        node.unembed


def test_a_param_mapping_wrapper_refuses_to_slice():
    """Verifies `ensemble` and `stack` refuse member slicing because parameters map over an axis."""
    from nodejax import ensemble, stack
    committee = ensemble(nn.Linear(3), n=4).with_input(jnp.zeros(2)).parameterize(
        rng=jax.random.PRNGKey(0))
    with pytest.raises(AttributeError, match='maps parameters'):
        committee.member
    tower = stack(nn.Linear(2), n=3).with_input(jnp.zeros(2)).parameterize(
        rng=jax.random.PRNGKey(0))
    with pytest.raises(AttributeError, match='maps parameters'):
        tower.layer


def test_slicing_stops_at_the_mapping_layer():
    """Verifies `batch(ensemble)` slices one hop to the `ensemble` `PNode` and refuses further slicing."""
    from nodejax import batch, ensemble
    committee = batch(ensemble(nn.Linear(3), n=4).with_input(jnp.zeros(2))
                      ).parameterize(rng=jax.random.PRNGKey(0))
    ens = committee.sample
    assert type(ens) is PNode
    with pytest.raises(AttributeError, match='maps parameters'):
        ens.member


def test_the_trainer_slices_to_the_initialization():
    """Verifies `trainer.pnode.objective.model` slices out the initial model `PNode`."""
    import optax
    from nodejax import train_step
    trainer = train_step(
        nn.Linear(2).with_input(jnp.zeros(3)).parameterize(
            rng=jax.random.PRNGKey(0)).initialize(),
        lambda out, target: jnp.mean((out - target) ** 2),
        optax.sgd(0.1))
    start = trainer.pnode.objective.model
    assert type(start) is PNode
    assert jnp.allclose(start.param.w, trainer.param.objective.model.w)


def test_a_state_bound_tree_slices_members_with_their_state():
    """Verifies slicing a `PSNode` extracts a standalone `PSNode` carrying the member's state."""
    from nodejax import PSNode
    session = (Integrator() >> Integrator()).parameterize().initialize()
    session, _ = session(1.0)
    part = session.integrator_2
    assert type(part) is PSNode
    assert jnp.allclose(part.state, session.state.integrator_2)
    advanced, _ = part(10.0)
    assert jnp.allclose(advanced.state, part.state + 10.0)
    assert jnp.allclose(session.state.integrator_2, part.state)   # the parent stands


def test_the_trainer_slices_to_the_live_model():
    """Verifies `trainer.trained()` returns an acyclic `PNode` carrying the updated model weights."""
    import optax
    from nodejax import PNode, train_step
    trainer = train_step(
        nn.Linear(2).with_input(jnp.zeros(3)).parameterize(
            rng=jax.random.PRNGKey(0)).initialize(),
        lambda out, target: jnp.mean((out - target) ** 2),
        optax.sgd(0.1))
    trainer, _ = trainer(input=jnp.ones(3), target=jnp.zeros(2))
    live = trainer.trained()
    assert type(live) is PNode
    assert jnp.allclose(live.param.w, trainer.state.opt.params.model.w)


def test_batch_refuses_the_state_slice_and_allows_the_acyclic_one():
    """Verifies `batch` refuses cyclic `PSNode` slicing while permitting `.pnode` and acyclic slices."""
    from nodejax import batch
    b = batch(Integrator().parameterize(), n=3).initialize()
    with pytest.raises(AttributeError, match='maps state'):
        b.sample
    assert b.pnode.sample.param is b.param     # unmapped param view slices directly

    acyclic = batch(model()).initialize()     # the Linear >> gelu pipe
    sample = acyclic.sample
    assert sample.state == ()
