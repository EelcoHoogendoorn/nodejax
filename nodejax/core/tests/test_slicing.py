"""Member slicing: model.encoder is the member as a bound node.

The inverse of the transport container. Embedding absorbs a bound
node's cargo into a composition; slicing reassembles a member's cargo
into a bound node, runnable in isolation and re-embeddable, with the
weights it trained. Attribute precedence on PNode: real attributes,
then methods, then members as slices, then param fields as values.
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
    # the slice runs on the SAME weights the composite holds
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
    """A wrapper's param IS its member's, so the slice is the whole tree:
    batch(net).sample is the unbatched net on identical weights."""
    net = model()
    b = batch(net)
    sample = b.sample
    assert jnp.allclose(sample.param.linear.w, net.param.linear.w)
    assert sample.apply(jnp.ones(4)).shape == (3,)


def test_a_slice_reembeds_all_bound():
    """Slice out of one model, compose with bound members only: the
    closure rule hands back a BOUND node already wearing the donor
    weights, with no parameterize in sight."""
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
    """A tied alias has no param slot by design, so the slice is
    member_param's named error."""
    from nodejax import tie
    lm = tie(nn.Embed(7, 4) >> nn.Unembed(7, 4), 'embed', 'unembed')
    node = lm.parameterize(rng=jax.random.PRNGKey(0))
    with pytest.raises(TypeError, match='no slot'):
        node.unembed


def test_a_param_mapping_wrapper_refuses_to_slice():
    """ensemble and stack map params over an axis, so no member has a
    single binding to slice; they declare destructurable=False and the
    slice raises, named, instead of handing the inner a stacked tree it
    never contracted for. Row indexing is a parked design item."""
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
    """batch preserves params, so batch(ensemble) slices ONE hop: the
    outer slice hands back the ensemble node, and the next hop refuses."""
    from nodejax import batch, ensemble
    committee = batch(ensemble(nn.Linear(3), n=4).with_input(jnp.zeros(2))
                      ).parameterize(rng=jax.random.PRNGKey(0))
    ens = committee.sample
    assert type(ens) is PNode
    with pytest.raises(AttributeError, match='maps parameters'):
        ens.member


def test_the_trainer_slices_to_the_initialization():
    """The generic slice reads the standard tree contract: trainer.model
    is the model bound to param.model, the INITIALIZATION, because param
    is what training starts from. The current weights live in state, and
    that reading is the model() method's, the custom destructuring hook
    that already wins the attribute precedence."""
    import optax
    from nodejax import train_step
    trainer = train_step(
        nn.Linear(2).with_input(jnp.zeros(3)).parameterize(
            rng=jax.random.PRNGKey(0)).initialize(),
        lambda out, target: jnp.mean((out - target) ** 2),
        optax.sgd(0.1))
    start = trainer.pnode.model
    assert type(start) is PNode
    assert jnp.allclose(start.param.w, trainer.param.model.w)


def test_a_state_bound_tree_slices_members_with_their_state():
    """The slice door at the state rung: a member comes out as a PSNode
    on its param and state slices, a fork of that slot, advancing
    without the parent noticing."""
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
    """trainer.model at the state-bound stage is the LIVE model: the weights
    training has reached ride the state slice, where the param slice
    holds only what training started from."""
    import optax
    from nodejax import PSNode, train_step
    trainer = train_step(
        nn.Linear(2).with_input(jnp.zeros(3)).parameterize(
            rng=jax.random.PRNGKey(0)).initialize(),
        lambda out, t: jnp.mean((out - t) ** 2),
        optax.sgd(0.1))
    trainer, _ = trainer(input=jnp.ones(3), target=jnp.zeros(2))
    live = trainer.opt                       # the optimizer member, live
    assert type(live) is PSNode
    assert jnp.allclose(live.state.params.w, trainer.state.opt.params.w)


def test_batch_refuses_the_state_slice_and_allows_the_acyclic_one():
    """batch tiles state per element while leaving params flat, so the
    state-bound slice refuses where the param-only one (.pnode.sample)
    still works; an acyclic inner has no state to tile and slices
    freely."""
    from nodejax import batch
    b = batch(Integrator().parameterize(), n=3).initialize()
    with pytest.raises(AttributeError, match='maps state'):
        b.sample
    assert b.pnode.sample.param is b.param     # the param door stays open (flat)

    acyclic = batch(model()).initialize()     # the Linear >> gelu pipe
    sample = acyclic.sample
    assert sample.state == ()
