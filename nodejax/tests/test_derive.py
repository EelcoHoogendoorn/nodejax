"""Methods and leaf definition derivation."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import scan, PNode, Node, Leaf, derive, batch, node
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def Gaussian() -> Node:
    """A node carrying behavior beyond apply: whiten as the mapping,
    log_prob/sample as methods."""
    def param(mean, log_std):
        return Struct(mean=jnp.asarray(mean), log_std=jnp.asarray(log_std))

    def apply(param, input):
        return (input - param.mean) / jnp.exp(param.log_std)

    def log_prob(param, x):
        z = (x - param.mean) / jnp.exp(param.log_std)
        return -0.5 * z ** 2 - param.log_std - 0.5 * jnp.log(2 * jnp.pi)

    def sample(param, rng):
        # rng is a channel, delivered as a KeyStream in every context
        return param.mean + jnp.exp(param.log_std) * jax.random.normal(rng.next())

    return Leaf(apply, param=param, name='gaussian',
                    methods=dict(log_prob=log_prob, sample=sample))


def test_methods_bind_param():
    g = Gaussian().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(g.log_prob(0.0), -0.5 * jnp.log(2 * jnp.pi))
    key = jax.random.PRNGKey(0)
    # rng is a channel: a bare node declares none, so the caller passes it
    # by keyword, explicitly
    assert jnp.allclose(g.sample(rng=key), g.sample(rng=key))

    # unbound access on the node: the raw function, channels and all
    raw = Gaussian().log_prob
    assert jnp.allclose(raw(g.param, 0.0), g.log_prob(0.0))


def test_grad_through_method():
    """The pytree is the object, methods included: grad of a method w.r.t.
    the node flows into its params."""
    g = Gaussian().parameterize(mean=jnp.asarray(1.0), log_std=jnp.asarray(0.0))
    grads = jax.grad(lambda n: n.log_prob(2.0))(g)
    assert type(grads) is PNode
    assert jnp.allclose(grads.param.mean, 1.0)  # d/dmean of -(x-mean)^2/2 at x=2


def test_missing_method_error_lists_methods():
    g = Gaussian().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))
    with pytest.raises(AttributeError, match='log_prob'):
        g.entropy()


def test_reserved_method_names_rejected():
    with pytest.raises(TypeError, match='apply'):
        Leaf(lambda input: input, name='x', methods={'apply': lambda p: p})


@pytest.mark.parametrize('name', [
    'contract', 'members', 'specialize', 'with_input',
    'input', 'input_spec', 'input_shape',
])
def test_framework_authoring_names_are_reserved(name):
    with pytest.raises(TypeError, match=name):
        Leaf(lambda input: input, name='x', methods={name: lambda: None})


@pytest.mark.parametrize('name', [
    'param_fn', 'init_fn', 'apply_fn', 'feed', 'feed_bundle',
    'resolve_input', 'rebuild_members', 'preserve_input',
    'get_apply_input_spec',
])
def test_old_internal_operation_names_are_valid_methods(name):
    node = Leaf(
        lambda input: input,
        name='x',
        methods={name: lambda value: value + 1},
    )

    assert getattr(node, name)(2) == 3


def test_derive_override_apply_can_call_a_closed_over_parent():
    """An apply replacement may explicitly call a captured parent Node."""
    integrator = Integrator()

    def apply(param, state, input):
        state, y = integrator.apply(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(integrator, apply=apply, name='clipped')
    node = Clipped.parameterize()

    final, outs = scan(node)(node.init(), jnp.ones(3))
    assert jnp.allclose(outs, jnp.array([1.0, 1.0, 1.0]))  # output clipped...
    assert jnp.allclose(final, 3.0)                        # ...state integrates on


def test_derive_can_add_state():
    """An init override adds cyclic state to a plain parent."""
    gain = Gain()

    def init(param):
        return jnp.asarray(0.0)

    def apply(param, state, input):
        y = gain.apply(param, input)
        smoothed = 0.5 * state + 0.5 * y
        return smoothed, smoothed

    Smoothed = derive(gain, apply=apply, init=init, name='smoothed')
    assert type(Smoothed) is Node and Smoothed.cyclic and Smoothed.parametric

    node = Smoothed.parameterize(scale=jnp.asarray(2.0))  # parent's param ctor inherited
    s = node.init()
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)   # 0.5*0 + 0.5*2
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)   # 0.5*1 + 0.5*2


def test_derive_merges_disjoint_parameter_fragments():
    def parent_param(scale=2.0):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    parent = Leaf(apply, param=parent_param)

    def child_param(offset=3.0):
        return Struct(offset=jnp.asarray(offset))

    child = derive(parent, param=child_param)

    assert child.contract.param_input_spec.__keys__ == ('scale', 'offset')
    bound = child.parameterize(scale=4.0, offset=5.0)
    assert bound.param.__keys__ == ('scale', 'offset')
    assert bound.param.scale == 4.0
    assert bound.param.offset == 5.0
    assert bound.apply(2.0) == 8.0


def test_derive_preserves_child_state_through_inherited_apply():
    def parent_init(start=1.0):
        return Struct(total=jnp.asarray(start))

    def apply(state, input):
        total = state.total + input
        return Struct(total=total), total

    parent = Leaf(apply, init=parent_init)

    def child_init(marker=7.0):
        return Struct(marker=jnp.asarray(marker))

    child = derive(parent, init=child_init)

    assert child.contract.state_input_spec.__keys__ == ('start', 'marker')
    state = child.init(start=2.0, marker=9.0)
    next_state, output = child.apply(state, 3.0)
    assert next_state.total == output == 5.0
    assert next_state.marker == 9.0


def test_derive_explicit_apply_owns_the_complete_state_transition():
    parent = Leaf(
        lambda state, input: (Struct(total=state.total + input), input),
        init=lambda: Struct(total=jnp.asarray(0.0)),
    )

    def init():
        return Struct(count=jnp.asarray(0))

    def apply(state, input):
        state = state.replace(
            total=state.total + input,
            count=state.count + 1,
        )
        return state, state

    child = derive(parent, init=init, apply=apply)
    state = child.init()
    state, output = child.apply(state, 2.0)
    assert output.total == 2.0
    assert output.count == 1


def test_derive_rejects_a_partial_explicit_state_transition():
    parent = Leaf(
        lambda state, input: (Struct(total=state.total + input), input),
        init=lambda: Struct(total=jnp.asarray(0.0)),
    )

    child = derive(
        parent,
        init=lambda: Struct(count=jnp.asarray(0)),
        apply=lambda state, input: (
            Struct(count=state.count + 1), input),
    )

    with pytest.raises(TypeError, match='return every state field'):
        child.apply(child.init(), 1.0)


def test_derive_requires_structs_only_when_constructor_results_merge():
    parent = Leaf(
        lambda param, input: param * input,
        param=lambda scale: jnp.asarray(scale),
    )
    child = derive(
        parent,
        param=lambda offset: Struct(offset=jnp.asarray(offset)),
    )

    with pytest.raises(TypeError, match='both constructors return Struct'):
        child.parameterize(scale=2.0, offset=1.0)

    plain = Leaf(lambda input: input)
    scalar = derive(
        plain,
        param=lambda scale: jnp.asarray(scale),
        apply=lambda param, input: param * input,
    )
    assert scalar.parameterize(scale=2.0).apply(3.0) == 6.0


def test_derive_rejects_constructor_result_field_collisions():
    parent = Leaf(
        lambda param, input: param.value * input,
        param=lambda left: Struct(value=jnp.asarray(left)),
    )
    child = derive(
        parent,
        param=lambda right: Struct(value=jnp.asarray(right)),
    )

    with pytest.raises(TypeError, match=r"param fields overlap: \['value'\]"):
        child.parameterize(left=1.0, right=2.0)

    state_parent = Leaf(
        lambda state, input: (state, input),
        init=lambda left: Struct(value=jnp.asarray(left)),
    )
    state_child = derive(
        state_parent,
        init=lambda right: Struct(value=jnp.asarray(right)),
    )

    with pytest.raises(TypeError, match=r"state fields overlap: \['value'\]"):
        state_child.init(left=1.0, right=2.0)


def test_derive_rejects_constructor_input_collisions():
    parent = Leaf(
        lambda param, input: param.left * input,
        param=lambda shared: Struct(left=jnp.asarray(shared)),
    )

    with pytest.raises(
            TypeError,
            match=r"param constructor inputs overlap: \['shared'\]"):
        derive(
            parent,
            param=lambda shared: Struct(right=jnp.asarray(shared)),
        )


def test_derive_rejects_invalid_inherited_state_fragments():
    def child_init():
        return Struct(child=jnp.asarray(0.0))

    non_struct = Leaf(
        lambda state, input: (state.parent + input, input),
        init=lambda: Struct(parent=jnp.asarray(0.0)),
    )
    non_struct = derive(non_struct, init=child_init)
    with pytest.raises(TypeError, match='must return a Struct state fragment'):
        non_struct.apply(non_struct.init(), 1.0)

    unknown = Leaf(
        lambda state, input: (Struct(other=input), input),
        init=lambda: Struct(parent=jnp.asarray(0.0)),
    )
    unknown = derive(unknown, init=child_init)
    with pytest.raises(
            TypeError, match=r"unknown state fields: \['other'\]"):
        unknown.apply(unknown.init(), 1.0)


def test_derive_routes_rng_to_both_constructor_fragments():
    def parent_param(rng):
        return Struct(parent=jax.random.normal(rng.next()))

    parent = Leaf(
        lambda param, input: param.parent + input,
        param=parent_param,
    )

    def child_param(rng):
        return Struct(child=jax.random.normal(rng.next()))

    child = derive(parent, param=child_param)
    key = jax.random.PRNGKey(0)
    first = child.parameterize(rng=key).param
    replay = child.parameterize(rng=key).param

    assert child.contract.param_takes_rng
    assert jnp.array_equal(first.parent, replay.parent)
    assert jnp.array_equal(first.child, replay.child)
    assert not jnp.array_equal(first.parent, first.child)


def test_derive_keeps_a_single_stochastic_fragment_reproducible():
    def parent_param(rng):
        return Struct(parent=jax.random.normal(rng.next()))

    parent = Leaf(
        lambda param, input: param.parent + input,
        param=parent_param,
    )
    child = derive(
        parent,
        param=lambda: Struct(child=jnp.asarray(1.0)),
    )
    key = jax.random.PRNGKey(0)

    first = child.parameterize(rng=key).param.parent
    replay = child.parameterize(rng=key).param.parent

    assert jnp.array_equal(first, replay)


def test_derive_combines_priming_and_rng_requirements():
    def parent_init(rng):
        return Struct(parent=jax.random.normal(rng.next()))

    parent = Leaf(
        lambda state, input: (state, input),
        init=parent_init,
    )

    def child_init(input, rng):
        return Struct(child=input + jax.random.normal(rng.next()))

    child = derive(parent, init=child_init)
    key = jax.random.PRNGKey(0)

    assert child.contract.init_requires_input
    assert child.contract.init_takes_rng
    with pytest.raises(TypeError, match='requires a real input value'):
        child.init(rng=key)

    first = child.init(input=3.0, rng=key)
    replay = child.init(input=3.0, rng=key)
    assert jnp.array_equal(first.parent, replay.parent)
    assert jnp.array_equal(first.child, replay.child)
    assert not jnp.array_equal(first.parent, first.child)


def test_derive_retains_constructor_owned_roles_when_apply_omits_them():
    parent = Integrator()

    child = derive(parent, apply=lambda input: input * 2.0)

    assert child.parametric
    assert child.cyclic
    node = child.parameterize()
    state = node.init()
    next_state, output = node.apply(state, 3.0)
    assert next_state == state
    assert output == 6.0


def test_derive_rejects_an_apply_role_without_an_effective_constructor():
    parent = Leaf(lambda input: input)

    with pytest.raises(TypeError, match='no param constructor exists'):
        derive(parent, apply=lambda param, input: input)

    with pytest.raises(TypeError, match='no initializer exists'):
        derive(parent, apply=lambda state, input: (state, input))


def test_derive_merges_methods():
    G = Gaussian()
    child = derive(G, name='gaussian2', methods=dict(
        log_prob=lambda param, x: jnp.asarray(42.0),          # override
        entropy=lambda param: param.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e),  # add
    ))
    node = child.parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(node.log_prob(0.0), 42.0)                       # child wins
    assert jnp.allclose(node.entropy(), 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(node.sample(rng=key),                            # parent's kept
                        G.parameterize(mean=0.0, log_std=0.0).sample(rng=key))


def test_derive_unions_tags():
    parent = Leaf(
        lambda input: input,
        tags={'parent', 'shared'},
    )
    child = derive(parent, tags={'child', 'shared'})

    assert child.tags == frozenset({'parent', 'shared', 'child'})


def test_nested_derivation_flattens_storage_and_preserves_added_state():
    def apply(state, input):
        total = state.total + input
        return Struct(total=total), total

    parent = Leaf(
        apply,
        param=lambda parent_value: Struct(
            parent_value=jnp.asarray(parent_value)),
        init=lambda parent_state: Struct(
            total=jnp.asarray(parent_state)),
        tags={'parent'},
    )
    middle = derive(
        parent,
        param=lambda middle_value: Struct(
            middle_value=jnp.asarray(middle_value)),
        init=lambda middle_state: Struct(
            middle_state=jnp.asarray(middle_state)),
        tags={'middle'},
    )
    child = derive(
        middle,
        param=lambda child_value: Struct(
            child_value=jnp.asarray(child_value)),
        init=lambda child_state: Struct(
            child_state=jnp.asarray(child_state)),
        tags={'child'},
    )

    bound = child.parameterize(
        parent_value=1.0, middle_value=2.0, child_value=3.0)
    state = bound.init(
        parent_state=4.0, middle_state=5.0, child_state=6.0)
    next_state, output = bound.apply(state, 2.0)

    assert bound.param.__keys__ == (
        'parent_value', 'middle_value', 'child_value')
    assert next_state.__keys__ == (
        'total', 'middle_state', 'child_state')
    assert next_state.total == output == 6.0
    assert next_state.middle_state == 5.0
    assert next_state.child_state == 6.0
    assert child.tags == frozenset({'parent', 'middle', 'child'})
    assert not child.members


def test_derive_rejects_bound_parent_values():
    param_bound = Gain().parameterize(scale=2.0)
    with pytest.raises(TypeError, match='unbound Node view'):
        derive(param_bound)

    stateful = Leaf(
        lambda state, input: (state, input),
        init=lambda: Struct(value=jnp.asarray(0.0)),
    ).initialize()
    with pytest.raises(TypeError, match='unbound Node view'):
        derive(stateful)


def test_derived_factory_records_its_own_construction_replay():
    @node
    def Scaled(multiplier: float) -> Node:
        def param(scale=1.0):
            return Struct(scale=jnp.asarray(scale))

        def apply(param, input):
            return param.scale * input * multiplier

        return Leaf(apply, param=param)

    @node
    def Shifted(multiplier: float) -> Node:
        parent = Scaled(multiplier)

        def param(offset=0.0):
            return Struct(offset=jnp.asarray(offset))

        return derive(parent, param=param)

    shifted = Shifted(2.0).specialize(multiplier=4.0)
    shifted = shifted.parameterize(scale=3.0, offset=5.0)

    assert shifted.param.offset == 5.0
    assert shifted.apply(2.0) == 24.0


def test_derived_apply_reads_current_methods_from_node():
    """An inherited apply resolves methods from the executing definition."""
    def param(scale=2.0):
        return Struct(scale=jnp.asarray(scale))

    def init(initial=1.0):
        return jnp.asarray(initial)

    def output(param, state):
        return param.scale * state

    def apply(node, param, state, input):
        state = state + input
        return state, node.output(param, state)

    parent = Leaf(
        apply, param=param, init=init, methods={'output': output})
    child = derive(
        parent,
        methods={'output': lambda param, state: output(param, state) + 10.0},
    )

    parent = parent.parameterize().initialize()
    child = child.parameterize().initialize()

    parent, parent_output = parent.apply(3.0)
    child, child_output = child.apply(3.0)

    assert parent.state == child.state == 4.0
    assert parent_output == 8.0
    assert child_output == 18.0


def test_derived_defs_stay_composable():
    """Derived nodes are ordinary nodes: they transform and compose."""
    integrator = Integrator()

    def apply(param, state, input):
        state, y = integrator.apply(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(integrator, apply=apply, name='clipped')

    b = batch(Clipped, n=2).parameterize()
    state = b.init()
    state, out = b.apply(state, jnp.array([0.4, 5.0]))
    assert jnp.allclose(out, jnp.array([0.4, 1.0]))

    pipe = (Gain() >> Clipped).parameterize(
        gain=Struct(scale=3.0))
    s = pipe.init()
    s, out = pipe.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)  # 3.0 integrated once, clipped to 1


def test_derived_apply_keeps_its_entropy_requirement():
    parent = Leaf(lambda input: input, name='identity')

    def apply(input, rng):
        return input + jax.random.normal(rng.next())

    child = derive(parent, apply=apply, name='noisy_identity')
    key = jax.random.PRNGKey(0)

    assert child.contract.apply_takes_rng
    assert child.contract.input_spec is None
    assert 'rng' not in child.contract.apply_fields
    assert jnp.allclose(child(input=0.0, rng=key), child(input=0.0, rng=key))


def test_derive_inherits_boundary_actions():
    def reset(carried, initialized, decided):
        return initialized

    parent = Leaf(lambda input: input, name='identity',
                  boundary={'episode': reset})
    child = derive(parent, name='derived_identity')

    assert child._def.boundaries == parent._def.boundaries
    assert child._def.boundaries['episode'] is reset


def test_method_channels_by_name():
    """Reserved names in a method signature are injected by name: the
    binding-stage prefix in binding order (self, node, param, state), the
    call's own arguments after them, positional or keyword. state on a bare
    node is the caller's to pass by keyword."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init():
        return jnp.asarray(0.0)

    def apply(param, state, input):
        return state + input, state + input

    def level(node, param, state, x):
        return (state + x) * param.scale, node.name

    acc = Leaf(apply, param=param, init=init, name='acc',
                   methods=dict(level=level)).parameterize(scale=2.0)

    # binding order is validated at definition: bound names come first...
    with pytest.raises(TypeError, match='come first'):
        Leaf(apply, param=param, init=init, name='acc',
                 methods=dict(bad=lambda x, param: x))
    # ...and keep their order among themselves
    with pytest.raises(TypeError, match='order'):
        Leaf(apply, param=param, init=init, name='acc',
                 methods=dict(bad=lambda state, param: state))

    # bare node: param and node inject; state is passed explicitly
    val, nm = acc.level(1.0, state=jnp.asarray(3.0))
    assert val == 8.0 and nm == 'acc'

    # wired member: all channels live, state chained through the step
    from nodejax import Composite

    def wapply(self, input):
        before = self.acc.level(0.0)[0]
        self.acc(input)
        after = self.acc.level(0.0)[0]
        return Struct(before=before, after=after)

    rig = Composite(acc=acc)(wapply, name='rig').parameterize()
    s = rig.with_input(jnp.asarray(1.0)).bind(rig.param).init()
    _, out = rig.apply(s, jnp.asarray(1.0))
    assert out.before == 0.0 and out.after == 2.0


def test_method_rng_is_a_stream_everywhere():
    """The rng slot arrives as a KeyStream in every context: the
    boundary stream inside a wiring, a wrapped explicit key on a bare
    node — one drawing idiom."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def draw(param, rng):
        return jax.random.normal(rng.next()) * param.scale

    g = Leaf(lambda param, input: input * param.scale, param=param,
                 name='g', methods=dict(draw=draw)).parameterize(scale=1.0)

    key = jax.random.PRNGKey(0)
    a, b = g.draw(rng=key), g.draw(rng=key)
    assert jnp.allclose(a, b)                      # explicit key: deterministic

    # in a wiring, the boundary stream feeds the method: the author
    # declares rng at the composite boundary, the method draws from it
    from nodejax import Composite

    def wapply(self, x, rng):
        return self.g.draw() + x

    rig = Composite(g=g)(wapply, name='rig').parameterize()
    o1 = rig.apply(x=jnp.asarray(0.0), rng=key)
    o2 = rig.apply(x=jnp.asarray(0.0), rng=key)
    o3 = rig.apply(x=jnp.asarray(0.0), rng=jax.random.PRNGKey(1))
    assert jnp.allclose(o1, o2) and not jnp.allclose(o1, o3)
