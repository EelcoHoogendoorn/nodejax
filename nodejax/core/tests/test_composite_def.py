"""Verifies `Composite` and `Wrapper` definition-level composition, member routing, and wiring discovery."""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import (Node, trained, Leaf, derive, node, scan, scanned,
                     train_step, Composite, Wrapper, serial)
from nodejax.struct import Struct
from nodejax.core.binding import (Aux)

DT = 0.02


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


# Control blocks used by the composite cascade tests.

def PI(dt: float) -> Node:
    """Builds a proportional-integral controller `Leaf`."""
    def param(kp, ki):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))

    def init():
        return jnp.zeros(())

    def apply(param, state, input):
        integral = state + input * dt
        return integral, param.kp * input + param.ki * integral

    return Leaf(apply, init=init, param=param, name='pi')


def Observer(dt: float, alpha: float=0.4) -> Node:
    """Builds a parameterless angle-and-velocity observer `Leaf`."""
    def init():
        return Struct(theta=jnp.zeros(()), omega=jnp.zeros(()))
    def apply(state, input):
        omega_raw = (input - state.theta) / dt
        omega = (1 - alpha) * state.omega + alpha * omega_raw
        estimate = Struct(theta=input, omega=omega)
        return estimate, estimate
    return Leaf(apply, init=init, name='observer')


def Motor() -> Node:
    """Builds a parameter-only back-EMF gain `Leaf`."""
    def param(ke):
        return Struct(ke=jnp.asarray(ke))

    def apply(param, input):
        return param.ke * input

    return Leaf(apply, param=param, name='motor')


def Delay() -> Node:
    """Builds a scalar one-step delay `Leaf`."""
    def init():
        return jnp.zeros(())

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='delay')


def Servo(dt: float, kt: float=1.0, ke: float=0.1, damping: float=0.2, inertia: float=0.5) -> Node:
    """Builds a physical motor-and-load plant `Leaf`."""
    def init():
        z = jnp.zeros(())
        return Struct(theta=z, omega=z)
    def apply(state, input):
        torque = kt * (input - ke * state.omega)
        omega = state.omega + dt * (torque - damping * state.omega) / inertia
        theta = state.theta + dt * omega
        return Struct(theta=theta, omega=omega), theta
    return Leaf(apply, init=init, name='servo')


# Composite definitions wired over member nodes.

def Cascade(motor: Node, observer: Node, position_ctrl: Node, velocity_ctrl: Node, feedforward: Node=1.0) -> Node:
    """Wires an observer, cascaded position/velocity controllers, and motor feedforward."""
    members = Composite(motor=motor, observer=observer,
                   position_ctrl=position_ctrl, velocity_ctrl=velocity_ctrl)

    def apply(self, ref, theta):
        est = self.observer(theta)
        w_ref = self.position_ctrl(ref - est.theta)
        fb_v = self.velocity_ctrl(w_ref - est.omega)
        return fb_v + feedforward * self.motor(est.omega)

    return members(apply, name='cascade')


def Loop(dt: float, ctrl: Node) -> Node:
    """Wires a closed-loop servo plant with a one-step measurement delay."""
    members = Composite(ctrl=ctrl, plant=Servo(dt), delay=Delay())

    def apply(self, input):
        v = self.ctrl(ref=input, theta=self.delay.state)
        theta = self.plant(v)
        self.delay(theta)
        return theta

    return members(apply, name='servo_loop')


def stock_loop(position_ctrl: Node, velocity_ctrl: Node) -> Node:
    """Builds the default closed-loop servo cascade."""
    return Loop(DT, Cascade(Motor()(ke=0.1), Observer(DT),
                                    position_ctrl, velocity_ctrl))


def test_composite_infers_member_param_shapes_from_input():
    # Shape-reading members infer their fan-in from authored `Composite` wiring.
    from nodejax import nn

    def gated(width):
        members = Composite(a=nn.Linear(width), b=nn.Linear(width), gate=nn.Linear(1))

        def apply(self, input):
            g = jax.nn.sigmoid(self.gate(input))
            return g * self.a(input) + (1 - g) * self.b(input)

        return members(apply, name='gated')

    m = gated(4).with_input(jnp.zeros(5)).parameterize(rng=jax.random.PRNGKey(0))
    assert m.param.a.w.shape == (5, 4)
    assert m.param.gate.w.shape == (5, 1)
    assert m.apply(jnp.zeros(5)).shape == (4,)
    # Rebinding with a different input width produces matching parameter shapes.
    m2 = gated(4).with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))
    assert m2.param.a.w.shape == (3, 4)


def test_members_live_on_the_def():
    """Verifies `members` stores child definitions while `param` holds plain `Struct` trees."""
    loop = stock_loop(PI(DT)(kp=1.0, ki=0.0), PI(DT)(kp=1.0, ki=0.5))
    assert set(loop.members.__keys__) == {'ctrl', 'plant', 'delay'}
    assert set(loop.members.ctrl.members.__keys__) == {'motor', 'observer',
                                                 'position_ctrl', 'velocity_ctrl'}

    bound = loop.parameterize()
    p = bound.param
    assert type(p.ctrl.position_ctrl) is Struct
    from nodejax import PNode
    assert not any(type(x) is PNode
                   for x in jax.tree.leaves(
                       p, is_leaf=lambda x: type(x) is PNode))
    assert p.ctrl.motor.ke == 0.1
    assert 'observer' not in p.ctrl

    paths = {jax.tree_util.keystr(path)
             for path, _ in jax.tree_util.tree_flatten_with_path(p)[0]}
    assert '.ctrl.position_ctrl.kp' in paths
    assert '.ctrl.motor.ke' in paths

    # Keyword overrides replace selected member slots while preserving captured defaults.
    other = loop.parameterize(ctrl=Struct(position_ctrl=Struct(kp=9.0, ki=2.0)))
    assert jax.tree.structure(bound) == jax.tree.structure(other)
    assert other.param.ctrl.position_ctrl.kp == 9.0
    assert other.param.ctrl.velocity_ctrl.ki == 0.5

    with pytest.raises(TypeError):
        loop.parameterize(ctrl=Struct(motr=Struct(ke=0.1),
                                    position_ctrl=Struct(kp=1.0, ki=0.0),
                                    velocity_ctrl=Struct(kp=1.0, ki=0.5)))

    state = bound.with_input(jnp.zeros(())).bind(bound.param).init()
    assert state.ctrl.velocity_ctrl.shape == ()
    assert state.plant.theta.shape == ()
    assert state.delay.shape == ()

    discovered = bound.with_input(jnp.zeros(())).bind(bound.param).init()
    assert discovered.ctrl.observer.theta.shape == ()


def test_cascade_tracks_a_step():
    """Verifies the cascaded controller tracks a unit step with bounded overshoot."""
    rollout = scanned(stock_loop(PI(DT)(kp=4.0, ki=0.0),
                              PI(DT)(kp=2.0, ki=1.0)))
    bound = rollout.parameterize()

    thetas = bound.apply(jnp.ones(300))
    assert jnp.all(jnp.isfinite(thetas))
    assert jnp.abs(thetas[-1] - 1.0) < 0.02
    assert jnp.max(jnp.abs(thetas)) < 1.6


def test_blocks_are_polymorphic():
    """Verifies a `derive`-adapted saturating controller substitutes into `stock_loop`."""
    base = PI(DT)

    def apply(param, state, input):
        state, u = base.apply(param, state, input)
        return state, jnp.clip(u, -2.0, 2.0)

    SatPI = derive(base, apply=apply, name='sat_pi')

    rollout = scanned(stock_loop(PI(DT)(kp=4.0, ki=0.0),
                              SatPI(kp=2.0, ki=1.0)))
    bound = rollout.parameterize()

    thetas = bound.apply(jnp.ones(300))
    assert jnp.all(jnp.isfinite(thetas))
    assert jnp.abs(thetas[-1] - 1.0) < 0.05


def test_autotune_reaches_into_subblocks():
    """Verifies `train_step` updates controller gains nested inside the `Composite` parameter tree."""
    rollout = scanned(stock_loop(PI(DT)(kp=0.5, ki=0.0),
                              PI(DT)(kp=0.5, ki=0.1)))
    trainer = train_step(rollout.parameterize().initialize(), mse, optax.adam(0.02))

    refs = jnp.ones(150)
    steps = 250

    def tile(x):
        return jnp.broadcast_to(x, (steps,) + x.shape)

    final, aux = trained(trainer).apply(input=tile(refs), target=tile(refs))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.5 * aux.loss[0]
    assert final.param.ctrl.position_ctrl.kp > 0.5
    assert 'observer' not in final.param.ctrl


def test_input_discovery_from_the_wiring():
    """Verifies `Composite` initialization infers per-member state shapes from call-site wiring."""
    def Accumulator():
        def init(node):
            return jnp.zeros_like(node.input)
        def apply(state, input):
            new = state + input
            return new, new
        return Leaf(apply, init=init, name='acc')

    def Projection():
        def param(w):
            return Struct(w=jnp.asarray(w))
        def apply(param, input):
            return input @ param.w
        return Leaf(apply, param=param, name='proj')

    def Chain():
        members = Composite(a=Accumulator(), proj=Projection(), b=Accumulator())

        def apply(self, input):
            hidden = self.a(input)
            return self.b(self.proj(hidden))

        return members(apply, name='comp')

    node = Chain().parameterize(proj=Struct(w=jnp.ones((3, 2))))
    state = node.with_input(jnp.zeros(3)).bind(node.param).init()
    assert state.a.shape == (3,)
    assert state.b.shape == (2,)



def test_full_custom_overrides():
    """Verifies explicit `param` and `init` overrides replace generated `Composite` constructors."""
    # Raw contract `apply` passes member inputs directly without `self` routing.
    members = Composite(coarse=PI(DT).with_input(0.0),
                        fine=PI(DT).with_input(0.0))

    def param(kp, ki):
        gains = Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))
        return Struct(coarse=gains, fine=gains)

    def init():
        return Struct(coarse=jnp.zeros(()), fine=jnp.zeros(()))

    def apply(param, state, input):
        s1, mid = members.coarse.node.apply(param.coarse, state.coarse, input.input)
        s2, out = members.fine.node.apply(param.fine, state.fine, mid)
        return Struct(coarse=s1, fine=s2), out

    tied = members(apply, param=param, init=init, name='tied')
    bound = tied.parameterize(kp=1.0, ki=0.0)
    assert bound.param.coarse.kp == bound.param.fine.kp

    initial = bound.with_input(
        bundle=Struct(input=jnp.asarray(1.0))).bind(bound.param).init()
    _, out = bound.apply(initial, input=jnp.asarray(1.0))
    assert out == 1.0

    with pytest.raises(TypeError):
        tied.parameterize(kp=1.0, kii=0.0)


def test_wrapper_is_a():
    """Verifies `Wrapper` preserves its inner member's flat parameter and state layout."""
    def apply(self, input):
        return jnp.clip(self.pi(input), -1.0, 1.0)

    sat = Wrapper(pi=PI(DT))(apply, name='sat_pi')
    assert sat.members.__keys__ == ('pi',)
    bound = sat.parameterize(kp=4.0, ki=1.0)
    assert '.kp' in {jax.tree_util.keystr(p)
                     for p, _ in jax.tree_util.tree_flatten_with_path(bound.param)[0]}

    state = bound.with_input(jnp.asarray(10.0)).bind(bound.param).init()
    s2, out = bound.apply(state, jnp.asarray(10.0))
    assert out == 1.0
    assert s2 != state


def test_wrapper_captured_param_is_replaced_by_explicit_parameterization() -> None:
    """A bound member supplies the wrapper's current param, while entering
    parameterization again uses the member's flat constructor form."""
    captured = Motor().parameterize(ke=2.0)

    def apply(self, input):
        return self.motor(input)

    wrapped = Wrapper(motor=captured)(apply, name='wrapped_motor')
    replaced = wrapped.parameterize(ke=5.0)

    assert wrapped.param.ke == 2.0
    assert wrapped.apply(jnp.asarray(3.0)) == 6.0
    assert replaced.param.ke == 5.0
    assert replaced.apply(jnp.asarray(3.0)) == 15.0


def test_wrapper_reparameterizes_a_captured_rng_param_like_an_unbound_member() -> None:
    def RandomScale() -> Node:
        def param(scale, rng):
            return scale * jax.random.normal(rng.next(), ())

        def apply(param, input):
            return param * input

        return Leaf(apply, param=param, name='random_scale')

    def WrappedScale(scale: Node) -> Node:
        def apply(self, input):
            return self.scale(input)

        return Wrapper(scale=scale)(apply, name='wrapped_scale')

    replacement_key = jax.random.PRNGKey(1)
    expected = WrappedScale(RandomScale()).parameterize(
        scale=3.0,
        rng=replacement_key,
    )
    captured = RandomScale().parameterize(
        scale=2.0,
        rng=jax.random.PRNGKey(0),
    )
    replaced = WrappedScale(captured).parameterize(
        scale=3.0,
        rng=replacement_key,
    )

    assert jnp.allclose(replaced.param, expected.param)


def test_pipes_take_constructed_members():
    """Verifies `>>` retains captured member parameters while allowing `parameterize` overrides."""
    pipe = PI(DT)(kp=2.0, ki=0.0) >> PI(DT)
    bound = pipe.parameterize(pi_2=Struct(kp=3.0, ki=0.0))
    assert bound.param.pi.kp == 2.0
    assert bound.param.pi_2.kp == 3.0


def test_member_aux_is_diverted():
    """Verifies `Composite` diverts member `Aux` outputs under member keys while returning primary outputs to `apply`."""
    def loud():
        def apply(input):
            return input * 2.0, Aux(gain=jnp.asarray(2.0))
        return Leaf(apply, name='loud')

    members = Composite(loud=loud())

    def apply(self, input):
        y = self.loud(input)
        return y + 1.0

    node = members(apply, name='aux_comp')
    y, aux = node.apply(jnp.asarray(3.0))
    assert y == 7.0
    assert aux.loud.gain == 2.0


def test_member_methods_bind_live_slices():
    """Verifies member methods accessed on `self` read live parameter and state slices."""
    def Tank():
        def param(capacity):
            return Struct(capacity=jnp.asarray(capacity))
        def init():
            return jnp.asarray(1.0)
        def apply(param, state, input):
            return state - input / param.capacity, state
        def fraction(param, state):
            return state
        return Leaf(apply, init=init, param=param,
                        methods=dict(fraction=fraction), name='tank')

    members = Composite(tank=Tank())

    def apply(self, input):
        before = self.tank.fraction()
        self.tank(input)
        after = self.tank.fraction()
        return Struct(before=before, after=after)

    node = members(apply, name='rig').parameterize(
        tank=Struct(capacity=2.0))
    initial = node.with_input(jnp.asarray(1.0)).bind(node.param).init()
    _, out = node.apply(initial, jnp.asarray(1.0))
    assert out.before == 1.0
    assert out.after == 0.5

    tank = Tank()
    assert tank.node.fraction(Struct(capacity=jnp.asarray(2.0)), 0.25) == 0.25


def test_param_shape_walk_preserves_method_only_members_and_live_slices():
    """Verifies parameter shape discovery exposes member methods and live `param`/`state` slices."""
    def Source():
        def param(scale):
            return Struct(scale=jnp.asarray(scale))

        def init():
            return jnp.ones((3,))

        def apply(param, state, input):
            return state, input

        def value(param, state):
            return param.scale * state

        return Leaf(apply, param=param, init=init,
                    methods=dict(value=value), name='source')

    def ShapeSink():
        def param(node):
            return jnp.zeros_like(node.input)

        def apply(param, input):
            return input + param

        return Leaf(apply, param=param, name='shape_sink')

    members = Composite(
        source=Source().parameterize(scale=2.0), sink=ShapeSink())

    def apply(self, trigger):
        through_method = self.source.value()
        through_slices = self.source.param.scale * self.source.state
        return self.sink(through_method + through_slices + trigger * 0.0)

    bound = members(apply, name='method_shape_walk').with_input(
        bundle=Struct(trigger=jnp.asarray(0.0))).parameterize()
    assert bound.param.sink.shape == (3,)
    assert bound.param.source.scale == 2.0


def test_pure_state_composite_binds_itself():
    """Verifies a `Composite` of parameterless stateful members returns a bound `PNode` directly."""
    from nodejax import PNode
    members = Composite(plant=Servo(DT), delay=Delay())

    def apply(self, input):
        theta = self.plant(input - self.delay.state)
        self.delay(theta)
        return theta

    node = members(apply, name='open_loop')
    assert type(node) is PNode
    assert len(jax.tree.leaves(node.param)) == 0

    initial = node.with_input(jnp.asarray(0.5)).bind(node.param).init()
    state, out = node.apply(initial, jnp.asarray(0.5))
    assert jnp.isfinite(out)
    assert state.delay == out


def test_rng_routes_by_def_and_calls_chain():
    """Verifies `Composite` routes boundary PRNG keys by member definition and chains repeated calls sequentially."""
    def Accumulator():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            new = state + input
            return new, new
        return Leaf(apply, init=init, name='acc')

    def Noise():
        def apply(input, rng):
            return input + jax.random.normal(rng.next())
        return Leaf(apply, name='noise')

    members = Composite(acc=Accumulator(), noise=Noise())

    def apply(self, input):
        self.acc(input)
        total = self.acc(input)
        d1 = self.noise(0.0)
        d2 = self.noise(0.0)
        return Struct(total=total, draws=jnp.stack([d1, d2]))

    node = members(apply, name='chained').parameterize()
    state = node.with_input(jnp.asarray(1.0)).bind(node.param).init()
    new_state, out = node.apply(
        state, jnp.asarray(1.0), rng=jax.random.PRNGKey(0))

    assert out.total == 2.0
    assert new_state.acc == 2.0
    assert out.draws[0] != out.draws[1]


def _running_norm() -> Node:
    """Builds a running-normalization `Leaf` whose `init` reads `node.input`."""
    def init(node):
        return Struct(mean=jnp.zeros_like(node.input[0]), var=jnp.ones_like(node.input[0]))

    def apply(state, input):
        return state, (input - state.mean) / jnp.sqrt(state.var + 1e-5)

    return Leaf(apply, init=init, name='running_norm')


def test_composite_init_threads_input_to_input_required_member():
    # `Composite` initialization infers `node.input` shapes for input-dependent members.
    def cell():
        def apply(self, input):
            return self.tail(self.norm(input))
        tail = Leaf(lambda input: input * 2.0, name='tail')
        return Composite(norm=_running_norm(), tail=tail)(apply, name='cell')

    X = jnp.ones((4, 3))
    node = cell().with_input(X)
    state = node.with_input(X).bind(node.param).init()
    assert state.norm.mean.shape == (3,)
    new_state, out = node.apply(state, X)
    assert out.shape == (4, 3)


def test_composite_recurrent_read_before_feed_is_allowed():
    # Reading a `Delay` state slot before calling `self.mem(...)` succeeds when shapes match.
    from nodejax.control import Delay

    def loop():
        def apply(self, input):
            prev = self.mem.state
            y = input + prev
            self.mem(y)
            return y
        return Composite(mem=Delay().with_input(0.0))(apply, name='loop')

    node = loop().parameterize()
    state = node.with_input(jnp.asarray(1.0)).bind(node.param).init()
    s1, y1 = node.apply(state, jnp.asarray(1.0))
    assert y1 == 1.0 and s1.mem == 1.0


def test_composite_init_shape_instability_raises_at_init():
    # Feeding a `Delay` a shape that conflicts with its bootstrap spec raises `TypeError` at `init`.
    from nodejax.control import Delay

    def grow():
        def apply(self, input):
            p = self.mem.state
            y = jnp.concatenate([jnp.atleast_1d(p), jnp.atleast_1d(input)])
            self.mem(y)
            return y
        return Composite(mem=Delay().with_input(0.0))(apply, name='grow')

    with pytest.raises(TypeError, match='conflicts with its declared spec'):
        grow().with_input(jnp.asarray(1.0)).init()


def test_init_fields_dereference_per_member():
    """Verifies `init` arguments route per member name in both `serial` and `Composite`."""
    def Biased():
        def init(bias=0.0):
            return jnp.asarray(bias)
        def apply(state, input):
            return state, input + state
        return Leaf(apply, init=init, name='biased')

    pipe = serial(a=Biased(), b=Biased()).parameterize()
    st = pipe.init(a=Struct(bias=1.0), b=Struct(bias=2.0))
    assert st.a == 1.0 and st.b == 2.0

    st2 = pipe.init(a=Struct(bias=5.0))
    assert st2.a == 5.0 and st2.b == 0.0

    with pytest.raises(TypeError, match='unknown'):
        pipe.init(a=Struct(bias=1.0), nope=Struct(bias=9.0))

    def apply(self, input):
        return self.x(input)
    comp = Composite(x=Biased(), y=Biased())(apply, name='c').parameterize()
    cs = comp.init(x=Struct(bias=3.0), y=Struct(bias=4.0))
    assert cs.x == 3.0 and cs.y == 4.0


def test_apply_unpacks_input_fields():
    """Verifies `Leaf` `apply` unpacks named data fields separately from the `rng` `KeyStream`."""
    def mixer():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def init():
            return jnp.zeros(())
        def apply(param, state, signal, rng):
            new = state + param.gain * signal + jax.random.normal(rng.next(), ())
            return new, new
        return Leaf(apply, param=param, init=init, name='mixer')

    node = mixer().parameterize(gain=1.0)
    s0 = node.init()
    s1, out = node.apply(s0, signal=jnp.asarray(2.0), rng=jax.random.PRNGKey(0))
    assert out == s1
    assert not jnp.allclose(out, 2.0)



def test_wired_composite_field_signature():
    """Verifies authored `Composite` `apply` unpacks and validates named input fields."""
    def double():
        return Leaf(lambda input: 2.0 * input, name='double')

    def block():
        def apply(self, x, offset):
            return self.double(x) + offset
        return Composite(double=double())(apply, name='block')

    b = block()
    assert b.contract.apply_fields == ('x', 'offset')

    node = b if b.bound else b.parameterize()
    with pytest.raises(TypeError, match='missing required input fields'):
        node.apply(x=3.0)
    with pytest.raises(TypeError, match='missing required input fields'):
        node.apply(3.0)
    assert node.apply(x=3.0, offset=1.0) == 7.0
    with pytest.raises(TypeError, match='unknown input fields'):
        node.apply(x=3.0, y=1.0)


def test_wired_composite_field_signature_rng():
    """Verifies authored `Composite` `apply` shares the boundary PRNG stream between direct draws and member calls."""
    import jax

    def jitter():
        def param(sigma):
            return Struct(sigma=jnp.asarray(sigma))
        def apply(param, x, rng):
            return x + param.sigma * jax.random.normal(rng.next())
        return Leaf(apply, param=param, name='jitter')

    def block():
        def apply(self, x, rng):
            own = jax.random.normal(rng.next())
            return self.jitter(x=x) + 0.001 * own
        return Composite(jitter=jitter())(apply, name='noisy_block')

    node = block().parameterize(jitter=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    a = node.apply(x=1.0, rng=key)
    b = node.apply(x=1.0, rng=key)
    c = node.apply(x=1.0, rng=jax.random.PRNGKey(1))
    assert jnp.allclose(a, b)
    assert not jnp.allclose(a, c)
    assert block().contract.apply_takes_rng


def test_a_nonparametric_pipe_nests_in_a_pipe():
    """Verifies a parameterless `serial` pipeline nests inside an outer `serial` pipeline."""
    def Acc(name):
        def apply(state, input):
            return state + input, state

        return Leaf(apply, init=lambda: jnp.zeros(()), name=name).node

    nested = serial(inner=serial(a=Acc('a'), b=Acc('b')), c=Acc('c')).parameterize()

    state = nested.init()
    assert set(state.__keys__) == {'inner', 'c'}
    assert set(state.inner.__keys__) == {'a', 'b'}

    state, _ = nested.apply(state, jnp.ones(()))
    assert float(state.inner.a) == 1.0


def test_authored_composite_methods_are_available_through_members():
    identity = Leaf(lambda input: input, name='identity')

    def inner_apply(self, input):
        return self.identity(input)

    def doubled(input):
        return 2.0 * input

    inner = Composite(identity=identity)(
        inner_apply,
        methods={'doubled': doubled},
    )

    def outer_apply(self, input):
        return self.inner.doubled(self.inner(input))

    outer = Composite(inner=inner)(outer_apply)
    assert inner.doubled(3.0) == 6.0
    assert outer.apply(3.0) == 6.0


def test_authored_composite_methods_survive_generic_specialization():
    @node
    def Scale(factor: float) -> Node:
        return Leaf(lambda input: factor * input)

    def apply(self, input):
        return self.scale(input)

    generic = Composite(scale=Scale())(
        apply,
        methods={'doubled': lambda input: 2.0 * input},
    )
    complete = generic.specialize(**{'scale.factor': 3.0}).parameterize()

    assert complete.apply(2.0) == 6.0
    assert complete.doubled(2.0) == 4.0


def test_authored_init_accepts_an_empty_slot_for_a_stateless_member():
    """Verifies an authored `Composite` `init` strips empty `()` slots returned for stateless members."""
    @node
    def Held(register: Node, gain: Node) -> Node:
        members = Composite(register=register, gain=gain)

        def apply(self, input):
            return self.register(self.gain(input))

        def init(param, input):
            return Struct(
                register=jnp.zeros(()),
                gain=gain.parameterize().init(),
            )

        return members(apply, init=init)

    def Register() -> Node:
        return Leaf(
            lambda state, input: (input, state),
            init=lambda: jnp.zeros(()),
            name='register',
        ).node

    def Gain() -> Node:
        return Leaf(lambda input: 2.0 * input, name='gain').node

    held = Held(Register(), Gain()).parameterize().initialize(
        input=jnp.zeros(()),
    )
    assert set(held.state.__keys__) == {'register'}

    held, output = held.apply(3.0)
    assert output == 0.0
    assert held.state.register == 6.0


def test_wrapper_init_over_a_stateless_member_keeps_its_state_input():
    """Verifies a `Wrapper` with an explicit `init` retains its `init` call fields over a stateless member."""
    def Double() -> Node:
        return Leaf(lambda input: 2.0 * input, name='double').node

    def apply(self, input):
        return self.double(input)

    def init(param, initial):
        return initial

    adopted = Wrapper(double=Double())(apply, init=init).parameterize()

    assert adopted.cyclic
    started = adopted.initialize(initial=())
    assert started.state == ()
    started, output = started.apply(3.0)
    assert output == 6.0
    assert started.state == ()


def test_authored_self_names_a_parameterless_member_as_empty():
    """Verifies an authored `Composite` `init` exposes `()` on `param` for parameterless members."""
    def Register() -> Node:
        return Leaf(
            lambda state, input: (input, state),
            init=lambda: jnp.zeros(()),
            name='register',
        ).node

    def Gain() -> Node:
        return Leaf(lambda input: 2.0 * input, name='gain').node

    def apply(self, input):
        return self.register(self.gain(input))

    def init(param, input):
        return Struct(
            register=jnp.asarray(1.0),
            gain=param.gain,
        )

    held = Composite(register=Register(), gain=Gain())(
        apply, init=init,
    ).parameterize().initialize(input=jnp.zeros(()))

    assert held.state.register == 1.0
