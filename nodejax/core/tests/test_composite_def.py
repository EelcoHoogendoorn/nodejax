"""Def-level object composition — members are program structure.

The FOC-cascade dogfood of the composite idiom: member nodes enter at
the DEF level, as factory arguments (the param-embedded-PNode
predecessor was reversed).
The doctrine under test: nodes contain nodes, params contain params, and
the PNode type never nests — program containment lives on
Node.members, where every node-level service (rng routing, input
discovery, introspection) reads structure from structure, and the param
tree is plain data, so two bindings of one node share a treedef by
construction.

composite() supplies the member plumbing of a serial pipe (param and
state unioned by name, one rng split toward consumers, per-member input
discovery by tracing the wiring) with the wiring free-form — the
arbitrary DAG that no >> can express. Polymorphism happens where
program choices belong: a block's def is an ordinary factory argument.

Self-contained: framework imports only.
"""

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


# --- the blocks (identical to the param-level sibling's) ---

def PI(dt: float) -> Node:
    """PI controller block: error -> command, integral as state."""
    def param(kp, ki):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))

    def init():
        return jnp.zeros(())

    def apply(param, state, input):
        integral = state + input * dt
        return integral, param.kp * input + param.ki * integral

    return Leaf(apply, init=init, param=param, name='pi')


def Observer(dt: float, alpha: float=0.4) -> Node:
    """Position/velocity observer: differentiate the measured angle,
    EMA-smooth the velocity. Param-less — its slot in the union param
    is (), behavior rides the node on the node side."""
    def init():
        return Struct(theta=jnp.zeros(()), omega=jnp.zeros(()))
    def apply(state, input):
        omega_raw = (input - state.theta) / dt
        omega = (1 - alpha) * state.omega + alpha * omega_raw
        estimate = Struct(theta=input, omega=omega)
        return estimate, estimate
    return Leaf(apply, init=init, name='observer')


def Motor() -> Node:
    """Motor model block, param-only: the back-EMF map from angular
    velocity to compensating voltage. Model data the composite reads is
    itself a member — its constants enter the member channel like every
    other block's."""
    def param(ke):
        return Struct(ke=jnp.asarray(ke))

    def apply(param, input):
        return param.ke * input

    return Leaf(apply, param=param, name='motor')


def Delay() -> Node:
    """One-tick memory block: emits the value stored last step, stores
    its input — the feedback path's unit delay. A memory inits from a
    CONCRETE example (its step-zero value), not a shape read."""
    def init():
        return jnp.zeros(())

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='delay')


def Servo(dt: float, kt: float=1.0, ke: float=0.1, damping: float=0.2, inertia: float=0.5) -> Node:
    """The physical plant: motor + load, back-EMF opposing drive."""
    def init():
        z = jnp.zeros(())
        return Struct(theta=z, omega=z)
    def apply(state, input):
        torque = kt * (input - ke * state.omega)
        omega = state.omega + dt * (torque - damping * state.omega) / inertia
        theta = state.theta + dt * omega
        return Struct(theta=theta, omega=omega), theta
    return Leaf(apply, init=init, name='servo')


# --- the composite: members are factory arguments, apply wires ---

def Cascade(motor: Node, observer: Node, position_ctrl: Node, velocity_ctrl: Node, feedforward: Node=1.0) -> Node:
    """FOC-shaped cascade controller, def-level: the blocks are program
    choices, made where program structure is built — the factory
    signature is the member list. The composite still only knows its
    blocks' calling interface, so any def with the same interface drops
    in. The motor model is a member like the rest — param-only, so its
    slot in the union carries constants and its behavior rides the node;
    parameterize supplies ke through the member channel."""
    members = Composite(motor=motor, observer=observer,
                   position_ctrl=position_ctrl, velocity_ctrl=velocity_ctrl)

    def apply(self, ref, theta):
        # free-form wiring: estimator -> cascaded loops -> model-based ff
        est = self.observer(theta)
        w_ref = self.position_ctrl(ref - est.theta)
        fb_v = self.velocity_ctrl(w_ref - est.omega)
        return fb_v + feedforward * self.motor(est.omega)     # back-EMF compensation

    return members(apply, name='cascade')


def Loop(dt: float, ctrl: Node) -> Node:
    """Close the loop: reference angle in, measured angle out. The
    controller sees Struct(ref, theta) — richer than an error signal,
    which is why this is hand-wired rather than feedback(ctrl >> plant).
    The one-tick measurement memory is a delay member: its stored value
    is read off its state slot, its advance is its ordinary call."""
    members = Composite(ctrl=ctrl, plant=Servo(dt), delay=Delay())

    def apply(self, input):
        v = self.ctrl(ref=input, theta=self.delay.state)
        theta = self.plant(v)
        self.delay(theta)                     # advance the one-tick memory
        return theta

    return members(apply, name='servo_loop')


def stock_loop(position_ctrl: Node, velocity_ctrl: Node) -> Node:
    """Test convenience: the full loop, blocks arriving CONSTRUCTED —
    bound Nodes crossing the factory boundary as transport containers,
    so the member tree is spelled here and nowhere else."""
    return Loop(DT, Cascade(Motor()(ke=0.1), Observer(DT),
                                    position_ctrl, velocity_ctrl))


# --- tests ---

def test_composite_infers_member_param_shapes_from_input():
    # param-side discovery: shapes thread through free-form wiring, so
    # shape-reading members (nn.Linear) infer their fan-in inside a
    # hand-wired composite, not just a pipe
    from nodejax import nn

    def gated(width):
        members = Composite(a=nn.Linear(width), b=nn.Linear(width), gate=nn.Linear(1))

        def apply(self, input):
            g = jax.nn.sigmoid(self.gate(input))
            return g * self.a(input) + (1 - g) * self.b(input)

        return members(apply, name='gated')

    m = gated(4).with_input(jnp.zeros(5)).parameterize(rng=jax.random.PRNGKey(0))
    assert m.param.a.w.shape == (5, 4)          # fan-in 5 inferred at the call site
    assert m.param.gate.w.shape == (5, 1)
    assert m.apply(jnp.zeros(5)).shape == (4,)
    # same def, different input value — shapes live in the params
    m2 = gated(4).with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))
    assert m2.param.a.w.shape == (3, 4)


def test_members_live_on_the_def():
    """Program containment on the node, data containment in the param:
    members are readable off Node.members (nested), the param tree is
    plain Structs of arrays throughout, paths read as the object graph,
    and two bindings of one node share a treedef by construction."""
    loop = stock_loop(PI(DT)(kp=1.0, ki=0.0), PI(DT)(kp=1.0, ki=0.5))
    assert set(loop.members.__keys__) == {'ctrl', 'plant', 'delay'}
    assert set(loop.members.ctrl.members.__keys__) == {'motor', 'observer',
                                                 'position_ctrl', 'velocity_ctrl'}

    bound = loop.parameterize()               # stored constructions fill the tree
    p = bound.param
    assert type(p.ctrl.position_ctrl) is Struct       # plain data, whole tree
    from nodejax import PNode
    assert not any(type(x) is PNode
                   for x in jax.tree.leaves(
                       p, is_leaf=lambda x: type(x) is PNode))
    assert p.ctrl.motor.ke == 0.1                     # the model block's constants
    assert 'observer' not in p.ctrl                   # param-less member: no slot

    paths = {jax.tree_util.keystr(path)
             for path, _ in jax.tree_util.tree_flatten_with_path(p)[0]}
    assert '.ctrl.position_ctrl.kp' in paths
    assert '.ctrl.motor.ke' in paths

    # kwargs override per member; stored constructions fill what they leave open
    other = loop.parameterize(ctrl=Struct(position_ctrl=Struct(kp=9.0, ki=2.0)))
    assert jax.tree.structure(bound) == jax.tree.structure(other)
    assert other.param.ctrl.position_ctrl.kp == 9.0
    assert other.param.ctrl.velocity_ctrl.ki == 0.5

    with pytest.raises(TypeError):                    # unknown names are loud
        loop.parameterize(ctrl=Struct(motr=Struct(ke=0.1),
                                    position_ctrl=Struct(kp=1.0, ki=0.0),
                                    velocity_ctrl=Struct(kp=1.0, ki=0.5)))

    state = bound.with_input(jnp.zeros(())).bind(bound.param).init()
    assert state.ctrl.velocity_ctrl.shape == ()       # named state, composed by init
    assert state.plant.theta.shape == ()
    assert state.delay.shape == ()                    # the memory member's slot

    discovered = bound.with_input(jnp.zeros(())).bind(bound.param).init()
    assert discovered.ctrl.observer.theta.shape == ()


def test_cascade_tracks_a_step():
    """With hand gains, the estimator + cascade + feedforward wiring holds
    the reference: type-1 loop, zero steady-state error."""
    rollout = scanned(stock_loop(PI(DT)(kp=4.0, ki=0.0),
                              PI(DT)(kp=2.0, ki=1.0)))
    bound = rollout.parameterize()

    thetas = bound.apply(jnp.ones(300))
    assert jnp.all(jnp.isfinite(thetas))
    assert jnp.abs(thetas[-1] - 1.0) < 0.02
    assert jnp.max(jnp.abs(thetas)) < 1.6             # bounded overshoot


def test_blocks_are_polymorphic():
    """Polymorphism at the node level: a saturating PI derived from the
    stock block, dropped in as a factory argument — the program choice
    made where the program is built; parameterize sees the same gain
    kwargs either way."""
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
    assert jnp.abs(thetas[-1] - 1.0) < 0.05           # still tracks, saturated actuation


def test_autotune_reaches_into_subblocks():
    """train_step tunes the PI gains inside the plain param tree —
    gradients flow through the hand-wired apply into the member slots."""
    rollout = scanned(stock_loop(PI(DT)(kp=0.5, ki=0.0),
                              PI(DT)(kp=0.5, ki=0.1)))
    trainer = train_step(rollout.parameterize().initialize(), mse, optax.adam(0.02))

    refs = jnp.ones(150)
    steps = 250

    def tile(x):
        return jnp.broadcast_to(x, (steps,) + x.shape)

    final, aux = trained(trainer).apply(input=tile(refs), target=tile(refs))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.5 * aux.loss[0]                        # tracking improved
    assert final.param.ctrl.position_ctrl.kp > 0.5             # the block was tuned
    assert 'observer' not in final.param.ctrl


def test_input_discovery_from_the_wiring():
    """Per-member state shapes derive from the WIRING: composite init
    traces apply once at shape level, records the input each member
    receives at its wire call site, and re-initializes members with
    those specs — built into composite(), driven by the node's members."""
    def Accumulator():
        def init(node):                              # shape genuinely from the wiring
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
            hidden = self.a(input)                   # sees (3,)
            return self.b(self.proj(hidden))         # sees (2,)

        return members(apply, name='comp')

    node = Chain().parameterize(proj=Struct(w=jnp.ones((3, 2))))
    state = node.with_input(jnp.zeros(3)).bind(node.param).init()
    assert state.a.shape == (3,)                     # discovered at the call site
    assert state.b.shape == (2,)                     # NOT the outer input's shape



def test_full_custom_overrides():
    """param= and init= replace the generated constructors wholesale,
    under node's ordinary contracts: the author owns the whole tree,
    member slots included — for layouts the member union cannot spell.
    Here one gain fills BOTH cascaded blocks — tied at CONSTRUCTION:
    the two slots are separate leaves thereafter, each with its own
    gradient (ties that survive training are the tie transform's
    business). The declared ctor signature makes unknown names loud.
    apply takes the raw contract triple — the all-custom corner:
    helper stood down, member states threaded by hand through the
    nodes' own apply_fns."""
    # The raw contract exposes no member-call routing, so member input
    # relations are explicit at construction.
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
    assert out == 1.0                                # kp=1, ki=0: identity twice

    with pytest.raises(TypeError):                   # declared ctor: typos are loud
        tied.parameterize(kp=1.0, kii=0.0)


def test_wrapper_is_a():
    """wrapper: is-a adaptation of one node — param, state, paths and
    init are the inner's, flat, and the apply signature names the
    member by its first parameter. The has-a spelling of the same
    shape is a singleton composite, one nesting level deeper with the
    member named in every path."""
    def apply(self, input):
        return jnp.clip(self.pi(input), -1.0, 1.0)

    sat = Wrapper(pi=PI(DT))(apply, name='sat_pi')
    assert sat.members.__keys__ == ('pi',)
    bound = sat.parameterize(kp=4.0, ki=1.0)          # the pi's ctor, unchanged
    assert '.kp' in {jax.tree_util.keystr(p)
                     for p, _ in jax.tree_util.tree_flatten_with_path(bound.param)[0]}

    state = bound.with_input(jnp.asarray(10.0)).bind(bound.param).init()
    s2, out = bound.apply(state, jnp.asarray(10.0))
    assert out == 1.0                                 # clipped actuation
    assert s2 != state                                # the integral advanced


def test_pipes_take_constructed_members():
    """Transport containers work in >> too: a bound member's params
    become stored construction values, parameterize fills what kwargs
    leave open — the member tree spelled once, at the composition
    site."""
    pipe = PI(DT)(kp=2.0, ki=0.0) >> PI(DT)
    bound = pipe.parameterize(pi_2=Struct(kp=3.0, ki=0.0))
    assert bound.param.pi.kp == 2.0                  # stored construction
    assert bound.param.pi_2.kp == 3.0                # kwarg-supplied


def test_member_aux_is_diverted():
    """A member emitting (output, aux) has the aux diverted: the wiring
    receives the clean signal, and the composite re-emits
    (output, Aux(<aux by member name>)) — core.split_aux's
    composite doctrine, self-shaped, nesting like the pipes'."""
    def loud():
        def apply(input):
            return input * 2.0, Aux(gain=jnp.asarray(2.0))
        return Leaf(apply, name='loud')

    members = Composite(loud=loud())

    def apply(self, input):
        y = self.loud(input)                         # the clean signal
        return y + 1.0

    node = members(apply, name='aux_comp')
    y, aux = node.apply(jnp.asarray(3.0))
    assert y == 7.0
    assert aux.loud.gain == 2.0


def test_member_methods_bind_live_slices():
    """Methods reached through self bind the live slices by channel
    name: leading param and state inject, state chained, so a read
    after a step sees the advance. Unbound calls through the node take
    the slots explicitly."""
    def Tank():
        def param(capacity):
            return Struct(capacity=jnp.asarray(capacity))
        def init():
            return jnp.asarray(1.0)
        def apply(param, state, input):
            return state - input / param.capacity, state
        def fraction(param, state):
            return state                     # charge fraction: a pure read
        return Leaf(apply, init=init, param=param,
                        methods=dict(fraction=fraction), name='tank')

    members = Composite(tank=Tank())

    def apply(self, input):
        before = self.tank.fraction()
        self.tank(input)
        after = self.tank.fraction()         # live: sees the advance
        return Struct(before=before, after=after)

    node = members(apply, name='rig').parameterize(
        tank=Struct(capacity=2.0))
    initial = node.with_input(jnp.asarray(1.0)).bind(node.param).init()
    _, out = node.apply(initial, jnp.asarray(1.0))
    assert out.before == 1.0
    assert out.after == 0.5

    tank = Tank()                        # unbound: slices explicit
    assert tank.node.fraction(Struct(capacity=jnp.asarray(2.0)), 0.25) == 0.25


def test_param_shape_walk_preserves_method_only_members_and_live_slices():
    """Running authored wiring to discover a downstream parameter shape must
    expose the same member surface as runtime wiring.  A member consulted only
    through a method is still a member, and direct param/state reads remain
    available while another member learns the shape of their result."""
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
    """A composite of non-parametric members is itself non-parametric:
    composite() returns the bound PNode directly (node's convention),
    its param the leafless Struct of member slots."""
    from nodejax import PNode
    members = Composite(plant=Servo(DT), delay=Delay())

    def apply(self, input):
        theta = self.plant(input - self.delay.state)  # unit feedback via the memory
        self.delay(theta)
        return theta

    node = members(apply, name='open_loop')
    assert type(node) is PNode
    assert len(jax.tree.leaves(node.param)) == 0

    initial = node.with_input(jnp.asarray(0.5)).bind(node.param).init()
    state, out = node.apply(initial, jnp.asarray(0.5))
    assert jnp.isfinite(out)
    assert state.delay == out                         # the memory advanced


def test_rng_routes_by_def_and_calls_chain():
    """A single boundary key splits toward the members whose nodes declare
    they consume RNG keys — read off the node alone, with the param tree
    never consulted. Repeated wire calls to one member stay SEQUENTIAL:
    an accumulator called twice advances twice, a stochastic member
    draws fresh noise per call."""
    def Accumulator():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            new = state + input
            return new, new
        return Leaf(apply, init=init, name='acc')

    def Noise():
        def init(rng):
            return Struct(rng=rng)
        def apply(state, input):
            return state, input + jax.random.normal(state.rng)
        return Leaf(apply, init=init, name='noise')

    members = Composite(acc=Accumulator(), noise=Noise())

    def apply(self, input):
        self.acc(input)
        total = self.acc(input)                      # second step, not a redo
        d1 = self.noise(0.0)
        d2 = self.noise(0.0)                         # fresh draw, not a copy
        return Struct(total=total, draws=jnp.stack([d1, d2]))

    node = members(apply, name='chained').parameterize()
    state = node.with_input(jnp.asarray(1.0)).bind(node.param).init(
        rng=jax.random.PRNGKey(0))
    new_state, out = node.apply(state, jnp.asarray(1.0))

    assert out.total == 2.0                          # accumulated across calls
    assert new_state.acc == 2.0                      # ...and collected
    assert out.draws[0] != out.draws[1]              # independent noise per call


def _running_norm() -> Node:
    """A member whose init REQUIRES an input (no default): its state
    shape is only knowable from the value it is fed."""
    def init(node):
        return Struct(mean=jnp.zeros_like(node.input[0]), var=jnp.ones_like(node.input[0]))

    def apply(state, input):
        return state, (input - state.mean) / jnp.sqrt(state.var + 1e-5)

    return Leaf(apply, init=init, name='running_norm')


def test_composite_init_threads_input_to_input_required_member():
    # a member whose init requires input inits inside a composite, its
    # shape discovered from the wiring (init threads the example through
    # apply, the state-side twin of param discovery)
    def cell():
        def apply(self, input):
            return self.tail(self.norm(input))
        tail = Leaf(lambda input: input * 2.0, name='tail')
        return Composite(norm=_running_norm(), tail=tail)(apply, name='cell')

    X = jnp.ones((4, 3))
    node = cell().with_input(X)
    state = node.with_input(X).bind(node.param).init()
    assert state.norm.mean.shape == (3,)             # shaped by the fed value
    new_state, out = node.apply(state, X)
    assert out.shape == (4, 3)


def test_composite_recurrent_read_before_feed_is_allowed():
    # a delay read before it is fed (a feedback loop) is legitimate when
    # the init state is shape-stable across a step
    from nodejax.control import Delay

    def loop():
        def apply(self, input):
            prev = self.mem.state                     # read last step's value
            y = input + prev
            self.mem(y)                               # then store this step's
            return y
        return Composite(mem=Delay().with_input(0.0))(apply, name='loop')

    node = loop().parameterize()
    state = node.with_input(jnp.asarray(1.0)).bind(node.param).init()
    s1, y1 = node.apply(state, jnp.asarray(1.0))
    assert y1 == 1.0 and s1.mem == 1.0


def test_composite_init_shape_instability_raises_at_init():
    # a delay fed a shape that differs from its declared bootstrap is a
    # named conflict, caught loudly at init, not as an obscure scan error
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
    """A non-reserved init argument routes to its member by name, the way
    param already does — dereferenced per member, never broadcast to every
    member. Input and node metadata keep their own routing, while RNG travels
    in its explicit frame rather than in this data record."""
    def Biased():
        # a cyclic node whose state builds from a non-reserved init arg
        def init(bias=0.0):
            return jnp.asarray(bias)
        def apply(state, input):
            return state, input + state
        return Leaf(apply, init=init, name='biased')

    # serial pipe path (the pipe init_fn)
    pipe = serial(a=Biased(), b=Biased()).parameterize()
    st = pipe.init(a=Struct(bias=1.0), b=Struct(bias=2.0))
    assert st.a == 1.0 and st.b == 2.0            # each member its own value

    # a member given nothing keeps its init default: no broadcast leak
    st2 = pipe.init(a=Struct(bias=5.0))
    assert st2.a == 5.0 and st2.b == 0.0

    # an unknown member name is loud, not a silent drop
    with pytest.raises(TypeError, match='unknown'):
        pipe.init(a=Struct(bias=1.0), nope=Struct(bias=9.0))

    # self-form composite path (_member_init) routes the same way
    def apply(self, input):
        return self.x(input)
    comp = Composite(x=Biased(), y=Biased())(apply, name='c').parameterize()
    cs = comp.init(x=Struct(bias=3.0), y=Struct(bias=4.0))
    assert cs.x == 3.0 and cs.y == 4.0


def test_apply_unpacks_input_fields():
    """Apply's trailing data parameters are fields unpacked by name.

    A trailing authored ``rng`` parameter is instead extracted into the static
    RNG plan and receives the authoring ``KeyStream`` from the explicit frame.
    Signature compilation declares both facts without runtime tracing.
    """
    def mixer():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def init():
            return jnp.zeros(())
        def apply(param, state, signal, rng):       # signal is data; rng declares entropy
            new = state + param.gain * signal + jax.random.normal(rng.next(), ())
            return new, new
        return Leaf(apply, param=param, init=init, name='mixer')

    node = mixer().parameterize(gain=1.0)
    s0 = node.init()
    s1, out = node.apply(s0, signal=jnp.asarray(2.0), rng=jax.random.PRNGKey(0))
    assert out == s1                                 # cyclic (state, output)
    assert not jnp.allclose(out, 2.0)                # noise from the input rng added



def test_wired_composite_field_signature():
    """A wired apply shares the leaf sugar: trailing field names unpack
    the input bundle, every declared input is required, and validation is the
    ordinary loud kind."""
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
    # the calling convention is loud: an unknown field is an error, not
    # a shrug
    with pytest.raises(TypeError, match='unknown input fields'):
        node.apply(x=3.0, y=1.0)


def test_wired_composite_field_signature_rng():
    """A declared authored rng role receives the boundary stream.

    Author draws and member-call routing share the frame's stream while the
    ordinary input bundle remains data-only.
    """
    import jax

    def jitter():
        def param(sigma):
            return Struct(sigma=jnp.asarray(sigma))
        def apply(param, x, rng):
            return x + param.sigma * jax.random.normal(rng.next())
        return Leaf(apply, param=param, name='jitter')

    def block():
        def apply(self, x, rng):
            own = jax.random.normal(rng.next())          # the author's draw
            return self.jitter(x=x) + 0.001 * own        # the member's draw, injected
        return Composite(jitter=jitter())(apply, name='noisy_block')

    node = block().parameterize(jitter=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    a = node.apply(x=1.0, rng=key)
    b = node.apply(x=1.0, rng=key)
    c = node.apply(x=1.0, rng=jax.random.PRNGKey(1))
    assert jnp.allclose(a, b)                    # same key, same draws
    assert not jnp.allclose(a, c)
    assert block().contract.apply_takes_rng               # sig-declared, flag-visible


def test_a_nonparametric_pipe_nests_in_a_pipe():
    """A composite's trivial param is (), not a Struct keyed by member, so
    indexing it by name is a tuple index. sum_junction and parallel each had a
    private guard for that; the pipe did not, and a NON-parametric pipe inside
    another pipe died on

        pipe member 'inner': tuple indices must be integers or slices, not str

    The parametric case worked, which is why nothing caught it: every nested
    pipe anyone had written happened to hold weights.

    Found by refactoring one leaf into two composed nodes, which is the shape
    of change the library claims not to leak upward."""
    def Acc(name):
        def apply(state, input):
            return state + input, state

        return Leaf(apply, init=lambda: jnp.zeros(()), name=name).node

    nested = serial(inner=serial(a=Acc('a'), b=Acc('b')), c=Acc('c')).parameterize()

    state = nested.init()
    assert set(state.__keys__) == {'inner', 'c'}
    assert set(state.inner.__keys__) == {'a', 'b'}

    state, _ = nested.apply(state, jnp.ones(()))
    assert float(state.inner.a) == 1.0          # the pipe threads as it always did


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
    """An authored init may name every member uniformly: an explicit empty
    slot for a stateless member is accepted and stripped, so the stored
    state stays canonically sparse; a non-empty claim stays a loud error."""
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
    """An adopt-this-field init may be declared unconditionally: over a
    stateless member the state it adopts is the empty slot, but its
    state-input field stays part of the call form, so callers never fork
    on the member's lifecycle."""
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
    """The authored self answers the empty slot for a declared member
    without parameters, the param-side mirror of the empty state slot, so
    authored inits never fork on a member's parametricity."""
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
