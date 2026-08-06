"""Def-level object composition — members are program structure.

The FOC-cascade dogfood of the composite idiom: member defs enter at
the DEF level, as factory arguments (the param-embedded-Node
predecessor was reversed).
The doctrine under test: defs contain defs, params contain params, and
the Node type never nests — program containment lives on
NodeDef.members, where every def-level service (rng routing, input
discovery, introspection) reads structure from structure, and the param
tree is plain data, so two bindings of one def share a treedef by
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

from nodejax import node_def, derive, scan, train_step, composite, composite_init, wrapper, serial
from nodejax import REQUIRED
from nodejax.struct import Struct

DT = 0.02


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


# --- the blocks (identical to the param-level sibling's) ---

def pi_def(dt):
    """PI controller block: error -> command, integral as state."""
    def param(kp, ki):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))

    def init():
        return jnp.zeros(())

    def apply(param, state, input):
        integral = state + input * dt
        return integral, param.kp * input + param.ki * integral

    return node_def(apply, init=init, param=param, name='pi')


def observer_def(dt, alpha=0.4):
    """Position/velocity observer: differentiate the measured angle,
    EMA-smooth the velocity. Param-less — its slot in the union param
    is (), behavior rides the def on the def side."""
    def init():
        return Struct(theta=jnp.zeros(()), omega=jnp.zeros(()))
    def apply(state, input):
        omega_raw = (input - state.theta) / dt
        omega = (1 - alpha) * state.omega + alpha * omega_raw
        estimate = Struct(theta=input, omega=omega)
        return estimate, estimate
    return node_def(apply, init=init, name='observer')


def motor_def():
    """Motor model block, param-only: the back-EMF map from angular
    velocity to compensating voltage. Model data the composite reads is
    itself a member — its constants enter the member channel like every
    other block's."""
    def param(ke):
        return Struct(ke=jnp.asarray(ke))

    def apply(param, input):
        return param.ke * input

    return node_def(apply, param=param, name='motor')


def delay_def():
    """One-tick memory block: emits the value stored last step, stores
    its input — the feedback path's unit delay. A memory inits from a
    CONCRETE example (its step-zero value), not a shape read."""
    def init():
        return jnp.zeros(())

    def apply(state, input):
        return input, state

    return node_def(apply, init=init, name='delay')


def servo_def(dt, kt=1.0, ke=0.1, damping=0.2, inertia=0.5):
    """The physical plant: motor + load, back-EMF opposing drive."""
    def init():
        z = jnp.zeros(())
        return Struct(theta=z, omega=z)
    def apply(state, input):
        torque = kt * (input - ke * state.omega)
        omega = state.omega + dt * (torque - damping * state.omega) / inertia
        theta = state.theta + dt * omega
        return Struct(theta=theta, omega=omega), theta
    return node_def(apply, init=init, name='servo')


# --- the composite: members are factory arguments, apply wires ---

def cascade_def(motor, observer, position_ctrl, velocity_ctrl, feedforward=1.0):
    """FOC-shaped cascade controller, def-level: the blocks are program
    choices, made where program structure is built — the factory
    signature is the member list. The composite still only knows its
    blocks' calling interface, so any def with the same interface drops
    in. The motor model is a member like the rest — param-only, so its
    slot in the union carries constants and its behavior rides the def;
    parameterize supplies ke through the member channel."""
    members = dict(motor=motor, observer=observer,
                   position_ctrl=position_ctrl, velocity_ctrl=velocity_ctrl)

    def apply(self, input):
        # free-form wiring: estimator -> cascaded loops -> model-based ff
        est = self.observer(input.theta)
        w_ref = self.position_ctrl(input.ref - est.theta)
        v_fb = self.velocity_ctrl(w_ref - est.omega)
        return v_fb + feedforward * self.motor(est.omega)     # back-EMF compensation

    return composite(apply, members=members, name='cascade')


def loop_def(dt, ctrl):
    """Close the loop: reference angle in, measured angle out. The
    controller sees Struct(ref, theta) — richer than an error signal,
    which is why this is hand-wired rather than feedback(ctrl >> plant).
    The one-tick measurement memory is a delay member: its stored value
    is read off its state slot, its advance is its ordinary call."""
    members = dict(ctrl=ctrl, plant=servo_def(dt), delay=delay_def())

    def apply(self, input):
        v = self.ctrl(Struct(ref=input, theta=self.state.delay))
        theta = self.plant(v)
        self.delay(theta)                     # advance the one-tick memory
        return theta

    return composite(apply, members=members, name='servo_loop')


def stock_loop(position_ctrl, velocity_ctrl):
    """Test convenience: the full loop, blocks arriving CONSTRUCTED —
    bound Nodes crossing the factory boundary as transport containers,
    so the member tree is spelled here and nowhere else."""
    return loop_def(DT, cascade_def(motor_def()(ke=0.1), observer_def(DT),
                                    position_ctrl, velocity_ctrl))


# --- tests ---

def test_composite_infers_member_param_shapes_from_input():
    # param-side discovery: shapes thread through free-form wiring, so
    # shape-reading members (nn.linear) infer their fan-in inside a
    # hand-wired composite, not just a pipe
    from nodejax import nn

    def gated(width):
        members = dict(a=nn.linear(width), b=nn.linear(width), gate=nn.linear(1))

        def apply(self, input):
            g = jax.nn.sigmoid(self.gate(input))
            return g * self.a(input) + (1 - g) * self.b(input)

        return composite(apply, members=members, name='gated')

    m = gated(4).with_input(jnp.zeros(5)).parameterize(rng=jax.random.PRNGKey(0))
    assert m.param.a.w.shape == (5, 4)          # fan-in 5 inferred at the call site
    assert m.param.gate.w.shape == (5, 1)
    assert m.apply(jnp.zeros(5)).shape == (4,)
    # same def, different input value — shapes live in the params
    m2 = gated(4).with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))
    assert m2.param.a.w.shape == (3, 4)


def test_members_live_on_the_def():
    """Program containment on the def, data containment in the param:
    members are readable off NodeDef.members (nested), the param tree is
    plain Structs of arrays throughout, paths read as the object graph,
    and two bindings of one def share a treedef by construction."""
    loop = stock_loop(pi_def(DT)(kp=1.0, ki=0.0), pi_def(DT)(kp=1.0, ki=0.5))
    assert set(loop.members) == {'ctrl', 'plant', 'delay'}
    assert set(loop.members['ctrl'].members) == {'motor', 'observer',
                                                 'position_ctrl', 'velocity_ctrl'}

    bound = loop.parameterize()               # stored constructions fill the tree
    p = bound.param
    assert isinstance(p.ctrl.position_ctrl, Struct)   # plain data, whole tree
    from nodejax import Node
    assert not any(isinstance(x, Node)
                   for x in jax.tree.leaves(p, is_leaf=lambda x: isinstance(x, Node)))
    assert p.ctrl.motor.ke == 0.1                     # the model block's constants
    assert p.ctrl.observer == ()                      # param-less member: empty slot

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

    state = bound.with_input(jnp.zeros(())).init()
    assert state.ctrl.velocity_ctrl.shape == ()       # named state, composed by init
    assert state.plant.theta.shape == ()
    assert state.delay.shape == ()                    # the memory member's slot

    discovered = bound.with_input(jnp.zeros(())).init()   # nested per-member discovery
    assert discovered.ctrl.observer.theta.shape == ()


def test_cascade_tracks_a_step():
    """With hand gains, the estimator + cascade + feedforward wiring holds
    the reference: type-1 loop, zero steady-state error."""
    rollout = scan(stock_loop(pi_def(DT)(kp=4.0, ki=0.0),
                              pi_def(DT)(kp=2.0, ki=1.0)))
    bound = rollout.parameterize()

    thetas = bound.apply(jnp.ones(300))
    assert jnp.all(jnp.isfinite(thetas))
    assert jnp.abs(thetas[-1] - 1.0) < 0.02
    assert jnp.max(jnp.abs(thetas)) < 1.6             # bounded overshoot


def test_blocks_are_polymorphic():
    """Polymorphism at the def level: a saturating PI derived from the
    stock block, dropped in as a factory argument — the program choice
    made where the program is built; parameterize sees the same gain
    kwargs either way."""
    base = pi_def(DT)

    def apply(param, state, input):
        state, u = base.apply_fn(param, state, input)
        return state, jnp.clip(u, -2.0, 2.0)

    SatPI = derive(base, apply=apply, name='sat_pi')

    rollout = scan(stock_loop(pi_def(DT)(kp=4.0, ki=0.0),
                              SatPI(kp=2.0, ki=1.0)))
    bound = rollout.parameterize()

    thetas = bound.apply(jnp.ones(300))
    assert jnp.all(jnp.isfinite(thetas))
    assert jnp.abs(thetas[-1] - 1.0) < 0.05           # still tracks, saturated actuation


def test_autotune_reaches_into_subblocks():
    """train_step tunes the PI gains inside the plain param tree —
    gradients flow through the hand-wired apply into the member slots."""
    rollout = scan(stock_loop(pi_def(DT)(kp=0.5, ki=0.0),
                              pi_def(DT)(kp=0.5, ki=0.1)))
    trainer = train_step(rollout, mse, optax.adam(0.02))

    weak = rollout.parameterize()
    refs = jnp.ones(150)
    steps = 250

    def tile(x):
        return jnp.broadcast_to(x, (steps,) + x.shape)

    final, losses = trainer.scan(trainer.init(model=weak.param), Struct(input=tile(refs), target=tile(refs)))

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]                        # tracking improved
    assert final.model.ctrl.position_ctrl.kp > 0.5             # the block was tuned
    assert len(jax.tree.leaves(final.model.ctrl.observer)) == 0


def test_input_discovery_from_the_wiring():
    """Per-member state shapes derive from the WIRING: composite init
    traces apply once at shape level, records the input each member
    receives at its wire call site, and re-initializes members with
    those specs — built into composite(), driven by the def's members."""
    def acc_def():
        def init(ndef):                              # shape genuinely from the wiring
            return jnp.zeros_like(ndef.input)
        def apply(state, input):
            new = state + input
            return new, new
        return node_def(apply, init=init, name='acc')

    def proj_def():
        def param(w):
            return Struct(w=jnp.asarray(w))
        def apply(param, input):
            return input @ param.w
        return node_def(apply, param=param, name='proj')

    def comp_def():
        members = dict(a=acc_def(), proj=proj_def(), b=acc_def())

        def apply(self, input):
            hidden = self.a(input)                   # sees (3,)
            return self.b(self.proj(hidden))         # sees (2,)

        return composite(apply, members=members, name='comp')

    node = comp_def().parameterize(proj=Struct(w=jnp.ones((3, 2))))
    state = node.with_input(jnp.zeros(3)).init()
    assert state.a.shape == (3,)                     # discovered at the call site
    assert state.b.shape == (2,)                     # NOT the outer input's shape



def test_full_custom_overrides():
    """param= and init= replace the generated constructors wholesale,
    under node_def's ordinary contracts: the author owns the whole tree,
    member slots included — for layouts the member union cannot spell.
    Here one gain seeds BOTH cascaded blocks — tied at CONSTRUCTION:
    the two slots are separate leaves thereafter, each with its own
    gradient (ties that survive training are the tie transform's
    business). The declared ctor signature makes unknown names loud.
    apply takes the raw contract triple — the all-custom corner:
    helper stood down, member states threaded by hand through the
    defs' own apply_fns."""
    members = dict(coarse=pi_def(DT), fine=pi_def(DT))

    def param(kp, ki):
        gains = Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))
        return Struct(coarse=gains, fine=gains)

    def init():
        return Struct(coarse=jnp.zeros(()), fine=jnp.zeros(()))

    def apply(param, state, input):
        s1, mid = members['coarse'].ndef.apply(param.coarse, state.coarse, input)
        s2, out = members['fine'].ndef.apply(param.fine, state.fine, mid)
        return Struct(coarse=s1, fine=s2), out

    tied = composite(apply, members=members, param=param, init=init, name='tied')
    bound = tied.parameterize(kp=1.0, ki=0.0)
    assert bound.param.coarse.kp == bound.param.fine.kp

    _, out = bound.apply(bound.with_input(jnp.asarray(1.0)).init(), jnp.asarray(1.0))
    assert out == 1.0                                # kp=1, ki=0: identity twice

    with pytest.raises(TypeError):                   # declared ctor: typos are loud
        tied.parameterize(kp=1.0, kii=0.0)


def test_wrapper_is_a():
    """wrapper: is-a adaptation of one node — param, state, paths and
    init are the inner's, flat, and the apply signature names the
    member by its first parameter. The has-a spelling of the same
    shape is a singleton composite, one nesting level deeper with the
    member named in every path."""
    def apply(pid, input):
        return jnp.clip(pid(input), -1.0, 1.0)

    sat = wrapper(apply, pi_def(DT), name='sat_pi')
    bound = sat.parameterize(kp=4.0, ki=1.0)          # the pi's ctor, unchanged
    assert '.kp' in {jax.tree_util.keystr(p)
                     for p, _ in jax.tree_util.tree_flatten_with_path(bound.param)[0]}

    state = bound.with_input(jnp.asarray(10.0)).init()   # the pi's init, unchanged
    s2, out = bound.apply(state, jnp.asarray(10.0))
    assert out == 1.0                                 # clipped actuation
    assert s2 != state                                # the integral advanced


def test_pipes_take_constructed_members():
    """Transport containers work in >> too: a bound member's params
    become stored construction values, parameterize fills what kwargs
    leave open — the member tree spelled once, at the composition
    site."""
    pipe = pi_def(DT)(kp=2.0, ki=0.0) >> pi_def(DT)
    bound = pipe.parameterize(pi_2=Struct(kp=3.0, ki=0.0))
    assert bound.param.pi.kp == 2.0                  # stored construction
    assert bound.param.pi_2.kp == 3.0                # kwarg-supplied


def test_member_aux_is_diverted():
    """A member emitting (output, aux) has the aux diverted: the wiring
    receives the clean signal, and the composite re-emits
    (output, Struct(<aux by member name>)) — core.split_aux's
    composite doctrine, self-shaped, nesting like the pipes'."""
    def loud_def():
        def apply(input):
            return input * 2.0, Struct(gain=jnp.asarray(2.0))
        return node_def(apply, name='loud')

    members = dict(loud=loud_def())

    def apply(self, input):
        y = self.loud(input)                         # the clean signal
        return y + 1.0

    node = composite(apply, members=members, name='aux_comp')
    y, aux = node.apply(jnp.asarray(3.0))
    assert y == 7.0
    assert aux.loud.gain == 2.0


def test_self_rng_stream_advances():
    """Entropy through the step object: self.rng wraps the state's rng
    key as a scope-local KeyStream, and the drawn stream's successor
    folds back into the rng slot at collect — one key enters at init
    (a KeyStream in the init's hands too), fresh draws every step,
    jax.random.split nowhere in user code."""
    def lag_def():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            return input, state
        return node_def(apply, init=init, name='lag')

    members = dict(lag=lag_def())

    def init(param, rng):
        states = composite_init(members, apply, param)
        return states.replace(rng=rng.next())

    def apply(self, input):
        draw = jax.random.normal(self.rng.next())
        self.lag(draw)
        return draw

    node = composite(apply, members=members, init=init, name='noisy_comp')

    state = node.init(rng=jax.random.PRNGKey(0))
    s1, d1 = node.apply(state, jnp.asarray(0.0))
    s2, d2 = node.apply(s1, jnp.asarray(0.0))
    assert d1 != d2                        # the fold-back advanced the stream
    assert jnp.any(s1.rng != state.rng)    # the successor was stored
    assert s2.lag == d2                    # member advances rode the same step


def test_member_methods_bind_live_slices():
    """Methods reached through self bind the live param slice, and a
    second parameter NAMED `state` declares the state role and binds
    the member's live slice — chained, so a read after a step sees the
    advance. Unbound calls through the def take both explicitly."""
    def tank_def():
        def param(capacity):
            return Struct(capacity=jnp.asarray(capacity))
        def init():
            return jnp.asarray(1.0)
        def apply(param, state, input):
            return state - input / param.capacity, state
        def fraction(param, state):
            return state                     # charge fraction: a pure read
        return node_def(apply, init=init, param=param,
                        methods=dict(fraction=fraction), name='tank')

    members = dict(tank=tank_def())

    def apply(self, input):
        before = self.tank.fraction()
        self.tank(input)
        after = self.tank.fraction()         # live: sees the advance
        return Struct(before=before, after=after)

    node = composite(apply, members=members, name='rig').parameterize(
        tank=Struct(capacity=2.0))
    _, out = node.apply(node.with_input(jnp.asarray(1.0)).init(), jnp.asarray(1.0))
    assert out.before == 1.0
    assert out.after == 0.5

    tank = tank_def()                        # unbound: slices explicit
    assert tank.ndef.fraction(Struct(capacity=jnp.asarray(2.0)), 0.25) == 0.25


def test_collect_closes_the_step():
    """collect reads the final member states and closes the step: a
    member call after it raises, because its advance could never reach
    the collected state. The self form collects automatically at
    return; an explicit early collect closes the step the same way."""
    def acc_def():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            new = state + input
            return new, new
        return node_def(apply, init=init, name='acc')

    members = dict(acc=acc_def())

    def apply(self, input):
        self.acc(input)
        closed = self.collect()
        with pytest.raises(TypeError):
            self.acc(input)
        return closed.acc

    node = composite(apply, members=members, name='guarded')
    state, out = node.apply(node.with_input(jnp.asarray(1.0)).init(), jnp.asarray(1.0))
    assert out == 1.0
    assert state.acc == 1.0


def test_pure_state_composite_binds_itself():
    """A composite of non-parametric members is itself non-parametric:
    composite() returns the bound Node directly (node_def's convention),
    its param the leafless Struct of member slots."""
    from nodejax import Node
    members = dict(plant=servo_def(DT), delay=delay_def())

    def apply(self, input):
        theta = self.plant(input - self.state.delay)  # unit feedback via the memory
        self.delay(theta)
        return theta

    node = composite(apply, members=members, name='open_loop')
    assert isinstance(node, Node)
    assert len(jax.tree.leaves(node.param)) == 0

    state, out = node.apply(node.with_input(jnp.asarray(0.5)).init(), jnp.asarray(0.5))
    assert jnp.isfinite(out)
    assert state.delay == out                         # the memory advanced


def test_rng_routes_by_def_and_calls_chain():
    """A single boundary key splits toward the members whose defs declare
    they consume entropy — read off the def alone, with the param tree
    never consulted. Repeated wire calls to one member stay SEQUENTIAL:
    an accumulator called twice advances twice, a stochastic member
    draws fresh noise per call."""
    def acc_def():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            new = state + input
            return new, new
        return node_def(apply, init=init, name='acc')

    def noise_def():
        def init(rng):
            return Struct(rng=rng)
        def apply(state, input):
            return state, input + jax.random.normal(state.rng)
        return node_def(apply, init=init, name='noise')

    members = dict(acc=acc_def(), noise=noise_def())

    def apply(self, input):
        self.acc(input)
        total = self.acc(input)                      # second step, not a redo
        d1 = self.noise(0.0)
        d2 = self.noise(0.0)                         # fresh draw, not a copy
        return Struct(total=total, draws=jnp.stack([d1, d2]))

    node = composite(apply, members=members, name='chained').parameterize()
    state = node.with_input(jnp.asarray(1.0)).init(rng=jax.random.PRNGKey(0))
    new_state, out = node.apply(state, jnp.asarray(1.0))

    assert out.total == 2.0                          # accumulated across calls
    assert new_state.acc == 2.0                      # ...and collected
    assert out.draws[0] != out.draws[1]              # independent noise per call


def _running_norm():
    """A member whose init REQUIRES an input (no default): its state
    shape is only knowable from the value it is fed."""
    def init(ndef):
        return Struct(mean=jnp.zeros_like(ndef.input[0]), var=jnp.ones_like(ndef.input[0]))

    def apply(state, input):
        return state, (input - state.mean) / jnp.sqrt(state.var + 1e-5)

    return node_def(apply, init=init, name='running_norm')


def test_composite_init_threads_input_to_input_required_member():
    # a member whose init requires input inits inside a composite, its
    # shape discovered from the wiring (init threads the example through
    # apply, the state-side twin of param discovery)
    def cell():
        def apply(self, input):
            return self.tail(self.norm(input))
        tail = node_def(lambda input: input * 2.0, name='tail')
        return composite(apply, members=dict(norm=_running_norm(), tail=tail), name='cell')

    X = jnp.ones((4, 3))
    node = cell().with_input(X).parameterize()
    state = node.with_input(X).init()
    assert state.norm.mean.shape == (3,)             # shaped by the fed value
    new_state, out = node.apply(state, X)
    assert out.shape == (4, 3)


def test_composite_recurrent_read_before_feed_is_allowed():
    # a delay read before it is fed (a feedback loop) is legitimate when
    # the init state is shape-stable across a step
    from nodejax.examples.actuator.blocks import delay_def

    def loop():
        def apply(self, input):
            prev = self.state.mem                     # read last step's value
            y = input + prev
            self.mem(y)                               # then store this step's
            return y
        return composite(apply, members=dict(mem=delay_def(0.0)), name='loop')

    node = loop().parameterize()
    state = node.with_input(jnp.asarray(1.0)).init()
    s1, y1 = node.apply(state, jnp.asarray(1.0))
    assert y1 == 1.0 and s1.mem == 1.0


def test_composite_init_shape_instability_raises_at_init():
    # a delay fed a shape that differs from its declared bootstrap is a
    # named conflict, caught loudly at init, not as an obscure scan error
    from nodejax.examples.actuator.blocks import delay_def

    def grow():
        def apply(self, input):
            p = self.state.mem
            y = jnp.concatenate([jnp.atleast_1d(p), jnp.atleast_1d(input)])
            self.mem(y)
            return y
        return composite(apply, members=dict(mem=delay_def(0.0)), name='grow')

    with pytest.raises(TypeError, match='conflicts with its declared spec'):
        grow().with_input(jnp.asarray(1.0)).parameterize().init()


def test_init_seeds_dereference_per_member():
    """A non-reserved init argument routes to its member by name, the way
    param already does — dereferenced per member, never broadcast to every
    member. The reserved channels (input, rng, ndef) keep their own routing."""
    def biased_def():
        # a cyclic node whose state seeds from a non-reserved init arg
        def init(bias=0.0):
            return jnp.asarray(bias)
        def apply(state, input):
            return state, input + state
        return node_def(apply, init=init, name='biased')

    # serial pipe path (the pipe init_fn)
    pipe = serial(a=biased_def(), b=biased_def()).parameterize()
    st = pipe.init(a=Struct(bias=1.0), b=Struct(bias=2.0))
    assert st.a == 1.0 and st.b == 2.0            # each member its own value

    # a member with no seed keeps its init default: no broadcast leak
    st2 = pipe.init(a=Struct(bias=5.0))
    assert st2.a == 5.0 and st2.b == 0.0

    # an unknown member name is loud, not a silent drop
    with pytest.raises(TypeError, match='unknown'):
        pipe.init(a=Struct(bias=1.0), nope=Struct(bias=9.0))

    # self-form composite path (_member_init) routes the same way
    def apply(self, input):
        return self.x(input)
    comp = composite(apply, members=dict(x=biased_def(), y=biased_def()),
                     name='c').parameterize()
    cs = comp.init(x=Struct(bias=3.0), y=Struct(bias=4.0))
    assert cs.x == 3.0 and cs.y == 4.0


def test_apply_unpacks_input_fields():
    """Stage 1c: apply's trailing params are input FIELDS — the input Struct
    is unpacked into them by name, and a trailing rng is promoted to a
    KeyStream. The signature declares the input structure, no tracing."""
    def mixer():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def init():
            return jnp.zeros(())
        def apply(param, state, signal, rng):       # signal, rng are input fields
            new = state + param.gain * signal + jax.random.normal(rng.next(), ())
            return new, new
        return node_def(apply, param=param, init=init, name='mixer')

    node = mixer().parameterize(gain=1.0)
    s0 = node.init()
    s1, out = node.apply(s0, Struct(signal=jnp.asarray(2.0), rng=jax.random.PRNGKey(0)))
    assert out == s1                                 # cyclic (state, output)
    assert not jnp.allclose(out, 2.0)                # noise from the input rng added



def test_wired_composite_field_signature():
    """A wired apply shares the leaf sugar: trailing field names unpack
    the input bundle, the signature declares the spec (REQUIRED or
    carrying defaults), and validation is the ordinary loud kind."""
    def double():
        return node_def(lambda input: 2.0 * input, name='double')

    def block():
        def apply(self, x, offset=0.0):
            return self.double(x) + offset
        return composite(apply, members=dict(double=double()), name='block')

    b = block()
    spec = b.ndef.apply_input_spec
    assert spec.x is REQUIRED and spec.offset == 0.0

    node = b if b.bound else b.parameterize()
    assert node.apply(x=3.0) == 6.0
    assert node.apply(x=3.0, offset=1.0) == 7.0
    # apply is the unvalidated fast path (specs are consulted at the
    # build entries): an extra field is ignored, exactly as at a leaf
    assert node.apply(x=3.0, y=1.0) == 6.0


def test_wired_composite_field_signature_rng():
    """A declared rng field arrives as the boundary key stream; author
    draws and member injections share one stream, never one key."""
    import jax

    def jitter():
        def param(sigma):
            return Struct(sigma=jnp.asarray(sigma))
        def apply(param, x, rng):
            return x + param.sigma * jax.random.normal(rng.next())
        return node_def(apply, param=param, name='jitter')

    def block():
        def apply(self, x, rng):
            own = jax.random.normal(rng.next())          # the author's draw
            return self.jitter(x=x) + 0.001 * own        # the member's draw, injected
        return composite(apply, members=dict(jitter=jitter()), name='noisy_block')

    node = block().parameterize(jitter=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    a = node.apply(x=1.0, rng=key)
    b = node.apply(x=1.0, rng=key)
    c = node.apply(x=1.0, rng=jax.random.PRNGKey(1))
    assert jnp.allclose(a, b)                    # same key, same draws
    assert not jnp.allclose(a, c)
    assert 'rng' in block().apply_input_spec     # sig-declared, spec-visible
