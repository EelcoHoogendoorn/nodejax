"""A register file over halving timescales: two writes per step.

One register per timescale octave. Level zero writes every step; a
staggered binary-carry schedule wakes exactly one higher level beside
it, level k at rate 1/2^k, so per-step compute is two updates at any
number of levels while the held span doubles per level. Between its
ticks a register holds bitwise: retention is the schedule's, not a
gate's, and credit between distant steps climbs through the slow
levels instead of crossing every transition.

The update at a tick is a convex law written on the register: a gate
conditioned on the drive and the next finer register mixes held
content with a candidate of the same reads, so the transport of held
is convex by construction and the candidate is centered on the drive
(identity plus a zero-initialized correction). Neither the gate nor
the candidate reads held.

The output is a carried readout of the whole file: each level owns a
bias-free output projection, and a write updates register and readout
together by projecting the register's change. Two rows of work per
step, every timescale in the output, and a slow register receives
gradient once per tick as the accumulated signal of the steps its
contribution sat frozen. Reading the whole file matters: with only
the fast register exposed, recalling old content waits on trained
descent hops, and the measured recall frontier sits octaves lower.

The training test is recall: the class shows only in the first token,
the readout answers at the last step, and the horizon exceeds every
fast level's reach, so learning it at all means the gradient climbed
the hierarchy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (Composite, Leaf, Node, Struct, Wrapper, batch, control,
                     ensemble, node, scanned, serial, train_step, trained, nn)
from nodejax.transforms.transform import transform


LEVELS = 7
HIDDEN = 16
CLASSES = 8
STEPS = 64
BATCH, ROUNDS = 64, 600


def active_levels(tick: jax.Array, levels: int) -> jax.Array:
    """The two levels writing at ``tick``: level zero, and the higher
    level whose phase-staggered counter bit flips; the flip belonging
    above the truncated hierarchy goes to the top level, which then
    ticks twice per period, in phase with the level below it."""
    above = jnp.arange(1, levels)
    period = 2 ** above
    flips = (tick + period // 2) % period == period - 1
    partner = jnp.where(jnp.any(flips), jnp.argmax(flips) + 1, levels - 1)
    return jnp.stack([jnp.zeros((), jnp.int32), partner.astype(jnp.int32)])


@node
def Counter() -> Node:
    """Counter whose initial value is a uniformly random phase, so
    learned behavior cannot exploit alignment to the sequence start."""
    def init(rng) -> jax.Array:
        bits = jax.random.bits(rng.next(), (), dtype=jnp.uint32)
        return jax.lax.bitcast_convert_type(bits, jnp.int32)

    def apply(state) -> tuple[jax.Array, jax.Array]:
        next_state = state + 1
        return next_state, next_state

    return Leaf(apply, init=init)


@transform(preserves='param,state')
def indexed_apply(copies: Node) -> Node:
    """Apply and update one dynamically selected ensemble member."""
    def apply_fn(contract, param, state, input, rng):
        member = contract.members.copies.members.member
        level = input.level
        member_param = (jax.tree.map(lambda value: value[level], param)
                        if member.parametric else ())
        member_state = jax.tree.map(lambda value: value[level], state)
        next_member_state, output = member.apply(
            member_param, member_state, member.feed(input.signal), rng)
        next_state = jax.tree.map(
            lambda values, value: values.at[level].set(value),
            state, next_member_state)
        return next_state, output

    return Wrapper(copies=copies).roles(
        name=f'indexed({copies.name})',
        apply=apply_fn,
        apply_fields=('level', 'signal'),
        destructurable=False,
        destructurable_state=False,
    )


@node
def ConvexUpdate(hidden: int) -> Node:
    """The convex register law: (held, input, neighbor) to the new
    held value. A gate conditioned on the input and neighbor reads
    mixes held content with a candidate of the same reads, so
    retention is one minus the write and the transport of held is
    convex by construction: neither the gate nor the candidate ever
    reads held. The candidate is centered on the drive with a
    zero-initialized correction, so a chain of registers begins as a
    pure identity and the corrections grow as they earn gradient."""
    members = Composite(
        update=nn.Linear(hidden),
        candidate=nn.Linear(hidden, weight_init=jax.nn.initializers.zeros),
    )

    def apply(self, input):
        reads = jnp.concatenate([input.input, input.neighbor])
        gate = jax.nn.sigmoid(self.update(reads))
        candidate = input.input + self.candidate(reads)
        return (1.0 - gate) * input.held + gate * candidate

    return members(apply, name='convex_update')


@transform
def SummaryRegisters(block: Node, levels: int, hidden: int,
                     out: int) -> Node:
    """Upward registers carrying a readout of the file beside the file.

    ``block`` maps the named triple (held, input, neighbor) to a
    level's new held value; each level draws its own params from the
    one definition, and the scheduled pair applies theirs through
    indexed_apply. Each level reads only the next finer level, so the
    state conditioning is acyclic and, with a convex block, the system
    Jacobian is block triangular with eigenvalues in [0, 1]. The input
    is a full file, (levels, hidden), read per level; a caller wanting
    the fast lane only zeroes every row above the bottom. The output
    is both: Struct(readout, held).

    The readout is the full-file projection maintained as state: each
    level owns a bias-free output projection (an offset is meaningless
    on a difference), and a write updates register and readout
    together, projecting the register's change. The readout equals the
    projection of the whole file at every step; both start at zero
    together."""
    bank = ensemble(block.with_input(Struct(
        held=jnp.zeros(hidden), input=jnp.zeros(hidden),
        neighbor=jnp.zeros(hidden))), n=levels)
    members = Composite(
        blocks=indexed_apply(bank),
        projections=indexed_apply(
            ensemble(nn.Linear(out, bias=False).with_input(jnp.zeros(hidden)),
                     n=levels)),
        held=control.Delay().with_input(jnp.zeros((levels, hidden))),
        readout=control.Delay().with_input(jnp.zeros(out)),
        counter=Counter(),
    )

    def apply(self, input):
        active = active_levels(self.counter.state, levels)
        held = self.held.state
        readout = self.readout.state
        for level in active:
            neighbor = jnp.where(level == 0,
                                 jnp.zeros(hidden), held[level - 1])
            value = self.blocks(level=level, signal=Struct(
                held=held[level], input=input[level], neighbor=neighbor))
            readout = readout + self.projections(
                level=level, signal=value - held[level])
            held = held.at[level].set(value)
        self.held(held)
        self.readout(readout)
        self.counter()
        return Struct(readout=readout, held=held)

    return members(apply, name='summary_registers')


def recall_model(levels: int, hidden: int) -> Node:
    """(steps, CLASSES) tokens -> (CLASSES,) logits at the last step."""
    return serial(
        sequence=scanned(serial(
            embed=nn.Linear(hidden),
            bottom=Leaf(lambda input: jnp.zeros((levels, hidden)).at[0].set(input),
                        name='bottom'),
            registers=SummaryRegisters(ConvexUpdate(hidden), levels, hidden,
                                       out=hidden),
            readout=Leaf(lambda input: input.readout, name='readout'),
        )),
        last=Leaf(lambda input: input[-1], name='last'),
        head=nn.Linear(CLASSES),
    )


def recall_data(rng: np.random.RandomState,
                rounds: int) -> tuple[jax.Array, jax.Array]:
    """The class shows only in the first token; zeros ever after."""
    labels = rng.randint(0, CLASSES, size=(rounds, BATCH))
    token = np.zeros((rounds, BATCH, STEPS, CLASSES), dtype=np.float32)
    token[np.arange(rounds)[:, None], np.arange(BATCH), 0, labels] = 1.0
    return jnp.asarray(token), jnp.asarray(labels)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, target))


def test_registers_hold_and_carry_an_exact_readout() -> None:
    """Unscheduled registers hold bitwise; the carried readout equals
    the projection of the whole file at every step of a full period."""
    drive = jnp.ones((LEVELS, HIDDEN))
    model = SummaryRegisters(
        ConvexUpdate(HIDDEN), LEVELS, HIDDEN, out=CLASSES).with_input(
        drive).parameterize(rng=jax.random.PRNGKey(0))
    state = model.initialize(input=drive, rng=jax.random.PRNGKey(1)).state
    weights = [value for value in jax.tree.leaves(model.param.projections)
               if value.shape == (LEVELS, HIDDEN, CLASSES)]
    assert len(weights) == 1
    for step in range(2 ** (LEVELS - 1)):
        drive = jax.random.normal(jax.random.PRNGKey(step), (LEVELS, HIDDEN))
        before = state.held
        state, output = model.apply(state, drive)
        assert jnp.allclose(
            output.readout, jnp.einsum('lh,lho->o', state.held, weights[0]),
            atol=1e-4)
        assert jnp.array_equal(output.readout, state.readout)
        assert jnp.array_equal(output.held, state.held)
        assert jnp.sum(jnp.any(state.held != before, axis=-1)) <= 2


def test_recall_across_the_full_horizon() -> None:
    """The class token at step zero outlives every fast level: recalling
    it at the last step means the gradient climbed the hierarchy."""
    episodes, labels = recall_data(np.random.RandomState(0), ROUNDS)
    model = batch(recall_model(LEVELS, HIDDEN)).with_input(
        episodes[0]).parameterize(rng=jax.random.PRNGKey(0))
    trainer = train_step(model, xent, optax.adam(3e-3))

    final, aux = trained(trainer).apply(input=episodes, target=labels,
                                        rng=jax.random.PRNGKey(1))
    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.2 * aux.loss[0]

    holdout, answers = recall_data(np.random.RandomState(1), 1)
    logits = final.node.bind(final.param).apply(holdout[0],
                                                rng=jax.random.PRNGKey(2))
    recall_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == answers[0])
    assert recall_accuracy > 0.9, recall_accuracy

    print(f'\n[register hierarchy] levels {LEVELS} | writes/step 2 | '
          f'loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} | '
          f'recall {recall_accuracy:.3f}')
