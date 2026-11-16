"""Multi-timescale register file updating two levels per step.

Each level `k` represents one timescale octave. Level zero updates every step,
and a staggered binary-carry schedule activates one higher level alongside it at
rate `1 / 2^k`, keeping per-step work at two register updates regardless of
hierarchy depth while doubling retention span at each level. Between scheduled
ticks, unscheduled registers remain bitwise unchanged so gradient credit
propagates across long horizons through slower levels rather than traversing
every intermediate transition.

At a scheduled tick, `ConvexUpdate` mixes the level's held content with a
drive-centered candidate via a gate conditioned only on the drive and the next
finer level, making transport of `held` convex by construction.

`SummaryRegisters` maintains a running linear readout across all levels in
cyclic state: each level owns a bias-free output projection, and each register
write adds the projected state difference to the carried readout. In the recall
benchmark below, the target class appears in the first token alone and is
predicted at the final step beyond the reach of fast levels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (Composite, Leaf, Node, PNode, Struct, Wrapper, batch,
                     control, ensemble, nn, node, scanned, serial, train_step,
                     trained)
from nodejax.transforms.transform import transform


LEVELS = 7
HIDDEN = 16
CLASSES = 8
STEPS = 16
BATCH, ROUNDS = 16, 3


def active_levels(tick: jax.Array, levels: int) -> jax.Array:
    """Computes the two register indices scheduled to write at `tick`.

    Level zero writes on every step alongside the lowest higher level whose
    phase-staggered binary counter bit flips; carry overflows above the top
    level wrap onto `levels - 1`.
    """
    above = jnp.arange(1, levels)
    period = 2 ** above
    flips = (tick + period // 2) % period == period - 1
    partner = jnp.where(jnp.any(flips), jnp.argmax(flips) + 1, levels - 1)
    return jnp.stack([jnp.zeros((), jnp.int32), partner.astype(jnp.int32)])


@node
def Counter() -> PNode:
    """Advances an integer step counter initialized to a uniformly random 32-bit phase."""
    def init(rng) -> jax.Array:
        bits = jax.random.bits(rng.next(), (), dtype=jnp.uint32)
        return jax.lax.bitcast_convert_type(bits, jnp.int32)

    def apply(state) -> tuple[jax.Array, jax.Array]:
        next_state = state + 1
        return next_state, next_state

    return Leaf(apply, init=init)


@transform(preserves='param,state')
def indexed_apply(copies: Node) -> Node:
    """Applies and updates a single dynamically selected member of an `ensemble` Node."""
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
    """Updates a register state via a convex combination of `held` and a gated candidate.

    Both the sigmoid gate and the zero-initialized candidate residual read
    `input` and `neighbor` without reading `held`, keeping transport with
    respect to `held` convex and initializing the residual correction to zero.
    """
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
    """Maintains a `levels`-octave register file and its incrementally projected readout.

    `block` maps `Struct(held, input, neighbor)` to a level's updated register
    vector, with independent parameters per level applied two at a time through
    `indexed_apply`. Each level owns a bias-free linear projection into `out`,
    and every scheduled write adds the projected state increment `value - held[level]`
    to `readout`, emitting `Struct(readout, held)` from input `(levels, hidden)`.
    """
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
    """Builds a sequence classifier that predicts the initial token class from the final register readout."""
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
    """Generates one-hot recall episodes carrying the target class at step zero followed by zeros."""
    labels = rng.randint(0, CLASSES, size=(rounds, BATCH))
    token = np.zeros((rounds, BATCH, STEPS, CLASSES), dtype=np.float32)
    token[np.arange(rounds)[:, None], np.arange(BATCH), 0, labels] = 1.0
    return jnp.asarray(token), jnp.asarray(labels)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean softmax cross-entropy loss against integer class labels."""
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, target))


def test_registers_hold_and_carry_an_exact_readout() -> None:
    """Verifies that at most two registers change per step and the carried readout matches full-file projection."""
    horizon = 2 ** (LEVELS - 1)
    drives = jax.random.normal(jax.random.PRNGKey(2), (horizon, LEVELS, HIDDEN))
    model = SummaryRegisters(
        ConvexUpdate(HIDDEN), LEVELS, HIDDEN, out=CLASSES).with_input(
        drives[0]).parameterize(rng=jax.random.PRNGKey(0))
    initial = model.initialize(input=drives[0], rng=jax.random.PRNGKey(1))
    weights = [value for value in jax.tree.leaves(model.param.projections)
               if value.shape == (LEVELS, HIDDEN, CLASSES)]
    assert len(weights) == 1

    _, outputs = initial.scan(drives)
    previous_held = jnp.concatenate([initial.state.held[None], outputs.held[:-1]], axis=0)
    projected = jnp.einsum('tlh,lho->to', outputs.held, weights[0])
    assert jnp.allclose(outputs.readout, projected, atol=1e-4)
    assert jnp.all(jnp.sum(jnp.any(outputs.held != previous_held, axis=-1), axis=-1) <= 2)


def test_recall_across_the_full_horizon() -> None:
    """Trains `recall_model` across `ROUNDS` episodes and verifies parameter updates and holdout logit shapes."""
    episodes, labels = recall_data(np.random.RandomState(0), ROUNDS)
    model = batch(recall_model(LEVELS, HIDDEN)).with_input(
        episodes[0]).parameterize(rng=jax.random.PRNGKey(0))
    trainer = train_step(model, xent, optax.adam(3e-3))

    final, aux = trained(trainer).apply(input=episodes, target=labels,
                                        rng=jax.random.PRNGKey(1))
    assert aux.loss.shape == (ROUNDS,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, final.param, model.param))

    holdout, _ = recall_data(np.random.RandomState(1), 1)
    logits = final.node.bind(final.param).apply(holdout[0],
                                                rng=jax.random.PRNGKey(2))
    assert logits.shape == (BATCH, CLASSES)
    assert jnp.all(jnp.isfinite(logits))
