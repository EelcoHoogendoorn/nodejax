"""A generic test-time-training wrapper in current Flax NNX.

The incumbent-middleware exhibit, sibling to ttt_rnn_by_hand.py: the
same task family, budget and scoring as ttt_nodejax's ttt rows,
but the question here is REUSABILITY: can TTT be written once, over
arbitrary recurrent modules, as nodejax's ttt transform is?

The answer is yes for modules that follow Flax's RNN cell protocol:
`initialize_carry(input_shape)` plus `(carry, input) -> (carry, output)`.
The TTT class below wraps Flax's stock `SimpleCell`, so no custom recurrent
cell is needed.

Current NNX graph-aware transforms keep the adapted module intact. TTT clones
the cell, carries that clone through `nnx.scan`, differentiates it with
`nnx.grad`, and updates its Params with `nnx.update`. Non-Param Variables stay
on the clone and advance through their ordinary module calls. No manual
`split` and `merge` boundary is needed.

The wrapper still declares policy that NodeJAX reads from its common contract:
the RNN cell protocol defines recurrent carry, Param Variables are adapted,
other Variables carry unchanged by the optimizer, and a separate Param tree
holds the learned step sizes. That is the substantive comparison.

flax is deliberately NOT a dependency of this repo; the file exists
as a reference exhibit and runs in any environment with flax
installed:  python -m examples.comparisons.ttt.ttt_rnn_flax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from nodejax.core.types import PyTree

VOCAB, HIDDEN = 8, 16
STREAM, SUPPORT = 192, 128
TASKS, META_STEPS = 8, 300
TTT_LR0, META_LR = 0.05, 1e-3
CONCENTRATION = 2.0
QUERY0 = SUPPORT


def make_tasks(rs, n_tasks: int):
    logits = CONCENTRATION * rs.standard_normal((n_tasks, VOCAB, VOCAB))
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)
    tokens = np.zeros((n_tasks, STREAM), dtype=np.int64)
    state = rs.randint(VOCAB, size=n_tasks)
    rows = np.arange(n_tasks)
    for t in range(STREAM):
        tokens[:, t] = state
        u = rs.random(n_tasks)[:, None]
        state = (P[rows, state].cumsum(-1) > u).argmax(-1)
    return jnp.asarray(tokens, dtype=jnp.int32)


def pair(tokens: jax.Array):
    """The next-token pairing, spelled in the loader: the input column is
    the sequence delayed one step (primed with its own first token), the
    target column the sequence itself. This is the next_step register
    hand-rolled as data, which is where a protocol wrapper without a
    state slot for it puts the pairing."""
    prev = jnp.concatenate([tokens[..., :1], tokens[..., :-1]], axis=-1)
    return dict(input=prev, target=tokens)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -jax.nn.log_softmax(logits)[target]


class Rate(nnx.Param):
    """Per-weight inner step sizes: a Param subclass so the outer
    optimizer trains them by default. They live outside the wrapped cell,
    so selecting that cell's Params excludes the rates."""


def _is_var(x: jax.Array):
    return isinstance(x, nnx.Variable)


def _strip(tree: PyTree):
    """Read Variable values for arithmetic over matching state trees."""
    return jax.tree.map(lambda v: v[...] if _is_var(v) else v, tree, is_leaf=_is_var)


class Forecast(nnx.Module):
    """An embedding, any protocol cell, and a logits head. It keeps the
    protocol from the outside, tokens in and logits out."""

    def __init__(self, cell, hidden: int, rngs: nnx.Rngs):
        self.embed = nnx.Embed(VOCAB, hidden, rngs=rngs)
        self.cell = cell
        self.head = nnx.Linear(hidden, VOCAB, rngs=rngs)

    def initialize_carry(self, input_shape, rngs=None):
        # the protocol asks for the CELL's input shape, which is the
        # embedding's output, not the token's; the wrapper answers for it
        return self.cell.initialize_carry((HIDDEN,), rngs)

    def __call__(self, carry, token):
        carry, y = self.cell(carry, self.embed(token))
        return carry, self.head(y)


class TTT(nnx.Module):
    """Test-time training over any RNNCellBase-protocol module: the
    wrapped cell's weights become fast state, updated by one gradient
    step on loss_fn per sample, predict-then-update; initialization
    and per-weight rates are this module's trainable params.

    The adapted clone and protocol carry advance together. Params receive the
    learned gradient update. Any other Variables advance only through the
    wrapped module's own calls."""

    def __init__(self, cell, loss_fn, lr0: float):
        self.cell = cell
        self.loss_fn = loss_fn
        params = nnx.state(cell, nnx.Param)
        self.lr = nnx.data(jax.tree.map(lambda x: Rate(jnp.full_like(x, lr0)),
                                        _strip(params)))

    def __call__(self, sequence):
        fast = nnx.clone(self.cell)
        lr = _strip(self.lr)
        carry = fast.initialize_carry(sequence['input'].shape[1:], nnx.Rngs(0))
        fast_axes = nnx.StateAxes({...: nnx.Carry})

        @nnx.scan(in_axes=(fast_axes, nnx.Carry, 0),
                  out_axes=(nnx.Carry, 0))
        def step(fast, carry, sample):
            def selfsup(adapted):
                next_carry, output = adapted(carry, sample['input'])
                return self.loss_fn(output, sample['target']), (next_carry, output)

            grads, (next_carry, output) = nnx.grad(selfsup, has_aux=True)(fast)
            values = _strip(nnx.state(fast, nnx.Param))
            slopes = _strip(grads)
            next_values = jax.tree.map(
                lambda value, slope, rate: value - rate * slope,
                values, slopes, lr)
            nnx.update(fast, next_values)
            return next_carry, output

        _, preds = step(fast, carry, sequence)
        return preds


def query_xent(preds, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(preds)
    picked = jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return jnp.mean(-picked[..., QUERY0:])


class Batched(nnx.Module):
    """nodejax's batch transform, module-shaped: inputs mapped over the
    leading axis, Params broadcast. This is straightforward, but the
    asymmetry is what the wrapper must LEGISLATE rather than read:
    an axis policy per variable collection (here Params broadcast and
    nothing else considered), where the contract-reading transform
    takes the roles positionally. A wrapped module that mutates
    broadcast state under the map is this design's characteristic
    failure class; nothing here checks for it."""

    def __init__(self, module):
        self.module = module

    def __call__(self, x):
        return nnx.vmap(lambda m, s: m(s), in_axes=(None, 0), out_axes=0)(self.module, x)


class TrainStep(nnx.Module):
    """nodejax's train_step transform, mirrored: __call__ is ONE
    optimizer update over a sequence element dict(input=..., target=...)
    scored by loss_fn(output, target), and scanned() is the packaged
    sequence view over the step. This is the same step/scan split as
    nodejax's PNode. The trainer's state (model and optimizer objects,
    mutated in place) rides the scan as its carry.

    With Batched composing the task axis, the whole arrangement now
    mirrors train_step(batch(model), ...) piece for piece. The
    external scan needs an axis policy where NodeJAX's scan transform
    reads fixed contract roles. Nested adaptation remains inside NNX: TTT
    clones the inner object, carries it through `nnx.scan`, differentiates it
    with `nnx.grad`, and lets the outer gradient pass through those updates."""

    def __init__(self, model, loss_fn, tx):
        self.model = model
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        self.loss_fn = loss_fn

    def __call__(self, batch):
        def objective(model):
            return self.loss_fn(model(batch['input']), batch['target'])

        loss, grads = nnx.value_and_grad(objective)(self.model)
        self.optimizer.update(self.model, grads)
        return loss

    def scan(self, sequence):
        """The whole run as one compiled scan of the step. The Carry
        ceremony packaged once. nodejax's counterpart takes and returns
        the trainer state as a value; this advances the objects."""
        @nnx.jit
        def fit(trainer, sequence):
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def loop(trainer, batch):
                return trainer, trainer(batch)

            _, losses = loop(trainer, sequence)
            return losses

        return fit(self, sequence)


def run(name: str, model):
    batched = Batched(model)
    trainer = TrainStep(batched, query_xent, optax.adam(META_LR))

    train = pair(make_tasks(np.random.RandomState(0), META_STEPS * TASKS))
    fold = lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:])
    sequence = dict(input=jax.tree.map(fold, train), target=fold(train['target']))
    losses = trainer.scan(sequence)

    tasks = pair(make_tasks(np.random.RandomState(99), TASKS))
    preds = batched(tasks)
    n = sum(x.size for x in jax.tree.leaves(nnx.state(model)))
    print(f'{name:16s} weights={n:5d} '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'query xent {query_xent(preds, tasks["target"]):.2f}', flush=True)


def main() -> None:
    rngs = nnx.Rngs(0)
    run('ttt(SimpleCell)', TTT(Forecast(nnx.SimpleCell(HIDDEN, HIDDEN, rngs=rngs),
                                        HIDDEN, rngs), xent, TTT_LR0))


if __name__ == '__main__':
    main()
