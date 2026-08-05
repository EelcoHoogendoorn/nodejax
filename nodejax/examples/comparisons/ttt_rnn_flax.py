"""A generic test-time-training wrapper in idiomatic Flax NNX.

The incumbent-middleware exhibit, sibling to ttt_rnn_by_hand.py: the
same task family, budget and scoring as meta_comparison's ttt rows,
but the question here is REUSABILITY — can TTT be written once, over
arbitrary recurrent modules, as nodejax's ttt transform is?

The answer is yes, over exactly the subset of flax that follows the
RNNCellBase protocol — initialize_carry(input_shape) plus
(carry, x) -> (carry, y) — which is the node contract minus the param
slot: flax's cells carry their state functionally because lax.scan
demanded it. The TTT class below is that generic wrapper, and main()
instantiates it over two of flax's own stock cells (SimpleCell and
GRUCell) with zero custom cell code. Outside the protocol — modules
that keep their recurrence in mutable Variables — genericity would
additionally need a per-call-site taxonomy of which variable
collections are fast state, carried state and frozen, because flax
has no universal role contract to read it from.

The wrapper's body is the boundary toll made visible: the cell's
Params play two roles (meta-params to the outer optimizer, initial
fast state to the inner scan), so every step splits the object world
from the value world and back — split, strip the Variable wrappers to
ride the carry, re-wrap and merge to apply, differentiate the merged
call. The per-weight rates are a Param subclass so the outer
optimizer trains them while the cell-only split misses them.

STATE CENSUS. The file exercises five distinct state mechanisms:
(1) mutable Variables on objects — Params under optimizer.update,
and the optimizer's own moments — advanced by in-place mutation;
(2) the collection taxonomy's special cases — BatchStat running
stats: the same mutation under a distinct type so transforms can
legislate per-collection policy, harvested here by re-splitting
inside the gradient; (3) the protocol carry — functional values by
convention, because lax.scan demanded it; (4) rng state — RngKey and
RngCount variables with their own types, their own lifting decorator
(split_rngs), their own flags and their own trace errors; (5) the
value-world shadow state this wrapper manufactures — stripped raw
pytrees riding lax.scan between split and merge, plus the
graphdef/template pairs in flight — state that exists only to ferry
between mechanisms 1 through 4. Even the scan carry appears three
ways in this one file: protocol values in the cell, object state
under nnx.Carry in the trainer, raw arrays under lax.scan in the ttt
interior. The node contract's census of the same model: one state
slot; running stats, rng, hidden state, fast weights and optimizer
moments are fields of it.

flax is deliberately NOT a dependency of this repo; the file exists
as a reference exhibit and runs in any environment with flax
installed:  python -m nodejax.examples.comparisons.ttt_rnn_flax
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

LAGS = 8
STREAM, SUPPORT = 192, 128
HIDDEN = 16
TASKS, META_STEPS = 8, 400
TTT_LR0, META_LR = 0.01, 1e-3
NOISE = 0.05
QUERY0 = SUPPORT - LAGS


def make_tasks(rs, n_tasks):
    t = np.arange(STREAM)[None, :]
    def draw(lo, hi):
        return rs.uniform(lo, hi, (n_tasks, 1))
    x = (draw(0.5, 1.5) * np.sin(2 * np.pi * draw(0.02, 0.1) * t + draw(0, 2 * np.pi))
         + draw(0.2, 0.8) * np.sin(2 * np.pi * draw(0.1, 0.25) * t + draw(0, 2 * np.pi))
         + NOISE * rs.standard_normal((n_tasks, STREAM))).astype(np.float32)
    M = STREAM - LAGS
    lags = np.stack([x[:, i:i + LAGS] for i in range(M)], axis=1)
    return dict(input=jnp.asarray(lags), target=jnp.asarray(x[:, LAGS:]))


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


class Rate(nnx.Param):
    """Per-weight inner step sizes: a Param subclass so the outer
    optimizer trains them by default; they never enter the fast-weight
    harvest because that split reads the cell submodule alone."""


def _is_var(x):
    return isinstance(x, nnx.Variable)


def _strip(tree):
    """Variable-wrapped state -> raw arrays (to ride a scan carry)."""
    return jax.tree.map(lambda v: v[...] if _is_var(v) else v, tree, is_leaf=_is_var)


def _wrap(tmpl, pure):
    """raw arrays -> Variable-wrapped state (to merge), rebuilt over a
    split template — _strip's inverse."""
    return jax.tree.map(lambda t, v: type(t)(v), tmpl, pure, is_leaf=_is_var)


class Stateless(nnx.Module):
    """Lift a feedforward module into the carry protocol. Flax's
    stateless modules have a different call shape (no carry), so
    covering them takes this adapter — the trivial state slot the
    module taxonomy lacks, supplied by hand. In the node contract the
    slot exists universally, which is why meta_comparison's
    ttt-linear and ttt-mlp rows use the same stock transform as
    ttt-rnn, unchanged."""

    def __init__(self, module):
        self.module = module

    def initialize_carry(self, input_shape, rngs=None):
        return ()

    def __call__(self, carry, x):
        return carry, self.module(x)


class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.up = nnx.Linear(LAGS, HIDDEN, rngs=rngs)
        self.down = nnx.Linear(HIDDEN, 1, rngs=rngs)

    def __call__(self, x):
        return self.down(jax.nn.gelu(self.up(x)))[..., 0]


class Stacked(nnx.Module):
    """A vectorized stack of shape-preserving cells — nodejax's
    stack(cell, n) hand-assembled: params vectorized at construction
    (nnx.vmap over split rngs of a caller-supplied cell factory),
    applied by nnx.scan over the layer axis, with the SECOND layer
    axis — each layer's hidden state — coordinated explicitly as the
    scan's per-layer input and output. The activation threads down
    the stack as the scan carry. The carry is the cell's own,
    per layer: initialize_carry is vmapped over the layer axis the
    same way construction was, so its structure (LSTM's (c, h) tuple,
    an array, anything) and any per-layer parameter dependence come
    from each layer's own slice."""

    def __init__(self, make_cell, n: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=n)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def make(rngs):
            return make_cell(rngs)

        self.cells = make(rngs)
        self.n = n

    def initialize_carry(self, input_shape, rngs=None):
        if rngs is None:
            @nnx.vmap(in_axes=(nnx.StateAxes({nnx.Param: 0}),), out_axes=0)
            def per_layer(cells):
                return cells.initialize_carry(input_shape, None)

            return per_layer(self.cells)

        @nnx.split_rngs(splits=self.n)
        @nnx.vmap(in_axes=(nnx.StateAxes({nnx.Param: 0}), 0), out_axes=0)
        def per_layer(cells, rngs):
            return cells.initialize_carry(input_shape, rngs)

        return per_layer(self.cells, rngs)

    def __call__(self, carry, x):
        @nnx.scan(in_axes=(nnx.Carry, nnx.StateAxes({nnx.Param: 0}), 0),
                  out_axes=(nnx.Carry, 0))
        def layers(act, cells, h):
            h2, y = cells(h, act)
            return y, h2

        y, carry2 = layers(x, self.cells, carry)
        return carry2, y


class DeepForecast(nnx.Module):
    """proj >> Stacked >> head, still the protocol — the flax
    spelling of ttt's stacked inner from meta_comparison."""

    def __init__(self, make_cell, n: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(LAGS, HIDDEN, rngs=rngs)
        self.stack = Stacked(make_cell, n, rngs)
        self.head = nnx.Linear(HIDDEN, 1, rngs=rngs)

    def initialize_carry(self, input_shape, rngs=None):
        return self.stack.initialize_carry(input_shape, rngs)

    def __call__(self, carry, x):
        carry, y = self.stack(carry, self.proj(x))
        return carry, self.head(y)[..., 0]


class RunStats(nnx.Module):
    """Running standardizer, NNX-idiomatic: per-feature EMA mean and
    variance as BatchStat variables, mutated in place by the call.
    Always-update streaming semantics — the point where flax would
    normally reach for use_running_average, sidestepped here because
    train and deployment share the streaming behavior."""

    def __init__(self, momentum: float, features: int):
        self.momentum = momentum
        self.mean = nnx.BatchStat(jnp.zeros(features))
        self.var = nnx.BatchStat(jnp.ones(features))

    def __call__(self, x):
        m, v = self.mean[...], self.var[...]
        out = (x - m) / jnp.sqrt(v + 1e-5)
        self.mean[...] = (1 - self.momentum) * m + self.momentum * x
        self.var[...] = (1 - self.momentum) * v + self.momentum * (x - m) ** 2
        return out


class StatsCell(nnx.Module):
    """RunStats in front of a protocol cell — still the protocol."""

    def __init__(self, momentum: float, cell, features: int):
        self.stats = RunStats(momentum, features)
        self.cell = cell

    def initialize_carry(self, input_shape, rngs=None):
        return self.cell.initialize_carry(input_shape, rngs)

    def __call__(self, carry, x):
        return self.cell(carry, self.stats(x))


class Forecast(nnx.Module):
    """Any protocol cell plus a scalar head — still the protocol."""

    def __init__(self, cell, hidden: int, rngs: nnx.Rngs):
        self.cell = cell
        self.head = nnx.Linear(hidden, 1, rngs=rngs)

    def initialize_carry(self, input_shape, rngs=None):
        return self.cell.initialize_carry(input_shape, rngs)

    def __call__(self, carry, x):
        carry, y = self.cell(carry, x)
        return carry, self.head(y)[..., 0]


class TTT(nnx.Module):
    """Test-time training over any RNNCellBase-protocol module: the
    wrapped cell's weights become fast state, updated by one gradient
    step on loss_fn per sample, predict-then-update; initialization
    and per-weight rates are this module's trainable params.

    The scan carry has THREE strands: the adapted weights, the
    protocol carry, and `rest` — every non-Param variable of the cell
    (running statistics above all). The taxonomy is decided here, in
    the split: Params are gradient-adapted, rest is carried and
    updated only by the module's own in-place mutations — which must
    be harvested by re-splitting the merged module after the call,
    inside the gradient function, and routed out through the aux."""

    def __init__(self, cell, loss_fn, lr0: float):
        self.cell = cell
        self.loss_fn = loss_fn
        _, params, _ = nnx.split(cell, nnx.Param, ...)
        self.lr = nnx.data(jax.tree.map(lambda x: Rate(jnp.full_like(x, lr0)),
                                        _strip(params)))

    def __call__(self, stream):
        graphdef, p_tmpl, r_tmpl = nnx.split(self.cell, nnx.Param, ...)
        lr = _strip(self.lr)
        carry0 = self.cell.initialize_carry(stream['input'].shape[1:], nnx.Rngs(0))

        def step(carry, sample):
            fast, rest, h = carry

            def selfsup(fast):
                cell = nnx.merge(graphdef, _wrap(p_tmpl, fast), _wrap(r_tmpl, rest))
                h2, out = cell(h, sample['input'])
                _, _, r2 = nnx.split(cell, nnx.Param, ...)   # harvest mutations
                return self.loss_fn(out, sample['target']), (h2, out, _strip(r2))

            grads, (h2, out, rest2) = jax.grad(selfsup, has_aux=True)(fast)
            fast2 = jax.tree.map(lambda w, g, r: w - r * g, fast, grads, lr)
            return (fast2, rest2, h2), out

        (_, _, _), preds = jax.lax.scan(
            step, (_strip(p_tmpl), _strip(r_tmpl), carry0), stream)
        return preds


def query_mse(preds, target):
    return jnp.mean((preds[..., QUERY0:] - target[..., QUERY0:]) ** 2)


class Batched(nnx.Module):
    """nodejax's batch transform, module-shaped: inputs mapped over the
    leading axis, Params broadcast. Perfectly writable — the
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
    optimizer update over a stream element dict(input=..., target=...)
    scored by loss_fn(output, target), and scan() is the packaged
    sequence view over the step — the same step/scan split as
    nodejax's Node. The trainer's state (model and optimizer objects,
    mutated in place) rides the scan as its carry.

    With Batched composing the task axis, the whole arrangement now
    mirrors train_step(batch(model), ...) piece for piece. The
    residual asymmetries, priced openly: the external scan costs a
    Carry-annotation ceremony where nodejax writes trainer.scan(init,
    stream); and NESTING a trainer (finetune/ttt's mechanism — the
    inner loop differentiated by the outer) is done here by
    reconstructing the inner model functionally from traced state
    (the TTT class's split/merge interior) rather than by scanning
    the trainer you hold — the object in your hand cannot be the
    thing adapted, only a traced reconstruction of it can, because
    gradients flow through values, not through mutation."""

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

    def scan(self, stream):
        """The whole run as one compiled scan of the step — the Carry
        ceremony packaged once. nodejax's counterpart takes and returns
        the trainer state as a value; this advances the objects."""
        @nnx.jit
        def fit(trainer, stream):
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def loop(trainer, batch):
                return trainer, trainer(batch)

            _, losses = loop(trainer, stream)
            return losses

        return fit(self, stream)


def run(name, model):
    batched = Batched(model)
    trainer = TrainStep(batched, query_mse, optax.adam(META_LR))

    train = make_tasks(np.random.RandomState(0), META_STEPS * TASKS)
    fold = lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:])
    stream = dict(input=jax.tree.map(fold, train), target=fold(train['target']))
    losses = trainer.scan(stream)

    tasks = make_tasks(np.random.RandomState(99), TASKS)
    preds = batched(tasks)
    n = sum(x.size for x in jax.tree.leaves(nnx.state(model)))
    print(f'{name:16s} weights={n:5d} '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'query mse {query_mse(preds, tasks["target"]):.4f}', flush=True)


def main():
    rngs = nnx.Rngs(0)
    run('ttt(SimpleCell)', TTT(Forecast(nnx.SimpleCell(LAGS, HIDDEN, rngs=rngs),
                                        HIDDEN, rngs), mse, TTT_LR0))
    run('ttt(GRUCell)', TTT(Forecast(nnx.GRUCell(LAGS, HIDDEN, rngs=rngs),
                                     HIDDEN, rngs), mse, TTT_LR0))
    # gelu inner: nothing saturates, so the cooler seed rate (the
    # meta_comparison ttt-mlp finding, reproduced here)
    run('ttt(MLP)', TTT(Stateless(MLP(rngs)), mse, 0.003))
    run('ttt(2xSimple)', TTT(DeepForecast(
        lambda r: nnx.SimpleCell(HIDDEN, HIDDEN, rngs=r), 2, rngs), mse, TTT_LR0))
    run('ttt(2xLSTM)', TTT(DeepForecast(
        lambda r: nnx.LSTMCell(HIDDEN, HIDDEN, rngs=r), 2, rngs), mse, TTT_LR0))
    run('ttt(stats+Simple)', TTT(Forecast(
        StatsCell(0.05, nnx.SimpleCell(LAGS, HIDDEN, rngs=rngs), LAGS),
        HIDDEN, rngs), mse, TTT_LR0))


if __name__ == '__main__':
    main()
