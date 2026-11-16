"""Gradient-free differential evolution packaged in the `train_step` trainer contract.

Plain JAX helpers `propose` and `select` generate mutant-and-crossover candidate weight PyTrees
and retain the lower-scoring survivor per population slot. `differential_evolution` wraps those
operations as a parameter-free cyclic optimizer `PNode` carrying `Struct(params, members)` in
state, with `params` holding the lowest-scoring member at the same path `train_step` uses for live
model weights.

`evolve` pairs that optimizer with a state-bound `PSNode` model inside a two-member `Composite`
exposing a `trained` method and emitting step loss via `Aux`. Unrolling with `.scan` advances
generations using call-time `rng`, and `trained(trainer)` extracts the champion `PSNode` without
computing gradients.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp

from nodejax import (
    node, Leaf, Node, Struct, Aux, PNode, PSNode, Composite, trained, tile,
)
from nodejax.core.types import PyTree
from examples.util import mse


def propose(members: PyTree, key: jax.Array, rate: float,
            crossover: float, bounds: tuple[PyTree, PyTree] | None = None) -> PyTree:
    """Generates one `DE/rand/1/bin` mutant-and-crossover candidate weight tree per member of `members`."""
    count = jax.tree.leaves(members)[0].shape[0]
    partners_key, mask_key, forced_key = jax.random.split(key, 3)
    picks = jax.random.randint(partners_key, (3, count), 0, count)
    anchor, first, second = (jax.tree.map(lambda leaf: leaf[rows], members)
                             for rows in picks)
    mutants = jax.tree.map(lambda base, high, low: base + rate * (high - low),
                           anchor, first, second)

    mutant_leaves, layout = jax.tree.flatten(mutants)
    parent_leaves = jax.tree.leaves(members)
    leaf_keys = jax.random.split(mask_key, len(mutant_leaves))
    total_dims = max(1, sum(math.prod(leaf.shape[1:]) for leaf in mutant_leaves))
    j_rand = jax.random.randint(forced_key, (count, 1), 0, total_dims)

    crossed_leaves = []
    offset = 0
    for mutant, parent, leaf_key in zip(mutant_leaves, parent_leaves, leaf_keys):
        dim = math.prod(mutant.shape[1:])
        forced_mask = (j_rand == (offset + jnp.arange(dim))).reshape((count,) + mutant.shape[1:])
        take = jax.random.bernoulli(leaf_key, crossover, mutant.shape) | forced_mask
        crossed_leaves.append(jnp.where(take, mutant, parent))
        offset += dim

    children = layout.unflatten(crossed_leaves)
    if bounds is not None:
        low, high = bounds
        children = jax.tree.map(lambda leaf: jnp.clip(leaf, low, high), children)
    return children


def select(members: PyTree, children: PyTree, member_scores: jax.Array,
           child_scores: jax.Array) -> tuple[PyTree, jax.Array]:
    """Selects the lower-scoring tree between `members` and `children` at each population slot."""
    child_wins = child_scores < member_scores

    def per_member(flags: jax.Array, leaf: jax.Array) -> jax.Array:
        return flags.reshape((-1,) + (1,) * (leaf.ndim - 1))

    survivors = jax.tree.map(
        lambda child, parent: jnp.where(per_member(child_wins, child), child, parent),
        children, members)
    return survivors, jnp.minimum(child_scores, member_scores)


def pointwise_mse(candidate: PNode, element: Struct) -> jax.Array:
    """Evaluates mean squared error of `candidate` on `element.input` against `element.target`."""
    return mse(candidate.apply(element.input), element.target)


@node
def Affine() -> Node:
    """Computes a scalar affine map `w * input + b`."""
    def param(w: float = 0.0, b: float = 0.0) -> Struct:
        return Struct(w=jnp.asarray(w), b=jnp.asarray(b))

    def apply(param, input) -> jax.Array:
        return param.w * input + param.b

    return Leaf(apply, param=param)


@node
def differential_evolution(population: int, rate: float = 0.7,
                           crossover: float = 0.9, spread: float = 1.0,
                           bounds: tuple[PyTree, PyTree] | None = None) -> PNode:
    """Maintains a differential-evolution population in `Struct(params, members)` state.

    `init` perturbs the starting parameter tree into `population` candidates, `propose` draws trial
    children from the current population using call-time `rng`, and `apply` runs pairwise selection
    and updates `params` to the lowest-scoring survivor.
    """
    def init(rng, input) -> Struct:
        leaves, layout = jax.tree.flatten(input)
        member_keys = jax.random.split(rng.next(), population)

        def draw_member(member_key: jax.Array) -> PyTree:
            # Independent per-leaf keys prevent initial perturbations from correlating across leaves.
            leaf_keys = jax.random.split(member_key, len(leaves))
            return layout.unflatten([
                leaf + spread * jax.random.normal(leaf_key, jnp.shape(leaf))
                for leaf, leaf_key in zip(leaves, leaf_keys)])

        members = jax.vmap(draw_member)(member_keys)
        return Struct(params=input, members=members)

    def propose_children(state, rng) -> PyTree:
        return propose(state.members, rng.next(), rate, crossover, bounds)

    def apply(state, children: PyTree, member_scores: jax.Array,
              child_scores: jax.Array) -> tuple[Struct, jax.Array]:
        survivors, scores = select(state.members, children,
                                   member_scores, child_scores)
        best = jnp.argmin(scores)
        champion = jax.tree.map(lambda leaf: leaf[best], survivors)
        return Struct(params=champion, members=survivors), scores[best]

    return Leaf(apply, init=init, methods=dict(propose=propose_children))


@node
def evolve(model: PSNode, fitness: Callable, optimizer: Node | PNode,
           rng: jax.Array) -> PSNode:
    """Builds a state-bound evolutionary trainer `Composite` pairing `optimizer` with `model`.

    Each `apply` step proposes candidate weight trees via `self.opt.propose(rng=rng.next())`,
    evaluates `fitness(model.node.bind(weights), Struct(input, target))` across `vmap`, updates the
    population, and emits the best score via `Aux(loss=best)`.
    """
    if not model.state_bound:
        raise TypeError('evolve takes the model fully bound, exactly as '
                        f'train_step does; got {model!r}')
    pool = optimizer.parameterize().initialize(rng=rng, input=model.param)

    # Wrapping `model` in a `Leaf` keeps a `state.model` slot even when `model` is stateless.
    def model_param() -> PyTree:
        return model.param

    def model_init() -> PyTree:
        return model.state

    def model_apply(param, state, input) -> tuple[PyTree, PyTree]:
        candidate = model.node.bind(param)
        if model.cyclic:
            return candidate.apply(state, input)
        return state, candidate.apply(input)

    model_member = Leaf(
        model_apply,
        param=model_param,
        init=model_init,
        name=model.name,
    ).bind(model.param, state=model.state)

    members = Composite(opt=pool, model=model_member)

    def apply(self, input, target: PyTree, rng) -> tuple[None, Aux]:
        element = Struct(input=input, target=target)

        def score(weights: PyTree) -> jax.Array:
            return fitness(model.node.bind(weights), element)

        children = self.opt.propose(rng=rng.next())
        best = self.opt(
            children=children,
            member_scores=jax.vmap(score)(self.opt.state.members),
            child_scores=jax.vmap(score)(children),
        )
        return None, Aux(loss=best)

    def trained(node, state) -> PSNode:
        """Binds the champion weights from `state.opt.params` onto the model member."""
        return node.members.model.bind(
            state.opt.params,
            state=state.model,
        )

    trainer = members(
        apply,
        name=f'evolve({model.name})',
        methods={'trained': trained},
    )
    return trainer.parameterize().initialize()


def test_evolution_wears_the_trainer_shape() -> None:
    """Verifies that `evolve` exposes `opt` and `model` members with champion weights at `state.opt.params`."""
    model = Affine().parameterize().initialize()
    trainer = evolve(model, pointwise_mse, differential_evolution(population=24),
                     rng=jax.random.PRNGKey(0))

    written = trainer.describe()
    print(written)
    assert 'opt: differential_evolution' in written and 'model:' in written
    assert trainer.state.opt.params.w == 0.0
    assert trainer.state.opt.members.w.shape == (24,)


def test_generations_scan_and_trained_finalizes() -> None:
    """Verifies that `.scan` updates affine weights over generations and agrees with `trained(trainer)`."""
    inputs = jnp.linspace(-1.0, 1.0, 16)
    targets = 2.0 * inputs + 1.0

    init_key, scan_key = jax.random.split(jax.random.PRNGKey(0))
    model = Affine().parameterize().initialize()
    trainer = evolve(model, pointwise_mse, differential_evolution(population=8),
                     rng=init_key)

    generations = 3
    advanced, (_, aux) = trainer.scan(input=tile(inputs, generations),
                                      target=tile(targets, generations),
                                      rng=scan_key)
    assert aux.loss.shape == (generations,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, advanced.state.opt.params, model.param))

    done, done_aux = trained(trainer).apply(input=tile(inputs, generations),
                                            target=tile(targets, generations),
                                            rng=scan_key)
    assert done.state_bound
    assert jnp.allclose(done_aux.loss, aux.loss)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, done.param, advanced.state.opt.params))
    _, prediction = done(jnp.asarray(0.5))
    assert prediction.shape == () and jnp.isfinite(prediction)
