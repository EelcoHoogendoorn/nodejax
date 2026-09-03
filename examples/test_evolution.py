"""A trainer with no gradient in it: differential evolution as a node.

The file is split on purpose to show a boundary. The top half is
differential evolution in plain jax, the two functions you would write
with no framework anywhere: propose and select, trees of stacked
candidate weights in and out. The bottom half wraps them into a cyclic
parametric Node with one improvement step per apply and its score riding
Aux. The population is carried where an optimizer carries its moments,
the reigning champion sits at state.opt.params, and the generation key
is entropy held as state. Its trained() view lets the stock trained
transform finalize an evolution run to the champion as a callable model;
scan runs generations, with neither knowing that no gradient was ever
taken.
"""

import jax
import jax.numpy as jnp

from typing import Callable

from nodejax import (
    node, Leaf, Node, Struct, Aux, PNode, PSNode, Composite, trained, tile,
)
from nodejax.core.types import PyTree
from examples.util import mse


# --- differential evolution, plain jax: usable with no framework ---

def propose(members: PyTree, key: jax.Array, rate: float,
            crossover: float, bounds: tuple | None = None) -> PyTree:
    """Propose one child for every member of the population. Each
    child starts from a randomly picked member, moves by `rate` times
    the difference of two other random members, and then takes each
    weight from either that mutant or its own parent with probability
    `crossover`, at least one weight always coming from the mutant
    (the recipe the literature calls DE/rand/1/bin). `bounds` is an
    optional (low, high) pair, scalars or trees, clipping the
    children. Partners are drawn with replacement, so a member
    occasionally mutates against itself, which only weakens that one
    proposal."""
    count = jax.tree.leaves(members)[0].shape[0]
    partners_key, mask_key, forced_key = jax.random.split(key, 3)
    picks = jax.random.randint(partners_key, (3, count), 0, count)
    anchor, first, second = (jax.tree.map(lambda leaf: leaf[rows], members)
                             for rows in picks)
    mutants = jax.tree.map(lambda base, high, low: base + rate * (high - low),
                           anchor, first, second)

    leaf_keys = jax.random.split(mask_key, len(jax.tree.leaves(members)))
    forced = jax.random.bernoulli(forced_key, 1.0 / max(1, len(leaf_keys)),
                                  (count,))

    def crossed(mutant, parent, leaf_key):
        take = jax.random.bernoulli(leaf_key, crossover, mutant.shape)
        take = take | forced.reshape((-1,) + (1,) * (mutant.ndim - 1))
        return jnp.where(take, mutant, parent)

    mutant_leaves, layout = jax.tree.flatten(mutants)
    parent_leaves = jax.tree.leaves(members)
    children = layout.unflatten([
        crossed(mutant, parent, leaf_key)
        for mutant, parent, leaf_key
        in zip(mutant_leaves, parent_leaves, leaf_keys)])
    if bounds is not None:
        low, high = bounds
        children = jax.tree.map(lambda leaf: jnp.clip(leaf, low, high), children)
    return children


def select(members: PyTree, children: PyTree, member_scores: jax.Array,
           child_scores: jax.Array) -> tuple[PyTree, jax.Array]:
    """Keep, for every population slot, whichever of parent and child
    scored better. Returns the survivors and their scores."""
    child_wins = child_scores < member_scores

    def per_member(flags, leaf):
        return flags.reshape((-1,) + (1,) * (leaf.ndim - 1))

    survivors = jax.tree.map(
        lambda child, parent: jnp.where(per_member(child_wins, child), child, parent),
        children, members)
    return survivors, jnp.minimum(child_scores, member_scores)


def pointwise_mse(candidate: PNode, element: Struct) -> jax.Array:
    """The simplest fitness: apply the candidate to the element's input
    and score against its target. A rollout, a domain cloud, or a
    percentile sits here just as well."""
    return mse(candidate.apply(element.input), element.target)


# --- the node wrappers: the same math, wearing the trainer's shape ---

@node
def Affine() -> Node:
    """The model under optimization: two weights, no interest in how
    they are found."""
    def param(w=0.0, b=0.0):
        return Struct(w=jnp.asarray(w), b=jnp.asarray(b))

    def apply(param, input):
        return param.w * input + param.b

    return Leaf(apply, param=param)


@node
def differential_evolution(population: int, rate: float = 0.7,
                           crossover: float = 0.9, spread: float = 1.0,
                           bounds=None) -> Node:
    """An optimizer with a population instead of moments: the argument
    evolve takes where train_step takes an optax one, swappable
    whole. Its state is Struct(params=<the champion>,
    members=<the population>, rng=<the generation key>), params sitting
    exactly where the family reads the live weights under train_step.
    Everything specific to DE lives here: the draw around the starting
    weights (init), the proposal (the propose method, calling the
    plain-jax half above), and the tournament (apply: scored children
    in, survivors and champion out, the key advanced)."""
    def init(rng, input):
        leaves, layout = jax.tree.flatten(input)
        member_keys = jax.random.split(rng.next(), population)

        def draw_member(member_key):
            # one key per LEAF per member: a shared draw would correlate
            # every weight and pin the whole population to a subspace
            # DE's linear recombination can never leave
            leaf_keys = jax.random.split(member_key, len(leaves))
            return layout.unflatten([
                leaf + spread * jax.random.normal(leaf_key, jnp.shape(leaf))
                for leaf, leaf_key in zip(leaves, leaf_keys)])

        members = jax.vmap(draw_member)(member_keys)
        return Struct(params=input, members=members, rng=rng.next())

    def propose_children(state):
        return propose(state.members, state.rng, rate, crossover, bounds)

    def apply(state, children, member_scores, child_scores):
        """The scoring happened outside, where the model lives; this is
        the tournament and the crowning."""
        survivors, scores = select(state.members, children,
                                   member_scores, child_scores)
        best = jnp.argmin(scores)
        champion = jax.tree.map(lambda leaf: leaf[best], survivors)
        advanced = jax.random.split(state.rng)[0]
        return Struct(params=champion, members=survivors, rng=advanced), scores[best]

    return Leaf(apply, init=init, methods=dict(propose=propose_children))


@node
def evolve(model: PSNode, fitness: Callable, optimizer: Node | PNode,
           rng) -> PSNode:
    """One improvement step per apply, in the trainer's shape, with the
    OPTIMIZER passed whole exactly as under train_step: this factory
    owes only the shape (the two members, the element contract, the
    aux), and everything evolutionary is the optimizer's. The model
    arrives fully bound and its bindings are the birth; `rng` is the
    population's birth entropy, spent where the reader sees it. Each
    apply asks the optimizer to propose and hands every candidate to
    fitness(candidate, element), the candidate an ordinary bound
    model (params are data) and the element handed over WHOLE, so
    the evaluation is the caller's: a pointwise loss, a rollout, a
    domain-randomized cloud with a percentile on top. Searching a
    transformed space (log-gains, bounded genomes) is the model's
    own business: params are whatever the model says they are."""
    if not model.state_bound:
        raise TypeError('evolve takes the model fully bound, exactly as '
                        f'train_step does; got {model!r}')
    pool = optimizer if optimizer.bound else optimizer.parameterize()
    pool = pool.initialize(rng=rng, input=model.param)

    # A trainer carries the model's state at ``state.model`` even for a
    # stateless model, where that state is simply ().  Express that shape as
    # an ordinary public leaf instead of manufacturing a compiled contract.
    # The adapter is intentionally boring: its params and initial state are
    # the model bindings that arrived, and stepping delegates to the public
    # PNode surface.
    def model_param():
        return model.param

    def model_init():
        return model.state

    def model_apply(param, state, input):
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

    def apply(self, input, target):
        element = Struct(input=input, target=target)

        def score(weights):
            return fitness(model.node.bind(weights), element)

        children = self.opt.propose()
        best = self.opt(
            children=children,
            member_scores=jax.vmap(score)(self.opt.state.members),
            child_scores=jax.vmap(score)(children),
        )
        return None, Aux(loss=best)

    def trained(node, state):
        """Bind the reigning champion as the optimized model."""
        return node.members.model.bind(
            state.opt.params,
            state=state.model,
        )

    trainer = members(
        apply,
        name=f'evolve({model.name})',
        methods={'trained': trained},
    )
    return trainer.parameterize().initialize(rng=rng)


def test_evolution_wears_the_trainer_shape():
    """The family reads shape, and this node has it: two members, the
    champion at state.opt.params, param what optimization starts from."""
    model = Affine().parameterize().initialize()
    trainer = evolve(model, pointwise_mse, differential_evolution(population=24),
                     rng=jax.random.PRNGKey(0))

    written = trainer.describe()
    print(written)
    assert 'opt: differential_evolution' in written and 'model:' in written
    assert trainer.state.opt.params.w == 0.0       # the champion starts home
    assert trainer.state.opt.members.w.shape == (24,)


def test_generations_scan_and_trained_finalizes():
    """No gradient anywhere, and the family composes anyway: .scan runs
    generations, trained() strikes the population and hands back the
    champion as a callable model."""
    inputs = jnp.linspace(-1.0, 1.0, 32)
    targets = 2.0 * inputs + 1.0

    model = Affine().parameterize().initialize()
    trainer = evolve(model, pointwise_mse, differential_evolution(population=24),
                     rng=jax.random.PRNGKey(0))

    generations = 120
    advanced, (_, aux) = trainer.scan(input=tile(inputs, generations),
                                      target=tile(targets, generations))
    assert aux.loss[-1] < 1e-3
    assert jnp.allclose(advanced.state.opt.params.w, 2.0, atol=0.05)

    done, aux = trained(trainer).apply(input=tile(inputs, generations),
                                       target=tile(targets, generations))
    assert done.state_bound                        # the population, struck
    _, prediction = done(jnp.asarray(0.5))
    assert jnp.allclose(prediction, 2.0, atol=0.1)
