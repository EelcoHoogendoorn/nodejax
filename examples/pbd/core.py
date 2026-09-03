"""Position-based dynamics over any entity record, from Node transforms.

Three things vary independently. The entities are a record, a Struct with a
leading entity axis, whose fields say what kind of body it is: particles or
rigid bodies. A constraint is a Node on the few entities it touches, the
physics. A schedule is how that Node is applied over a constraint set:
``gauss_seidel`` one after another through ``stack``, ``jacobi`` all at
once through ``ensemble`` with the corrections summed, ``red_black`` two
Jacobi passes in turn. ``Index`` gathers the entities a constraint touches
and scatters them back, and is the only place that knows about indices.

The record is data, not state, through every repetition beneath a
timestep: over the constraints of a pass, over the passes of a solve, over
substeps. Each of those is a map from a record to a record. Only
``PBDStep`` is time, one call one tick. ``pbd_step`` and ``xpbd_step``
wrap it in ``cyclic``, which makes the record the world's state there and
nowhere beneath. Nothing under it needs a state slot, and everything under
it batches, stacks, and differentiates as a map.

"""

import jax
import jax.numpy as jnp

from nodejax import (
    Composite,
    Leaf,
    Node,
    PNode,
    Struct,
    Wrapper,
    ensemble,
    node,
    reduce,
    serial,
    stack,
    tree_len,
    tree_take,
)


@node
def Index() -> Node:
    """Bound indices into an entity collection, used through its methods:
    ``gather`` takes the indexed entities, ``scatter`` writes them back,
    ``scatter_add`` adds to them. It has no apply of its own."""
    def param(indices):
        return indices

    def apply(param, entities):
        raise TypeError('Index is used through gather, scatter, and scatter_add')

    def gather(param, entities):
        return tree_take(entities, param)

    def scatter(param, entities, local):
        return jax.tree.map(
            lambda whole, part: whole.at[param].set(part), entities, local)

    def scatter_add(param, entities, delta):
        return jax.tree.map(
            lambda whole, part: whole.at[param].add(part), entities, delta)

    return Leaf(apply, param=param, methods=dict(
        gather=gather, scatter=scatter, scatter_add=scatter_add))


@node
def IndexedConstraint(
    index: Node,
    constraint: Node,
) -> Node:
    """Gather, project, and scatter-set one described constraint."""
    members = Composite(index=index, constraint=constraint)

    def apply(self, entities):
        gathered = self.index.gather(entities)
        projected = self.constraint(gathered)
        return self.index.scatter(entities, projected)

    return members(apply)


@node
def IndexedConstraintCorrection(
    index: Node,
    constraint: Node,
) -> Node:
    """The displacement one projected constraint asks of the whole collection:
    zero except at its own indexed entities."""
    members = Composite(index=index, constraint=constraint)

    def apply(self, entities):
        gathered = self.index.gather(entities)
        projected = self.constraint(gathered)
        diff = jax.tree.map(
            lambda projected_leaf, gathered_leaf: projected_leaf - gathered_leaf,
            projected,
            gathered,
        )
        zero = jax.tree.map(jnp.zeros_like, entities)
        return self.index.scatter_add(zero, diff)

    return members(apply)


@node
def Displaced(corrections: Node) -> Node:
    """The collection moved by the summed corrections of every constraint."""
    def apply(self, entities):
        delta = self.corrections(entities)
        return jax.tree.map(
            lambda entity_leaf, delta_leaf: entity_leaf + delta_leaf,
            entities,
            delta,
        )

    return Wrapper(corrections=corrections)(apply)


def gauss_seidel(constraints: Struct, constraint: Node) -> PNode:
    """Apply ``constraint`` to the constraints one after another via ``stack``."""
    projection = IndexedConstraint(index=Index(), constraint=constraint)
    return stack(projection, n=tree_len(constraints)).bind(constraints)


def tree_sum(tree, axis=0):
    """Sum each array leaf in a PyTree across ``axis``."""
    return jax.tree.map(lambda leaf: jnp.sum(leaf, axis=axis), tree)


def jacobi(constraints: Struct, constraint: Node) -> Node:
    """Apply ``constraint`` to every constraint from the same positions and add up
    the corrections: an ``ensemble`` of one correction per constraint, summed
    by ``reduce``."""
    corrections = ensemble(
        IndexedConstraintCorrection(Index(), constraint), n=tree_len(constraints)
    ).bind(constraints)
    return Displaced(corrections >> reduce(tree_sum))


def red_black(constraints: Struct, constraint: Node) -> Node:
    """Jacobi over two colors in turn, the even constraints then the odd ones."""
    indices = jnp.arange(tree_len(constraints))
    return serial(
        even=jacobi(tree_take(constraints, indices[::2]), constraint),
        odd=jacobi(tree_take(constraints, indices[1::2]), constraint),
    )


@node
def Broadcast(value: tuple | jax.Array) -> Node:
    """Broadcast one constant vector over the entities of a collection."""
    constant = jnp.array(value)
    return Leaf(lambda entities: jnp.broadcast_to(constant, entities.position.shape))


@node
def PBDStep(
    forcing: Node,
    predict: Node,
    solve: Node,
    finalize: Node,
) -> Node:
    """One timestep, a map from an entity collection to the next: force it,
    predict free motion, project the constraints, and update velocity."""
    members = Composite(
        forcing=forcing,
        predict=predict,
        solve=solve,
        finalize=finalize,
    )

    def apply(self, entities):
        predicted = self.predict(entities, self.forcing(entities))
        projected = self.solve(predicted)
        return self.finalize(entities, projected)

    return members(apply)
