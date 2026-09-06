"""Position-based dynamics over any entity record, from Node transforms.

Three things vary independently. The entities are a record, a Struct with a
leading entity axis, whose fields say what kind of body it is: particles or
rigid bodies. A constraint is a Node on the few entities it touches, the
physics. A schedule is how that Node is applied over a constraint set:
``gauss_seidel`` one after another through ``stack``, ``jacobi`` all at
once through ``ensemble`` with the corrections summed, ``red_black`` two
Jacobi passes in turn. ``Index`` gathers the entities a constraint touches
and scatters them back, and is the only place that knows about indices.

The entity collection is passed as data through individual constraint and
substep operations. At the outer timestep boundary, ``pbd_step`` and
``xpbd_step`` wrap ``PBDStep`` in ``cyclic``, promoting the collection
to state.
"""

from typing import Any

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
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
    """Indexed gather and scatter operations over entity PyTrees."""
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
def IndexedConstraint(index: Node, constraint: Node) -> Node:
    """Gather, project, and scatter-set one described constraint."""
    def apply(self, entities):
        gathered = self.index.gather(entities)
        projected = self.constraint(gathered)
        return self.index.scatter(entities, projected)

    return Composite(index=index, constraint=constraint)(apply)


@node
def IndexedConstraintCorrection(index: Node, constraint: Node) -> Node:
    """Compute displacement corrections from one constraint, zero elsewhere."""

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

    return Composite(index=index, constraint=constraint)(apply)


@node
def Displaced(corrections: Node) -> Node:
    """Displace an entity collection by summed constraint corrections."""
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


def safe_norm(v: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Safe Euclidean norm with non-zero gradient at the origin."""
    return jnp.sqrt(jnp.maximum(jnp.sum(v ** 2), eps))


def tree_dot(a: Any, b: Any) -> jax.Array:
    """Inner product between two matching PyTrees of arrays."""
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@node
def Constraint(objective: Node) -> Node:
    """Least-action constraint projection over any entity record."""
    def apply(self, pair):
        error, grad = jax.value_and_grad(self.objective)(pair.coords)
        delta = jax.tree.map(jnp.multiply, pair.inv_inertia, grad)
        denom = tree_dot(grad, delta) + self.objective.param.compliance
        step = error / jnp.maximum(denom, 1e-8)
        displaced = jax.tree.map(lambda q, dq: q - step * dq, pair.coords, delta)
        return pair.replace(**displaced.__as_dict__), Aux(error=error)

    return Wrapper(objective=objective)(apply)


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
def PBDStep(
    predict: Node,
    solve: Node,
    finalize: Node,
) -> Node:
    """One PBD timestep: predict free motion from force, solve constraints, and update velocity."""
    members = Composite(
        predict=predict,
        solve=solve,
        finalize=finalize,
    )

    def apply(self, entities, force):
        predicted = self.predict(entities, force)
        projected = self.solve(predicted)
        return self.finalize(entities, projected)

    return members(apply)
