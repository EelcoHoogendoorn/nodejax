"""Defines entity-agnostic constraint projection, index gather/scatter, and Gauss-Seidel/Jacobi solver schedules.

Separates three orthogonal choices: the entity `Struct` with a leading entity axis (`Particle` or `Body`), the local `Constraint` `Node` acting on the subset of entities it couples, and the solver schedule (`gauss_seidel` via `stack`, `jacobi` via `ensemble` and `reduce`, or `red_black` via two alternating `jacobi` passes). `Index` isolates entity-axis gather and scatter operations, while `pbd_step` and `xpbd_step` wrap `PBDStep` in `cyclic` so the entity collection becomes carried state across outer timesteps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    PyTree,
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
    """Provides indexed `gather`, `scatter`, and `scatter_add` methods over entity PyTrees."""
    def param(indices: PyTree) -> PyTree:
        return indices

    def apply(param, entities):
        raise TypeError('Index is used through gather, scatter, and scatter_add')

    def gather(param, entities: PyTree) -> PyTree:
        return tree_take(entities, param)

    def scatter(param, entities: PyTree, local: PyTree) -> PyTree:
        return jax.tree.map(
            lambda whole, part: whole.at[param].set(part), entities, local)

    def scatter_add(param, entities: PyTree, delta: PyTree) -> PyTree:
        return jax.tree.map(
            lambda whole, part: whole.at[param].add(part), entities, delta)

    return Leaf(apply, param=param, methods=dict(
        gather=gather, scatter=scatter, scatter_add=scatter_add))


@node
def IndexedConstraint(index: BaseNode, constraint: BaseNode) -> Node | PNode:
    """Gathers the entities addressed by `index`, projects `constraint`, and scatters the updated entities back."""
    def apply(self, entities: PyTree) -> PyTree:
        gathered = self.index.gather(entities)
        projected = self.constraint(gathered)
        return self.index.scatter(entities, projected)

    return Composite(index=index, constraint=constraint)(apply)


@node
def IndexedConstraintCorrection(index: BaseNode, constraint: BaseNode) -> Node | PNode:
    """Computes sparse entity-level displacement deltas induced by one indexed `constraint`."""
    def apply(self, entities: PyTree) -> PyTree:
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
def Displaced(corrections: BaseNode) -> Node | PNode:
    """Adds summed constraint displacements from `corrections(entities)` back onto `entities`."""
    def apply(self, entities: PyTree) -> PyTree:
        delta = self.corrections(entities)
        return jax.tree.map(
            lambda entity_leaf, delta_leaf: entity_leaf + delta_leaf,
            entities,
            delta,
        )

    return Wrapper(corrections=corrections)(apply)


def gauss_seidel(constraints: Struct, constraint: BaseNode) -> PNode:
    """Projects `constraint` sequentially across `constraints` via `stack`."""
    projection = IndexedConstraint(index=Index(), constraint=constraint)
    return stack(projection, n=tree_len(constraints)).bind(constraints)


def tree_sum(tree: PyTree, axis: int = 0) -> PyTree:
    """Sums each array leaf in `tree` along `axis`."""
    return jax.tree.map(lambda leaf: jnp.sum(leaf, axis=axis), tree)


def safe_norm(vector: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Computes `sqrt(max(sum(vector ** 2), eps))` with a finite gradient at the origin."""
    return jnp.sqrt(jnp.maximum(jnp.sum(vector ** 2), eps))


def tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    """Computes the scalar inner product across corresponding array leaves of `left` and `right`."""
    return sum(
        jnp.vdot(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


@node
def Constraint(objective: BaseNode) -> Node | PNode:
    """Projects a local entity tuple onto `objective(pair.coords) == 0` weighted by `pair.inv_inertia` and `compliance`."""
    def apply(self, pair: PyTree) -> tuple[PyTree, Aux]:
        error, grad = jax.value_and_grad(self.objective)(pair.coords)
        delta = jax.tree.map(jnp.multiply, pair.inv_inertia, grad)
        denom = tree_dot(grad, delta) + self.objective.param.compliance
        step = error / jnp.maximum(denom, 1e-8)
        displaced = jax.tree.map(lambda q, dq: q - step * dq, pair.coords, delta)
        return pair.replace(**displaced.__as_dict__), Aux(error=error)

    return Wrapper(objective=objective)(apply)


def jacobi(constraints: Struct, constraint: BaseNode) -> Node | PNode:
    """Evaluates all `constraints` concurrently from the same entity poses via `ensemble` and sums their corrections."""
    corrections = ensemble(
        IndexedConstraintCorrection(Index(), constraint), n=tree_len(constraints)
    ).bind(constraints)
    return Displaced(corrections >> reduce(tree_sum))


def red_black(constraints: Struct, constraint: BaseNode) -> Node | PNode:
    """Runs two alternating `jacobi` passes over even-indexed and odd-indexed `constraints`."""
    indices = jnp.arange(tree_len(constraints))
    return serial(
        even=jacobi(tree_take(constraints, indices[::2]), constraint),
        odd=jacobi(tree_take(constraints, indices[1::2]), constraint),
    )


@node
def PBDStep(
    predict: BaseNode,
    solve: BaseNode,
    finalize: BaseNode,
) -> Node | PNode:
    """Executes one PBD timestep by predicting unconstrained motion, solving constraints, and recovering velocities."""
    members = Composite(
        predict=predict,
        solve=solve,
        finalize=finalize,
    )

    def apply(self, entities: PyTree, force: PyTree) -> PyTree:
        predicted = self.predict(entities, force)
        projected = self.solve(predicted)
        return self.finalize(entities, projected)

    return members(apply)
