"""Defines discrete unit-feedback and observer-feedback loop Composites."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.ambient import node
from nodejax.core.composite import Composite
from nodejax.core.node import BaseNode, Node
from nodejax.core.pnode import PNode
from nodejax.core.types import PyTree
from nodejax.control.blocks import Delay
from nodejax.transforms.policy import state_reinit


@node
def closed_loop(pipe: BaseNode, output_spec: PyTree = 0.0) -> Node | PNode:
    """Closes a unit-feedback loop around `pipe` using a one-step `Delay` register `last`.

    Feeds `error = input - last.state` into `pipe`, emits `pipe(error)`, and stores that output into `last` for the next step. Because `last.state` is read before `last` is called, forward shape propagation does not reach `last` first; `output_spec` supplies the signal spec carried around the loop. Preserves `pipe` as a direct Composite member so inner collective nodes such as `BatchNorm` remain visible to outer transforms.
    """
    def apply(self, input) -> PyTree:
        error = jax.tree.map(jnp.subtract, input, self.last.state)
        out = self.pipe(error)
        self.last(out)
        return out

    last = Delay().with_input(output_spec)
    return Composite(pipe=pipe, last=last)(apply)


@node
def observed_loop(
        pipe: BaseNode, belief_spec: PyTree, output_spec: PyTree = 0.0,
) -> Node | PNode:
    """Closes a unit-feedback loop around an observer-augmented `pipe` that consumes `(error, belief)` and emits `Struct(output=..., belief=...)`.

    Carries two `Delay` registers (`last` for `output_spec` and `belief` for `belief_spec`). Marks `last` with `state_reinit` so multi-episode scans reset the measurement register at episode boundaries while carrying the observer `belief` across episodes.
    """
    def apply(self, input) -> PyTree:
        error = jax.tree.map(jnp.subtract, input, self.last.state)
        out = self.pipe(error=error, belief=self.belief.state)
        self.last(out.output)
        self.belief(out.belief)
        return out.output

    # Reset measurement feedback at episode boundaries while carrying belief across episodes.
    last = state_reinit(Delay().with_input(output_spec))
    belief = Delay().with_input(belief_spec)
    return Composite(pipe=pipe, last=last, belief=belief)(apply)
