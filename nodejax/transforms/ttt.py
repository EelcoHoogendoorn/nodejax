"""Test-time-training cells and their input/target assembly strategies."""

from __future__ import annotations

from nodejax.struct import Struct
from nodejax.node import BaseNode
from nodejax.pnode import PNode
from nodejax.wrapper import (Wrapper)
from nodejax.authoring import Leaf
from nodejax.ambient import node
from nodejax.transforms.train_step import _require_train_step


@node
def reconstruction() -> PNode:
    """Use each value as both input and target.

    This is useful only when a bottleneck, corruption, or other constraint
    prevents the learner from implementing the identity directly.
    """
    return Leaf(lambda input: Struct(input=input, target=input),)


@node
def next_step() -> PNode:
    """Pair the previous value as input with the current value as target."""
    def init(input):
        return input
    def apply(state, input):
        return input, Struct(input=state, target=input)
    return Leaf(apply, init=init)


@node
def ttt(step: BaseNode) -> BaseNode:
    """Mark a ``train_step`` node as a test-time-training step.

    Each call emits the prediction made before its update and carries the
    adapted weights in state. Inputs are ``Struct(input=..., target=...)``.
    """
    step_node = _require_train_step(step, 'ttt')
    marked = Wrapper(step=step_node)(
        name=f'ttt({step_node.name})')
    return step._with_definition(marked._def)


@node
def supervised_ttt(step: BaseNode):
    """Consume externally assembled ``input``/``target`` pairs."""
    return ttt(step)


@node
def next_step_ttt(step: BaseNode):
    """Train each value to predict the following value."""
    return next_step() >> ttt(step)


@node
def reconstruction_ttt(step: BaseNode):
    """Train each value to reconstruct itself."""
    return reconstruction() >> ttt(step)
