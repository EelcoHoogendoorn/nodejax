"""Defines `ModelEstimator`, blending filtered measurements with physics-model predictions."""

from __future__ import annotations

from collections.abc import Callable

from nodejax import BaseNode, Node, PNode, PyTree, node, Composite

from nodejax.control import Blend, Delay
from examples.actuator.dq import DQ


@node
def ModelEstimator(
        dt: float,
        filter: BaseNode,
        model_fn: Callable[[PyTree, PyTree, PyTree], PyTree] = (
            lambda filtered, previous, model: filtered
        ),
) -> Node | PNode:
    """Blends `filter(value)` with `model_fn(filtered, prev.state, model)` using a one-pole `Blend` Node.

    Defaults `mix.tau` to `0.0` (pure measurement path) and stores each blended `DQ` estimate in `prev` for the next step's prediction.
    """
    members = Composite(filter=filter, mix=Blend(dt)(tau=0.0),
                        prev=Delay().with_input(DQ()))

    def apply(self, value, model):
        filtered = self.filter(value)
        predicted = model_fn(filtered, self.prev.state, model)
        blended = self.mix(fast=filtered, slow=predicted)
        self.prev(blended)
        return blended

    return members(apply)
