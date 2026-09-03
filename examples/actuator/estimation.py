"""Model-based estimation: blend a filtered measurement with a model
prediction.

The measurement path is program structure: `filter` (a node or a
constructed node: sensor/filter pipeline) and the model function bind
at creation. The blend memory is a delay member and the blend itself a
node (mix) — nothing flows at step time except signals.
"""

from __future__ import annotations

from nodejax import Node, node, ambient, Composite
from nodejax.struct import Struct

from nodejax.control import Blend, Delay
from examples.actuator.dq import DQ


@node
def ModelEstimator(dt: float, filter, model_fn: Callable=lambda filtered, previous, model: filtered) -> Node:
    """Blend a filtered measurement with a model prediction.

    input fields: value (the measured value), model (whatever model_fn
    needs). model_fn(filtered, previous_blend, model) -> prediction;
    the default is the identity model, a pure measurement path. mix.tau
    is the model-influence time constant (0 = pure measurement); prev
    carries the previous blend (a DQ) for the model's prediction."""
    members = Composite(filter=filter, mix=Blend(dt)(tau=0.0),
                        prev=Delay().with_input(DQ()))

    def apply(self, value, model):
        filtered = self.filter(value)
        predicted = model_fn(filtered, self.prev.state, model)
        blended = self.mix(fast=filtered, slow=predicted)
        self.prev(blended)
        return blended

    return members(apply)
