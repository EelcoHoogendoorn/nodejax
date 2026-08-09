"""Model-based estimation: blend a filtered measurement with a model
prediction.

The measurement path is program structure: `filter` (a def or a
constructed node: sensor/filter pipeline) and the model function bind
at creation. The blend memory is a delay member and the blend itself a
node (mix) — nothing flows at step time except signals.
"""

from __future__ import annotations

from nodejax import ambient, composite
from nodejax.struct import Struct

from nodejax.control import Blend, Delay
from nodejax.examples.actuator.dq import DQ


@ambient
def ModelEstimator(dt, filter, model_fn=lambda filtered, previous, model: filtered):
    """Blend a filtered measurement with a model prediction.

    input fields: value (the measured value), model (whatever model_fn
    needs). model_fn(filtered, previous_blend, model) -> prediction;
    the default is the identity model, a pure measurement path. mix.tau
    is the model-influence time constant (0 = pure measurement); prev
    carries the previous blend (a DQ) for the model's prediction."""
    members = dict(filter=filter, mix=Blend(dt)(tau=0.0), prev=Delay(DQ()))

    def apply(self, value, model=None):
        filtered = self.filter(value)
        predicted = model_fn(filtered, self.state.prev, model)
        blended = self.mix(fast=filtered, slow=predicted)
        self.prev(blended)
        return blended

    return composite(apply, members=members, name='model_estimator')
