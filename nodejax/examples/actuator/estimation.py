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

from nodejax.examples.actuator.blocks import blend_def, delay_def
from nodejax.examples.actuator.dq import DQ


@ambient
def model_estimator_def(dt, filter, model_fn=None):
    """Blend a filtered measurement with a model prediction.

    input: Struct(value=<measured value>, model=<whatever model_fn
    needs>). model_fn(filtered, previous_blend, input.model) ->
    prediction; None means identity (pure measurement path). mix.tau
    is the model-influence time constant (0 = pure measurement); prev
    carries the previous blend (a DQ) for the model's prediction."""
    members = dict(filter=filter, mix=blend_def(dt)(tau=0.0), prev=delay_def(DQ()))

    def apply(self, input):
        filtered = self.filter(input.value)
        predicted = filtered if model_fn is None else \
            model_fn(filtered, self.state.prev, input.model)
        blended = self.mix(Struct(fast=filtered, slow=predicted))
        self.prev(blended)
        return blended

    return composite(apply, members=members, name='model_estimator')
