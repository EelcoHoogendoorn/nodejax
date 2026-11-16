"""Provides feedback-loop combinators, PID controllers, signal-conditioning filters, and discrete plant models."""

from nodejax.control.pid import PID, PD
from nodejax.transforms.wiring import sum_junction        # block-diagram vocabulary
from nodejax.control.loops import closed_loop, observed_loop
from nodejax.control.blocks import (
    Quantize, Deadband, RateLimit, Clamp, Delay, Diff, EMA,
    MovingAverage, Blend, Gain, Integrator, Walker,
)
from nodejax.control.systems import FirstOrder, StateSpace

__all__ = [
    'PID', 'PD',
    'closed_loop', 'observed_loop', 'sum_junction',
    'Quantize', 'Deadband', 'RateLimit', 'Clamp', 'Delay',
    'Diff', 'EMA', 'MovingAverage', 'Blend', 'Gain', 'Integrator', 'Walker',
    'FirstOrder', 'StateSpace',
]
