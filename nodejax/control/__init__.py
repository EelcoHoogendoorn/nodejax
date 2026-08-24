"""Control-loop combinators, controllers, filters, and plant blocks.

Factories return components with explicit params and state where the modeled
system requires them. Lowercase loop functions transform components into
larger closed-loop components.

Organized into modular subfiles:
- pid: PID, PD
- loops: feedback, closed_loop, observed_loop
- blocks: signal nonlinearities, filters, gain, integration, and noise
- systems: FirstOrder, StateSpace
"""

from nodejax.control.pid import PID, PD
from nodejax.transforms import sum_junction        # block-diagram vocabulary
from nodejax.control.loops import feedback, closed_loop, observed_loop
from nodejax.control.blocks import (
    Quantize, Deadband, RateLimit, Clamp, ClampNorm, Delay, Diff, EMA,
    MovingAverage, Blend, Gain, Integrator, Walker,
)
from nodejax.control.systems import FirstOrder, StateSpace

__all__ = [
    'PID', 'PD',
    'feedback', 'closed_loop', 'observed_loop', 'sum_junction',
    'Quantize', 'Deadband', 'RateLimit', 'Clamp', 'ClampNorm', 'Delay',
    'Diff', 'EMA', 'MovingAverage', 'Blend', 'Gain', 'Integrator', 'Walker',
    'FirstOrder', 'StateSpace',
]
