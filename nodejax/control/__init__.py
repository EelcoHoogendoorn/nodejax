"""nodejax.control — control-loop combinators, controllers, and plant blocks.

Capitalized names are units: a def that owns params or state, called to
get one. Lowercase names are transforms, which take a def and return a
def, owning nothing themselves.

Organized into modular subfiles:
- pid: PID, PD
- loops: feedback, closed_loop, observed_loop
- blocks: RateLimit, Clamp, ClampNorm, Delay, Diff, EMA, Blend,
          Gain, Integrator, Walker
"""

from nodejax.control.pid import PID, PD
from nodejax.control.loops import feedback, closed_loop, observed_loop
from nodejax.control.blocks import (
    RateLimit, Clamp, ClampNorm, Delay, Diff, EMA, Blend,
    Gain, Integrator, Walker,
)

__all__ = [
    'PID', 'PD',
    'feedback', 'closed_loop', 'observed_loop',
    'RateLimit', 'Clamp', 'ClampNorm', 'Delay', 'Diff', 'EMA', 'Blend',
    'Gain', 'Integrator', 'Walker',
]
