"""A composable FOC/PMSM actuator simulation.

The actuator is a stack of stateful blocks: battery -> bus-voltage
estimation -> command controller -> current controller (model-based
current estimation, one DQ PID, per-term voltage feedforward, thermal
rollback) -> electrical motor. Mechanical state is an input, integrated
at the environment level; every quantity the controller consumes is
estimated through sensor models while the physics runs on truth.
BenchMotor composes the same electrical motor with the mechanism
for open-loop physics work: cogging and hysteresis live with the
motor, stiction with the mechanism.

Capitalized names are units: a node that owns params or state, called to
get one. Lowercase names are pure of both, either a transform over nodes
or a node that is a plain function of its input.
"""

from examples.actuator.dq import DQ
from examples.actuator.motor import BenchMotor, Electrical, Mechanical
from examples.actuator.blocks import (Observer, Encoder, CurrentSensor,
                                              Noisy, Bag, wrap)
from examples.actuator.estimation import ModelEstimator
from examples.actuator.current_controller import (CurrentController,
                                                          Feedforward,
                                                          VelocityCommand,
                                                          foc_current_model,
                                                          torque_command)
from examples.actuator.power import Battery, Thermal, DeratingThermal, FET
from examples.actuator.actuator import ActuatorStack
