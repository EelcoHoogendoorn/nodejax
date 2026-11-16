"""Models a composable field-oriented-control (FOC) permanent-magnet synchronous motor (PMSM) actuator.

Composes encoder and observer sensing, outer command control, `DQ`-frame current control with model-based current estimation, voltage feedforward, thermal derating, and electrical motor physics. Mechanical state and battery bus voltage enter as external environment inputs so one `Battery` pack can power multiple `ActuatorStack` instances; `BenchMotor` pairs `Electrical` and `Mechanical` directly for open-loop characterization.
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
