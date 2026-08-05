"""A composable FOC/PMSM actuator simulation.

The actuator is a stack of stateful blocks: battery -> bus-voltage
estimation -> command controller -> current controller (model-based
current estimation, one DQ PID, per-term voltage feedforward, thermal
rollback) -> electrical motor. Mechanical state is an input, integrated
at the environment level; every quantity the controller consumes is
estimated through sensor models while the physics runs on truth.
plant_def is the monolithic full-fidelity plant (cogging, stiction,
hysteresis) for open-loop physics work.
"""

from nodejax.examples.actuator.dq import DQ
from nodejax.examples.actuator.motor import (motor_params, motor_model_def, plant_def,
                                          emotor_def, mechanical_def)
from nodejax.examples.actuator.blocks import (pid_def, rate_limit_def, clamp_def, wrap_def,
                                           observer_def, encoder_def,
                                           current_sensor_def, ema_def, noisy_def)
from nodejax.examples.actuator.estimation import model_estimator_def
from nodejax.examples.actuator.current_controller import (current_controller_def, foc_current_model,
                                                       torque_command_def, velocity_command_def)
from nodejax.examples.actuator.power import battery_def, thermal_def, derating_thermal_def, fet_def
from nodejax.examples.actuator.actuator import actuator_stack_def
