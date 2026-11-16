"""Tunes actuator observer time constants, loop gains, and sensor filters on the whip via differential evolution.

Searches eight log-space actuator parameters by rolling the whip out open-loop
under reversing strokes from multiple start angles and scoring velocity tracking
error, observer error, and torque chatter with `differential_evolution` from
`examples/test_evolution.py`. Each genome field maps by dotted path into the
plant's parameter tree via `replace_by_path`.
"""

import time

import jax
import jax.numpy as jnp

from nodejax import Leaf, Node, PNode, PSNode, Struct, node, replace_by_path, scan, split_aux, tile
from examples.test_evolution import differential_evolution, evolve
from examples.whip.learn import VELOCITY_SCALE
from examples.whip.plots import plot_training_curves
from examples.whip.whip import TARGET, TORQUE_LIMIT, whip_plant


GENOME = {  # genome field -> the plant parameter it sets, searched as a log
    'tau_pos': '.actuator.mechanical_est.observer.tau_pos',
    'tau_vel': '.actuator.mechanical_est.observer.tau_vel',
    'velocity_kp': '.actuator.command_ctrl.velocity_ctrl.kp',
    'velocity_ki': '.actuator.command_ctrl.velocity_ctrl.ki',
    'current_kp': '.actuator.current_ctrl.controller.kp',
    'current_ki': '.actuator.current_ctrl.controller.ki',
    'current_filter_tau': '.actuator.current_ctrl.estimator.filter.ema.tau',
    'bus_filter_tau': '.actuator.current_ctrl.bus_est.ema.tau',
}
LOG_FLOOR = 1e-3  # clamps zero-valued plant parameters before taking logs
LOG_BOUNDS = (jnp.log(1e-4), jnp.log(1e4))
CHATTER_WEIGHT = 0.5
N_STEPS = 100  # one second of strokes at the policy clock
STROKE_PERIOD = 25  # policy steps between reversals
STROKE_SETPOINT = 30.0  # rad/s at the shoulder, with the elbow at half and opposite sign
START_ANGLES = (0.3, 1.5, 2.8)
POPULATION = 16
N_GENERATIONS = 40


def stroke_commands(n_steps: int = N_STEPS, stroke_period: int = STROKE_PERIOD) -> jax.Array:
    """Builds `(time, joint)` shoulder strokes reversing every `stroke_period` steps with half-amplitude opposing elbow setpoints."""
    sign = jnp.where((jnp.arange(n_steps) // stroke_period) % 2 == 0, 1.0, -1.0)
    return jnp.stack((STROKE_SETPOINT * sign, -0.5 * STROKE_SETPOINT * sign), axis=-1)


@node
def Genome() -> Node:
    """Holds log-space tuned actuator parameters as a parametric Leaf for evolutionary search."""
    def param(tau_pos, tau_vel, velocity_kp, velocity_ki,
              current_kp, current_ki, current_filter_tau, bus_filter_tau) -> Struct:
        return Struct(
            tau_pos=tau_pos, tau_vel=tau_vel, velocity_kp=velocity_kp, velocity_ki=velocity_ki,
            current_kp=current_kp, current_ki=current_ki,
            current_filter_tau=current_filter_tau, bus_filter_tau=bus_filter_tau)

    def apply(param, input):
        return param

    return Leaf(apply, param=param)


def parameter_at(param: Struct, path: str) -> jax.Array:
    """Reads the parameter leaf at dotted `path` from `param`."""
    value = param
    for name in path.strip('.').split('.'):
        value = value[name]
    return value


def plant_genome(base: PNode) -> Struct:
    """Extracts the untuned genome parameter values from `base`."""
    return Struct(**{name: parameter_at(base.param, path) for name, path in GENOME.items()})


def initial_genome(base: PNode) -> PSNode:
    """Initializes `Genome` with the log-clamped parameter values carried by `base`."""
    logs = {name: jnp.log(jnp.maximum(value, LOG_FLOOR))
            for name, value in plant_genome(base).__items__}
    return Genome().parameterize(**logs).initialize()


def tuned_plant(base: PNode, genome: Struct) -> PNode:
    """Returns `base` with exponentiated `genome` values written into its parameter tree."""
    updates = {GENOME[name]: jnp.exp(genome[name]) for name in GENOME}
    return base.node.bind(replace_by_path(base.param, updates))


def scenario_cost(plant: PNode, command: jax.Array) -> Struct:
    """Evaluates `plant` under `command` across all start angles and averages tracking, estimation, and chatter costs."""
    n_steps = command.shape[0]

    def one_start(angle):
        state = plant.init(rng=jax.random.PRNGKey(0), start=plant.start(angle=angle, target=TARGET))
        program = scan(plant, n=n_steps).bind(state=state)
        _, outputs = program.apply(
            command=command, disturbance=jnp.zeros((n_steps,)), rng=jax.random.PRNGKey(1))
        outputs, _ = split_aux(outputs)
        true = plant.mechanical(state=outputs.state)
        estimate = outputs.state.actuator.mechanical_est.observer.velocity
        tracking = jnp.mean(((true.velocity - command) / VELOCITY_SCALE) ** 2)
        estimating = jnp.mean(((estimate - true.velocity) / VELOCITY_SCALE) ** 2)
        chatter = jnp.mean((jnp.diff(outputs.action, axis=0) / TORQUE_LIMIT) ** 2)
        return Struct(tracking=tracking, estimating=estimating, chatter=chatter)

    terms = jax.tree.map(jnp.mean, jax.vmap(one_start)(jnp.array(START_ANGLES)))
    return terms.replace(total=terms.tracking + terms.estimating + CHATTER_WEIGHT * terms.chatter)


def tune(
    base: PNode,
    n_generations: int = N_GENERATIONS,
    population: int = POPULATION,
    *,
    n_steps: int = N_STEPS,
    stroke_period: int = STROKE_PERIOD,
) -> Struct:
    """Evolves the actuator genome of `base` for `n_generations` and returns tuned values, initial values, and loss history."""
    def fitness(candidate: PNode, element: Struct) -> jax.Array:
        return scenario_cost(tuned_plant(base, candidate.param), element.input).total

    init_key, scan_key = jax.random.split(jax.random.PRNGKey(0))
    trainer = evolve(
        initial_genome(base), fitness,
        differential_evolution(population=population, spread=0.5, bounds=LOG_BOUNDS),
        rng=init_key,
    )
    command = stroke_commands(n_steps=n_steps, stroke_period=stroke_period)
    advanced, (_, aux) = trainer.scan(
        input=tile(command, n_generations),
        target=tile(jnp.zeros(()), n_generations),
        rng=scan_key,
    )
    champion = advanced.state.opt.params
    return Struct(
        tuned=jax.tree.map(jnp.exp, champion),
        initial=plant_genome(base),
        history=aux.loss,
    )


def main() -> None:
    """Tunes the whip actuator parameters by differential evolution and plots the cost curve."""
    base = whip_plant().parameterize()
    started = time.perf_counter()
    result = tune(base)
    jax.block_until_ready(result.history)
    seconds = time.perf_counter() - started
    command = stroke_commands()
    before = scenario_cost(base, command)
    after = scenario_cost(tuned_plant(base, jax.tree.map(jnp.log, result.tuned)), command)
    print(f'{N_GENERATIONS} generations of {POPULATION} in {seconds:.0f} s')
    print(f'{"parameter":20s} {"initial":>10s} {"tuned":>10s}')
    for name in GENOME:
        print(f'{name:20s} {float(result.initial[name]):10.4g} {float(result.tuned[name]):10.4g}')
    for label, terms in (('before', before), ('after', after)):
        print(f'{label}: total {float(terms.total):.4f} = tracking {float(terms.tracking):.4f} '
              f'+ estimating {float(terms.estimating):.4f} '
              f'+ {CHATTER_WEIGHT} x chatter {float(terms.chatter):.4f}')
    curve = plot_training_curves(
        {'actuator tuning by evolution': Struct(curve=result.history, seconds=seconds)},
        'whip_actuator_tuning.png')
    print(f'cost curve: {curve}')


if __name__ == '__main__':
    main()
