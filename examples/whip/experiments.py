"""Questions about the learners, each answered by a run with descriptive
artifacts: PPO given ten times the comparison's iterations; each learner
with the tip sensed, the privileged variant, against blind; SHAC on the
ideal actuator against the stack; both learners on the exact encoder
against the quantized one over several seeds; SHAC at several gradient
clips; and every learner on the stiff-loop exact-encoder plant.
"""

import jax
import optax

from nodejax import Struct
from examples.rl import control
from examples.whip.learn import (
    WhipFeatures,
    evaluation_starts,
    summarize,
    timed,
    whip_critic,
    whip_evaluation,
    write_animation,
    write_artifacts,
)
from examples.whip.mppi_whip import learned_controller, train_whip_mppi
from examples.whip.ppo_whip import mean_policy, train_whip_ppo
from examples.whip.plots import plot_training_curves
from examples.whip.shac_whip import ACTOR_OPTIMIZER, ACTOR_RATE, train_whip_shac
from examples.whip.whip import LINK_LENGTHS, VELOCITY_GAIN, VELOCITY_GAIN_STIFF, whip_plant


N_PPO_ITERATIONS = 600
N_SHAC_EPISODES = 120  # about the wall time of the PPO run
N_MPPI_ITERATIONS = 40  # of 150 steps: the same transitions as 120 of the 50-step iterations
EVALUATION_KEY = jax.random.PRNGKey(21)


def configured(
    *,
    tip_sensor: bool = False,
    ideal_actuator: bool = False,
    exact_encoder: bool = False,
    velocity_gain: float = VELOCITY_GAIN,
    link_lengths: tuple = LINK_LENGTHS,
) -> Struct:
    """The plant of one experimental condition and the features a policy
    reads of it, built once and handed to the trainer and the evaluation
    alike."""
    plant = whip_plant(
        tip_sensor=tip_sensor, ideal_actuator=ideal_actuator, exact_encoder=exact_encoder,
        velocity_gain=velocity_gain, link_lengths=link_lengths)
    return Struct(plant=plant.parameterize(), features=WhipFeatures(tip_sensor))


def run_ppo(label: str, prefix: str, condition: Struct, *, seed: int = 0) -> Struct:
    """Train PPO on ``condition``, report, and write artifacts under
    ``prefix``; the cost curve and the wall seconds come back for the
    shared figure."""
    run = timed(lambda: train_whip_ppo(
        condition.plant, condition.features, iterations=N_PPO_ITERATIONS, seed=seed))
    print(f'{label}: {run.seconds:.0f} s for {N_PPO_ITERATIONS} iterations')
    evaluation = whip_evaluation(
        mean_policy(run.result.policy), condition.plant, evaluation_starts(condition.plant), EVALUATION_KEY)
    summarize(evaluation, label)
    write_artifacts(evaluation, prefix)
    return Struct(curve=run.result.history.mean_cost, seconds=run.seconds)


def run_shac(
    label: str,
    prefix: str,
    condition: Struct,
    *,
    seed: int = 0,
    actor_optimizer=ACTOR_OPTIMIZER,
    artifacts=write_artifacts,
) -> Struct:
    """Train SHAC on ``condition``, report, and write ``artifacts`` under
    ``prefix``; the cost curve and the wall seconds come back for the
    shared figure. The ideal actuator has no readouts to draw, so its runs
    pass ``write_animation``."""
    run = timed(lambda: train_whip_shac(
        condition.plant, condition.features, n_episodes=N_SHAC_EPISODES, seed=seed,
        actor_optimizer=actor_optimizer))
    print(f'{label}: {run.seconds:.0f} s for {N_SHAC_EPISODES} episodes')
    evaluation = whip_evaluation(
        run.result.policy, condition.plant, evaluation_starts(condition.plant), EVALUATION_KEY)
    summarize(evaluation, label)
    artifacts(evaluation, prefix)
    return Struct(curve=run.result.history.mean_cost, seconds=run.seconds)


def run_mppi(label: str, prefix: str, condition: Struct) -> Struct:
    """Train the MPPI terminal critic on ``condition``, report, and write
    artifacts under ``prefix``; the cost curve and the wall seconds come
    back for the shared figure. The planner sees the true state, so only
    the condition's actuators reach it."""
    run = timed(lambda: train_whip_mppi(
        condition.plant, whip_critic(condition.plant), n_iterations=N_MPPI_ITERATIONS))
    explained = run.result.history.critic_explained
    print(f'{label}: {run.seconds:.0f} s for {N_MPPI_ITERATIONS} iterations, '
          f'critic explained variance {float(explained[:10].mean()):.2f} over the first ten '
          f'and {float(explained[-10:].mean()):.2f} over the last ten')
    controller = learned_controller(run.result.critic, condition.plant)
    evaluation = whip_evaluation(
        controller, condition.plant, evaluation_starts(condition.plant), EVALUATION_KEY, step=control.PlannedStep)
    summarize(evaluation, label)
    write_artifacts(evaluation, prefix)
    return Struct(curve=run.result.history.mean_cost, seconds=run.seconds)


def ppo_blind_versus_tip_sensed() -> None:
    runs = {
        'PPO, blind': run_ppo(
            'PPO, blind, 600 iterations', 'whip_ppo_blind_600_iterations', configured()),
        'PPO, tip sensed': run_ppo(
            'PPO, tip sensed, 600 iterations', 'whip_ppo_tip_sensed_600_iterations',
            configured(tip_sensor=True)),
    }
    curves = plot_training_curves(runs, 'whip_ppo_blind_vs_tip_sensed_600_iterations.png')
    print(f'training curves: {curves}')


def shac_blind_versus_tip_sensed() -> None:
    runs = {
        'SHAC, blind': run_shac(
            'SHAC, blind, 120 episodes', 'whip_shac_blind_120_episodes', configured()),
        'SHAC, tip sensed': run_shac(
            'SHAC, tip sensed, 120 episodes', 'whip_shac_tip_sensed_120_episodes',
            configured(tip_sensor=True)),
    }
    curves = plot_training_curves(runs, 'whip_shac_blind_vs_tip_sensed_120_episodes.png')
    print(f'training curves: {curves}')


def shac_stack_versus_ideal_actuator() -> None:
    """SHAC through the actuator stack against SHAC through plain torque
    sources, blind and tip sensed."""
    runs = {}
    for tip_sensor in (False, True):
        for ideal_actuator in (False, True):
            sensing = 'tip sensed' if tip_sensor else 'blind'
            joints = 'ideal actuator' if ideal_actuator else 'actuator stack'
            label = f'SHAC, {sensing}, {joints}'
            prefix = (f'whip_shac_{sensing.replace(" ", "_")}_{joints.replace(" ", "_")}'
                      f'_{N_SHAC_EPISODES}_episodes')
            runs[label] = run_shac(
                label, prefix, configured(tip_sensor=tip_sensor, ideal_actuator=ideal_actuator),
                artifacts=write_animation if ideal_actuator else write_artifacts)
    curves = plot_training_curves(
        runs, f'whip_shac_actuator_stack_vs_ideal_{N_SHAC_EPISODES}_episodes.png')
    print(f'training curves: {curves}')


def quantized_versus_exact_encoder(seeds: tuple = (0, 1, 2)) -> None:
    """Both blind learners with the stock encoder and with none, the
    observer fed the true angle, over several seeds, since one run of SHAC
    says little on its own."""
    runs = {}
    for encoder, exact in (('quantized encoder', False), ('exact encoder', True)):
        slug = encoder.replace(' ', '_')
        condition = configured(exact_encoder=exact)
        for seed in seeds:
            label = f'SHAC, {encoder}, seed {seed}'
            prefix = f'whip_shac_blind_{slug}_seed_{seed}_{N_SHAC_EPISODES}_episodes'
            runs[label] = run_shac(label, prefix, condition, seed=seed)
            label = f'PPO, {encoder}, seed {seed}'
            prefix = f'whip_ppo_blind_{slug}_seed_{seed}_{N_PPO_ITERATIONS}_iterations'
            runs[label] = run_ppo(label, prefix, condition, seed=seed)
    curves = plot_training_curves(runs, 'whip_quantized_vs_exact_encoder.png')
    print(f'training curves: {curves}')


def shac_gradient_clip(clips: tuple = (1.0, 10.0, None)) -> None:
    """SHAC at each actor gradient clip, tip sensed with the exact encoder
    so that sensing is not the excuse; ``None`` is plain Adam."""
    runs = {}
    condition = configured(tip_sensor=True, exact_encoder=True)
    for clip in clips:
        bound = 'no clip' if clip is None else f'clip {clip:g}'
        optimizer = (
            optax.adam(ACTOR_RATE) if clip is None else
            optax.chain(optax.clip_by_global_norm(clip), optax.adam(ACTOR_RATE)))
        label = f'SHAC, tip sensed, exact encoder, {bound}'
        slug = bound.replace(' ', '_')
        prefix = f'whip_shac_tip_sensed_exact_encoder_{slug}_{N_SHAC_EPISODES}_episodes'
        runs[label] = run_shac(label, prefix, condition, actor_optimizer=optimizer)
    curves = plot_training_curves(runs, f'whip_shac_gradient_clip_{N_SHAC_EPISODES}_episodes.png')
    print(f'training curves: {curves}')


def strong_actuator_all_learners() -> None:
    """Every learner on the exact encoder, whose velocity loop is stiff
    enough to sit on the torque clamp: PPO and SHAC with the tip sensed,
    MPPI on the true state."""
    sensed = configured(tip_sensor=True, exact_encoder=True, velocity_gain=VELOCITY_GAIN_STIFF)
    runs = {
        'PPO, tip sensed, exact encoder': run_ppo(
            'PPO, tip sensed, exact encoder',
            f'whip_ppo_tip_sensed_exact_encoder_strong_{N_PPO_ITERATIONS}_iterations', sensed),
        'SHAC, tip sensed, exact encoder': run_shac(
            'SHAC, tip sensed, exact encoder',
            f'whip_shac_tip_sensed_exact_encoder_strong_{N_SHAC_EPISODES}_episodes', sensed),
        'MPPI, exact encoder': run_mppi(
            'MPPI, exact encoder',
            f'whip_mppi_exact_encoder_strong_{N_MPPI_ITERATIONS}_iterations',
            configured(exact_encoder=True, velocity_gain=VELOCITY_GAIN_STIFF)),
    }
    curves = plot_training_curves(runs, 'whip_strong_actuator_all_learners.png')
    print(f'training curves: {curves}')


def longer_upper_arm(upper_arms: tuple = (0.3, 0.45, 0.6), seed: int = 1) -> None:
    """PPO, tip sensed on the exact encoder, on arms whose upper link grows
    while the forearm stays 0.3 m: whether distinct link lengths give the
    two joints distinct roles."""
    runs = {}
    for upper_arm in upper_arms:
        label = f'PPO, upper arm {upper_arm:.2f} m'
        prefix = f'whip_ppo_upper_arm_{upper_arm:.2f}_m_{N_PPO_ITERATIONS}_iterations'
        condition = configured(tip_sensor=True, exact_encoder=True, link_lengths=(upper_arm, 0.3))
        runs[label] = run_ppo(label, prefix, condition, seed=seed)
    curves = plot_training_curves(runs, 'whip_ppo_upper_arm_lengths.png')
    print(f'training curves: {curves}')


def main() -> None:
    ppo_blind_versus_tip_sensed()
    shac_blind_versus_tip_sensed()
    shac_stack_versus_ideal_actuator()
    quantized_versus_exact_encoder()
    shac_gradient_clip()
    strong_actuator_all_learners()
    longer_upper_arm()


if __name__ == '__main__':
    main()
