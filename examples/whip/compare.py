"""SHAC and PPO on the same whip, on one axis.

The differentiable learner gets a gradient through the motor's current
loops, the encoder's rounding, and the rope's constraint projection. The
gradient-free one gets returns. Same plant, same policy body, same
evaluation.
"""

import jax

from nodejax import Struct
from examples.whip.learn import (
    WhipFeatures, evaluation_starts, summarize, timed, whip_evaluation, write_artifacts,
)
from examples.whip.ppo_whip import mean_policy, train_whip_ppo
from examples.whip.shac_whip import train_whip_shac
from examples.whip.plots import plot_training_curves
from examples.whip.whip import whip_plant


def main() -> None:
    plant = whip_plant().parameterize()
    features = WhipFeatures(tip_sensor=False)
    shac = timed(lambda: train_whip_shac(plant, features))
    ppo = timed(lambda: train_whip_ppo(plant, features))
    print('SHAC mean cost per step by episode:', shac.result.history.mean_cost)
    print('PPO mean cost per step by iteration:', ppo.result.history.mean_cost)
    print(f'wall time: SHAC {shac.seconds:.0f} s, PPO {ppo.seconds:.0f} s')
    runs = {
        'SHAC': Struct(curve=shac.result.history.mean_cost, seconds=shac.seconds),
        'PPO': Struct(curve=ppo.result.history.mean_cost, seconds=ppo.seconds),
    }
    print(f'training curves: {plot_training_curves(runs, "whip_training_compared.png")}')
    starts = evaluation_starts(plant)
    for label, policy in (('SHAC', shac.result.policy), ('PPO', mean_policy(ppo.result.policy))):
        evaluation = whip_evaluation(policy, plant, starts, jax.random.PRNGKey(21))
        summarize(evaluation, label)
        write_artifacts(evaluation, f'whip_{label.lower()}')


if __name__ == '__main__':
    main()
