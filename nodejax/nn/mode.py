"""Provides evaluation-mode specialization and running-statistic freezing for trained models."""

from __future__ import annotations

from nodejax.core.pnode import PNode
from nodejax.core.psnode import PSNode
from nodejax.transforms.policy.freeze import tree_freeze


def eval_mode(model: PSNode) -> PNode:
    """Rebuilds a state-bound `model` for evaluation and freezes `'running_stats'` state.

    Specializes every `'*.train'` static argument to `False`, rebinds `model.param` and `model.state`, and folds state tagged `'running_stats'` into fixed constants via `tree_freeze`.
    """
    rebuilt = model.specialize(**{'*.train': False})
    return tree_freeze(
        rebuilt.bind(model.param, state=model.state), tag='running_stats')
