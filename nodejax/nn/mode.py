"""Evaluation-mode rebuilding for trained models."""

from __future__ import annotations

from nodejax.core.pnode import PNode
from nodejax.core.psnode import PSNode
from nodejax.transforms.policy.freeze import tree_freeze


def eval_mode(model: PSNode) -> PNode:
    """Rebuild a trained model for evaluation.

    Every ``train`` static becomes False, the model's params and state are
    rebound, and state tagged 'running_stats' freezes at its trained
    values.
    """
    rebuilt = model.specialize(**{'*.train': False})
    return tree_freeze(
        rebuilt.bind(model.param, state=model.state), tag='running_stats')
