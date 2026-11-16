"""Episodic support-set adaptation and query evaluation (`finetune`).

`finetune` wraps a `train_step` Node into a stateless episodic Node taking
`(support, query)`. Each call primes fresh optimizer and model state from the
episode's outer `param` tree, scans support updates, extracts the adapted model
via `.trained()`, and evaluates `query`, keeping the entire inner trajectory
differentiable with respect to the starting parameters (MAML-style meta-learning).
"""

from __future__ import annotations

from nodejax.core.ambient import node
from nodejax.core.node import BaseNode, Node
from nodejax.core.pnode import PNode
from nodejax.core.spec import add_axis, element_spec
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.learning.train_step import (
    _model_contract, _require_train_step,
)
from nodejax.transforms.transform import bind, scan_steps
from nodejax.tree import tree_first


@node
def finetune(step: BaseNode) -> Node | PNode:
    """Adapts a `train_step` across a `support` sequence and evaluates the adapted model on `query`.

    Outer `param` values seed each episode's initial weights before scanning
    `support`, so an outer `train_step` or `jax.grad` differentiates directly
    through the inner adaptation updates.
    """
    step_node = _require_train_step(step, 'finetune')
    model = _model_contract(step_node.contract)

    apply_takes_rng = (
        step_node.contract.init_takes_rng
        or step_node.contract.apply_takes_rng
        or model.apply_takes_rng)

    def apply_fn(contract, param, input, rng):
        current = contract.members.step
        current_model = _model_contract(current)
        init_rng = rng.child(current.init_takes_rng)
        first = tree_first(input.support)
        start = current.prime(param, Struct(), first, init_rng)
        final, _ = scan_steps(
            current, param, start, input.support, rng)
        fitted = bind(current, param, state=final).trained()
        _, output = fitted.contract.apply(
            fitted.param,
            fitted.state,
            fitted.contract.feed(input.query),
            rng.child(current_model.apply_takes_rng),
        )
        return output

    def param_fn(contract, param_input, rng):
        # Shape-reading constructors beneath `step` resolve against a single support element.
        current = contract.members.step
        episode = contract.input_spec
        current = current.for_input(
            None if episode is None else element_spec(episode.support))
        return current.param(param_input, rng)

    episode_spec = (
        Struct(
            support=add_axis(step_node.contract.input_spec),
            query=model.intake(model.input_spec),
        )
        if (step_node.contract.input_spec is not None
            and model.input_spec is not None)
        else None
    )
    episode = Wrapper(step=step_node).roles(
        name=f'finetune({step.name})',
        param=param_fn,
        init=False,
        apply=apply_fn,
        apply_fields=('support', 'query'),
        input_spec=episode_spec,
        apply_takes_rng=apply_takes_rng,
    )
    # Bound step parameters become the differentiable episode start.
    return step._transfer_bindings(episode, ('param',))
