from __future__ import annotations

from nodejax.ambient import node
from nodejax.node import BaseNode, Node
from nodejax.pnode import PNode
from nodejax.spec import add_axis, element_spec, tree_first
from nodejax.struct import Struct
from nodejax.transform import scan_steps
from nodejax.transforms.train_step import _require_train_step
from nodejax.wrapper import Wrapper


@node
def finetune(step: BaseNode) -> Node | PNode:
    """Run ``train_step`` on support data, then evaluate on a query.

    The step's parameters are the episode's starting parameters, so an
    outer optimization can differentiate through the support updates.
    """
    step_node = _require_train_step(step, 'finetune')
    model = step_node.members.model

    apply_takes_rng = (
        step_node.contract.init_takes_rng
        or step_node.contract.apply_takes_rng
        or model.contract.apply_takes_rng)

    def apply_fn(contract, param, input, rng):
        """Reset from ``param``, train on support, then evaluate the query."""
        current = contract.members.step
        current_model = current.members.model
        init_rng = rng.child(current.init_takes_rng)
        first = tree_first(input.support)
        start = current.prime(param, Struct(), first, init_rng)
        final, _ = scan_steps(
            current, param, start, input.support, rng)
        query_rng = rng.child(current_model.apply_takes_rng)
        _, output = current_model.apply(
            final.opt.params,
            final.model,
            current_model.feed(input.query),
            query_rng,
        )
        return output

    def param_fn(contract, param_input, rng):
        """Fresh meta-params through the train step's own constructor. The
        episode is Struct(support=<sequence>, query=<one input>) and the
        train step consumes one support element per call, so a shape-reading
        constructor beneath resolves against that slice rather than the
        episode."""
        current = contract.members.step
        episode = contract.input_spec
        current = current.for_input(
            None if episode is None else element_spec(episode.support))
        return current.param(param_input, rng)

    episode_spec = (
        Struct(
            support=add_axis(step_node.contract.input_spec),
            query=model.contract.intake(model.contract.input_spec),
        )
        if (step_node.contract.input_spec is not None
            and model.contract.input_spec is not None)
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
