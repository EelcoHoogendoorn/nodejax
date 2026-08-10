from __future__ import annotations

from nodejax.core import Node, NodeDef, _input_or_none, _resolve
from nodejax.struct import Struct
from nodejax.generic import _over_generic
from nodejax.paths import set_by_path
from nodejax.spec import materialize
from nodejax.transforms.common import _transform_def


@_over_generic
def externalize(node_def: NodeDef, member: str,
                at_init: Any | None = None) -> NodeDef:
    """Demote a subtree from parameter to input: the member disappears from
    the node's param tree and becomes a required field on `input`.

    The dual of parameterize: parameterize demotes input kwargs to params;
    externalize promotes a param slot to input kwargs. Use when ONE
    instance of a composite must vary across batch elements (which
    ensemble cannot do, because ensemble stacks ALL members) or when
    one member's params must vary where the rest are shared — per-task
    worlds under a meta-learned controller, one model evaluated
    against many plants — the varying subtree has to leave the tree,
    and the only other channel is input. externalize is that move;
    stock batch is then correct again.

    Scoping falls out as a bonus, tuner-agnostically: a gradient tuner
    given this node's params has no `member` to differentiate, so it
    cannot adapt the world it is being adapted to. (Derivative-free
    optimizers scope by enumeration instead — dicts of paths, as in
    the actuator tuning workflow; gradient transforms consume whole
    trees, so they scope by tree structure.)

    tie's sibling in the reparameterization family: where tie shares a
    subtree within the tree, externalize hands one to the caller as
    data. The member's params are bound at apply; init runs against
    `at_init` when supplied — a composite init that spec-propagates by
    running its members (a persisted scan makes the outer init real)
    needs values in the slot, and since state shapes are independent
    of them, the member's own param_fn defaults are the natural
    stand-in. With at_init omitted the slot stays empty at init,
    sufficient for inits that read shapes alone. param-rewriting:
    defs only."""
    if node_def.bound:
        raise TypeError('externalize changes the meaning of param; apply it '
                        'to the NodeDef')

    def param_fn(outer, param_input=Struct()):
        return set_by_path(node_def.build_param(param_input), {member: ()})

    def apply_fn(param, state, input):
        full = set_by_path(param, {member: input[member]})
        return node_def.apply_fn(full, state, input.input)

    def init_fn(outer, param, state_input=Struct(), input=None):
        if at_init is not None:
            param = set_by_path(param, {member: at_init})
        carry = input if input is not None else _input_or_none(outer)
        if carry is None:
            return node_def.build_state(param, state_input)
        # the inner runs on the .input field; the externalized member rides
        # in as data, so it is projected out of the offered shape
        inner_in = materialize(carry)['input']
        return _resolve(node_def, inner_in).build_state(param, state_input,
                                                    input=inner_in)

    return _transform_def(
        node_def,
        name=f'externalize({node_def.name})',
        param_fn=param_fn,
        init_fn=init_fn,
        apply_fn=apply_fn,
    )
