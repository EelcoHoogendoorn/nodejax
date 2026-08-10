from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.types import LossFn
from nodejax.core import Node, NodeDef, _input_or_none, hoist_rng
from nodejax.authoring import node_def
from nodejax.generic import _over_generic
from nodejax.transforms.common import _transform_def


# The standard self-supervision assembly: shapes a wire into the
# (input, target) element a ttt cell consumes per step, with the input
# itself as the target. Its siblings (next-step prediction via a delay
# register, masked views via rng) are the caller's to write as nodes of
# the same shape.
reconstruction = node_def(lambda input: Struct(input=input, target=input),
                          name='reconstruction')


@_over_generic
def ttt(node: NodeDef, loss_fn: LossFn, lr0: float) -> NodeDef:
    """Weights-as-state: the wrapped node's params become the cell's
    state, and the state transition is one gradient step per sample
    (test-time training). metasgd's per-step sibling: the same
    param = Struct(init, lr) — initialization and per-leaf step
    sizes, both meta-learned by whatever optimizes this node — but
    adaptation runs every step of the stream, where metasgd runs per
    support episode.

    The cell consumes train_step's stream element, one per step:
    Struct(input=<the node's input>, target=<what this sample
    teaches>), scored by an ordinary loss_fn(output, target).
    Self-supervision is a property of the data assembly, and the
    caller's modeling choice: derive target from the stream itself —
    the input again (reconstruction), the value after it (next-step
    prediction), a held-out view (masking). The order is
    predict-then-update (prequential, the classical online-learning
    discipline): the output comes from the weights as they arrived,
    so every prediction is of a sample those weights have never
    trained on, and the update lands after, ready for the next step.
    A cyclic wrapped node keeps its own state threading alongside the
    adapted weights: two memories at two speeds inside one cell."""
    if node.bound:
        raise TypeError('ttt changes the meaning of param; apply it to the '
                        'NodeDef and parameterize the (init, lr) pair')
    if not node.parametric:
        raise TypeError(f'ttt requires a parametric node, got {node!r}')

    def param(param: Struct) -> Struct:
        # the cell's construction inputs ARE the wrapped node's: the bundle
        # passes through verbatim (rng required/optional/absent exactly as
        # the inner declares), and lr seeds at lr0 alongside
        init = node.build_param(param)
        return Struct(init=init, lr=jax.tree.map(lambda x: jnp.full_like(x, lr0), init))

    def init(ndef, param: Struct, state: Struct) -> Struct:
        # the inner's seeds nest under inner=, mirroring the state field;
        # the hoisted boundary key joins them. The cell's stream element is
        # Struct(input=..., target=...), and the wrapped node runs on the
        # .input field, so its state builds from that slice of the shape
        seed = state.inner if 'inner' in state else Struct()
        if 'rng' in state:
            seed = seed.replace(rng=state.rng)
        carry = _input_or_none(ndef)
        inner_in = carry['input'] if carry is not None else None
        return Struct(w=param.init,
                      inner=node.build_state(param.init, seed, input=inner_in))

    def apply(param: Struct, state: Struct, input: Struct):
        def inner_loss(w):
            s2, out = node.apply_fn(w, state.inner, input.input)
            return loss_fn(out, input.target), (s2, out)
        grads, (new_inner, out) = jax.grad(inner_loss, has_aux=True)(state.w)
        w = jax.tree.map(lambda wi, g, lr: wi - lr * g, state.w, grads, param.lr)
        return Struct(w=w, inner=new_inner), out

    lifted = node_def(apply, param=param, init=init, name=f'ttt({node.name})')
    # the seed spec is the boundary hoist over the one wrapped slot,
    # mirroring the cell's state field
    seed_spec = hoist_rng(dict(inner=node.state_input_spec if node.cyclic else Struct()))
    lifted = lifted._replace(param_input_spec=node.param_input_spec,
                             state_input_spec=seed_spec)
    return _transform_def(
        node,
        name=lifted.name,
        param_fn=lifted._param_impl,
        init_fn=lifted._init_impl,
        apply_fn=lifted._apply_impl,
        parametric=lifted.parametric,
        cyclic=lifted.cyclic,
        apply_input_spec=lifted.apply_input_spec,
        init_requires_input=False,
        init_reads_shape=False,
        state_input_spec=lifted.state_input_spec,
        tags=lifted.tags,
        rebuild=lambda d: ttt(d, loss_fn, lr0),
    )
