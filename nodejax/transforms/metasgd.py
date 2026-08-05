from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.types import LossFn
from nodejax.core import Node, NodeDef
from nodejax.authoring import node_def
from nodejax.generic import _over_generic
from nodejax.transforms.train_step import train_step


def _leaf_sgd(lrs) -> 'optax.GradientTransformation':
    """sgd with a per-leaf step-size pytree (values may be traced)."""
    import optax

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return jax.tree.map(lambda g, lr: -lr * g, updates, lrs), state

    return optax.GradientTransformation(init_fn, update_fn)


@_over_generic
def metasgd(node: NodeDef, loss_fn: LossFn, lr0: float) -> NodeDef:
    """finetune with the inner step sizes as meta-params (Meta-SGD):
    param = Struct(init=<the node's params>, lr=<same-shaped step
    sizes, seeded at lr0>), both trained by whatever optimizes this
    node. apply takes Struct(support=Struct(input=..., target=...),
    query=...): one inner gradient step per support element from
    `init` at the learned per-parameter rates, then the tuned weights
    run the query.

    The conditioning of adaptation is thereby learned, never hand-set —
    the outer loop may discover rates spanning orders of magnitude,
    including negative ones. Stateful models carry their state through
    the inner trainer exactly as in finetune."""
    if node.bound:
        raise TypeError('metasgd changes the meaning of param; apply it to the '
                        'NodeDef and parameterize the (init, lr) pair')
    if not node.parametric:
        raise TypeError(f'metasgd requires a parametric node, got {node!r}')

    def param(param: Struct) -> Struct:
        # construction inputs pass through to the wrapped node verbatim;
        # lr seeds at lr0 alongside
        init = node.build_param(param)
        return Struct(init=init, lr=jax.tree.map(lambda x: jnp.full_like(x, lr0), init))

    def apply(param: Struct, input: Struct):
        # the model's own input is one support episode's .input field —
        # resolve what you wrap, so the inner state inits see the shape
        element = jax.tree.map(lambda x: x[0], input.support.input)
        inner = train_step(node.with_input(element), loss_fn, _leaf_sgd(param.lr))
        tuned, _ = inner.scan(inner.init(model=param.init), input.support)
        _, output = node.apply_fn(tuned.model, tuned.inner, input.query)
        return output

    lifted = node_def(apply, param=param, name=f'metasgd({node.name})')
    return lifted._replace(param_input_spec=node.param_input_spec)
