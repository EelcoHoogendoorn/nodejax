from __future__ import annotations

from typing import TYPE_CHECKING

from nodejax.core import Node, NodeDef
from nodejax.authoring import node_def, derive
from nodejax.types import LossFn
from nodejax.generic import _over_generic
from nodejax.transforms.common import _over_bound
from nodejax.transforms.train_step import train_step

if TYPE_CHECKING:
    import optax


@_over_generic
@_over_bound
def finetune(nd: NodeDef, loss_fn: LossFn,
             optimizer: optax.GradientTransformation) -> NodeDef | Node:
    """Finetuning as a transform (the meta-learning literature calls this
    'adaptation'): parametric -> parametric, with the SAME param meaning.

    Where train_step moves params into evolving state, finetune closes the
    loop back to parametric: the result's param is the STARTING weights, and
    its apply takes Struct(support=Struct(input=..., target=...), query=...) —
    it scans the train step over the support sequence starting from param
    (one step per leading-axis element), keeps the final tuned weights (not
    the loss trace), and applies them to the query.

    This reframes 'training the model' as an ordinary differentiable function
    initial-weights -> prediction. Meta-learning is therefore literally
    train_step(finetune(model)), and task-batched MAML is
    train_step(batch(finetune(model))): learn an init that finetunes well.

    Stateful models finetune their internal state on the support set too — it
    rides along inside the inner trainer state, as in train_step.
    """
    if not nd.parametric:
        raise TypeError(f'finetune requires a parametric node, got {nd!r}')
    inner = train_step(nd, loss_fn, optimizer)

    def apply(param, input):
        tuned, _ = inner.scan(inner.init(model=param), input.support)
        _, output = nd.apply_fn(tuned.model, tuned.inner, input.query)
        return output

    # a derivation of the inner: same params (constructor, spec, axis
    # metadata inherited), the step machinery replaced by the
    # adapt-then-answer apply
    return derive(nd, apply=apply, name=f'finetune({nd.name})')
