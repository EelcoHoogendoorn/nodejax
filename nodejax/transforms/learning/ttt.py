"""Online test-time-training (`ttt`) wrappers and self-supervised pair builders.

A `train_step` Node already emits the pre-update prediction and carries adapted
weights in state, so `ttt` wraps a `train_step` directly as an online adaptive
cell while `supervised_ttt`, `next_step_ttt`, and `reconstruction_ttt` select
how each step's `(input, target)` pair is formed:

- `supervised_ttt` consumes externally supplied `(input, target)` pairs.
- `next_step_ttt` delays the stream by one step (`next_step`) so the previous
  observation predicts the current observation.
- `reconstruction_ttt` duplicates each observation (`reconstruction`) as both
  `input` and `target` for bottleneck or denoising adaptation.
"""

from __future__ import annotations

from nodejax.core.ambient import node
from nodejax.core.authoring import Leaf
from nodejax.core.node import BaseNode
from nodejax.core.pnode import PNode
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.learning.train_step import _require_train_step


@node
def reconstruction() -> PNode:
    """Duplicates each `input` into `Struct(input=input, target=input)` for reconstruction objectives."""
    return Leaf(lambda input: Struct(input=input, target=input))


@node
def next_step() -> PNode:
    """Carries the previous `input` in state and emits `Struct(input=previous, target=current)`."""
    def init(input):
        return input
    def apply(state, input):
        return input, Struct(input=state, target=input)
    return Leaf(apply, init=init)


@node
def ttt(step: BaseNode) -> BaseNode:
    """Labels a `train_step` Node as an online test-time-training cell carrying adapted weights in state."""
    step_node = _require_train_step(step, 'ttt')
    marked = Wrapper(step=step_node)(
        name=f'ttt({step_node.name})')
    return step._with_definition(marked._def)


@node
def supervised_ttt(step: BaseNode) -> BaseNode:
    """Adapts `step` online from externally supplied `(input, target)` pairs."""
    return ttt(step)


@node
def next_step_ttt(step: BaseNode) -> BaseNode:
    """Chains `next_step()` into `ttt(step)` so each observation adapts `step` to predict the next one."""
    return next_step() >> ttt(step)


@node
def reconstruction_ttt(step: BaseNode) -> BaseNode:
    """Chains `reconstruction()` into `ttt(step)` so each observation adapts `step` to reconstruct itself."""
    return reconstruction() >> ttt(step)
