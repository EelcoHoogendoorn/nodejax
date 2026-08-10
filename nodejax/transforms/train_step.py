from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from nodejax.struct import Struct
from nodejax.types import LossFn
from nodejax.core import (Node, NodeDef, _trivial_param_fn, hoist_rng,
                                REQUIRED)
from nodejax.generic import _over_generic

if TYPE_CHECKING:
    import optax


@_over_generic
def train_step(node: NodeDef | Node, loss_fn: LossFn,
               optimizer: optax.GradientTransformation) -> Node:
    """Internalize the optimization loop: a parametric node becomes a bound
    cyclic node whose state holds (model params, optimizer state, model state).

    Because every def has a state slot, this works for stateful
    models (and pipes containing them) unchanged: the model's own state
    simply travels inside the trainer state.

    The def binds ONCE, at construction; runtime carries data only.
    init(model=<param pytree>, rng=..., inner=...) -> trainer state: the
        seeds mirror the state — model= the initial params, inner= the
        wrapped def's seed bundle (nested, the boundary hoist over the
        one wrapped slot), rng riding the boundary exactly when the
        inner init requires it. A def whose init reads shape is resolved
        by the caller before wrapping (with_input). Resuming from a
        checkpoint needs no init at all — a saved trainer state goes
        straight back into apply/scan.
    apply(state, input)  -> (state, loss)   (input: Struct(input=..., target=...))
    """
    import optax

    if not node.ndef.parametric:
        raise TypeError(f'train_step requires a parametric node, got {node.ndef!r}')

    # contract slots ndef/param/input are unread: the trainer has no params,
    # reflects no shape of its own (the inner state is shaped through the
    # trained def), and its input channel is the stream, roleless at init
    def init_fn(ndef, param, state_input, input=None):
        # the seeds mirror the trainer state: model= the initial params,
        # inner= the wrapped def's seeds, the hoisted boundary key
        # joining them — the one-member case of the composite split
        seed = state_input.inner if 'inner' in state_input else Struct()
        if 'rng' in state_input:
            seed = seed.replace(rng=state_input.rng)
        return Struct(model=state_input.model,
                      opt=optimizer.init(state_input.model),
                      inner=node.ndef.build_state(state_input.model, seed))

    def apply_fn(nd, _, ts, inp):
        def loss_wrapper(model_param):
            new_inner, output = node.ndef.apply_fn(model_param, ts.inner, inp.input)
            return loss_fn(output, inp.target), new_inner

        (loss, new_inner), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(ts.model)
        updates, new_opt = optimizer.update(grads, ts.opt, ts.model)
        new_model = optax.apply_updates(ts.model, updates)
        return Struct(model=new_model, opt=new_opt, inner=new_inner), loss

    # the stream contract is a MARKER spec: the trainer knows its element
    # FIELDS (input, target), their shapes only arrive with real data —
    # resolution fills them at the first apply/scan
    apply_input_spec = Struct(input=REQUIRED, target=REQUIRED)
    # the composite boundary hoist over the one wrapped slot —
    # Struct(rng=..., inner=...) — plus the trainer's own model field
    state_input_spec = hoist_rng(dict(
        inner=node.ndef.state_input_spec if node.ndef.cyclic else Struct()))
    out = NodeDef(f'train({node.ndef.name})', _trivial_param_fn, init_fn, apply_fn,
                  parametric=False, cyclic=True, apply_input_spec=apply_input_spec,
                  state_input_spec=state_input_spec.replace(model=REQUIRED),
                  tags=node.ndef.tags)
    return Node(out, ())
