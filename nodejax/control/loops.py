"""Control loop feedback combinators."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax.ambient import node
from nodejax.composite import (Composite)
from nodejax.node import (Node)
from nodejax.compose import composite
from nodejax.control.blocks import Delay
from nodejax.transforms import state_reinit




@node
def feedback(pipe: Node, output_spec: Any = 0.0) -> Node:
    """Close the loop: the wrapped node maps tracking error to output, and
    feedback supplies error = reference minus the previous output.

    A COMPOSITE of the controlled node and the register carrying its output
    back round, because that is what a closed loop is made of. The register
    is the stock one-tick Delay, starting at rest like any cold register.

    `output_spec` says what flows round the loop, scalar by default and a
    shaped zero for a MIMO loop. The register is READ before it is fed, which
    is what closing a loop means, so forward shape propagation never reaches
    it and the loop must declare it. The value is deducible in principle,
    since the register carries the pipe's output and jax.eval_shape derives
    that exactly; it is passed because resolving a member from the composite
    that holds it is machinery the framework has not got. See scratch/todo.md.

    Deriving from the pipe instead would flatten it into an opaque leaf,
    taking its members with it: a normalizer inside the loop would then be
    handed per-sample statistics under batch(), silently."""
    def apply(self, input):
        error = jax.tree.map(jnp.subtract, input, self.state.last)
        out = self.pipe(error)
        self.last(out)                        # carry it round for the next step
        return out

    last = Delay().with_input(output_spec)
    return Composite(pipe=pipe, last=last)(apply, name=f'feedback({pipe.name})')


@node(name='loop')
def closed_loop(pipe: Node, output_spec: Any = 0.0) -> Node:
    """Unit feedback around an actuation pipe: reference in, measurement
    out, with the pipe mapping tracking error to actuation to measurement.
    feedback under another name, kept for the control vocabulary."""
    def apply(self, input):
        error = jax.tree.map(jnp.subtract, input, self.state.last)
        out = self.pipe(error)
        self.last(out)
        return out

    last = Delay().with_input(output_spec)
    return Composite(pipe=pipe, last=last)(apply)


@node(name='oloop')
def observed_loop(pipe: Node, belief_spec: Any, output_spec: Any = 0.0) -> Node:
    """closed_loop with an in-loop observer riding the plant side — the
    indirect-adaptive-control block. Two registers, so two members: the
    fed-back measurement and the observer's belief. Both are read before
    they are fed, so both declare what they carry, `belief_spec` being the
    one nothing else could supply: a belief is the pipe's second output and
    has nothing to do with the loop's signal."""
    def apply(self, input):
        error = jax.tree.map(jnp.subtract, input, self.state.last)
        out = self.pipe(error=error, belief=self.state.belief)
        self.last(out.output)
        self.belief(out.belief)
        return out.output

    # the belief OUTLIVES an episode, the fed-back measurement does not:
    # the half that departs from carrying declares it, inert until an
    # enclosing scan claims the boundary
    last = state_reinit(Delay().with_input(output_spec))
    belief = Delay().with_input(belief_spec)
    return Composite(pipe=pipe, last=last, belief=belief)(apply)
