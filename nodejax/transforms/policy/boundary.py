"""What restarts at a boundary.

A carry carries. That is what a scan does between calls and what it keeps
doing at a boundary, so state outlives a run unless something says otherwise.
This transform is how a node says otherwise, and it is an ordinary node: the
declaration lives on the thing it describes, and the enclosing scan
contributes only the boundary's IDENTITY, as a tag both sides name.

The underlying mechanism is more general than this one transform, and a node
can reach it directly by passing `boundary=` to inner: an action over its
own state layout, run at the tag it names. What the library ships is the case
worth a name. A node wanting to hold one slot back, or to keep one across a
restart, writes the action itself, over its own slot names.

The tag is a rendezvous label, not a jax axis. batch and ensemble name theirs
because vmap's axis_name exists so collectives have something to reduce over;
a scan is sequential and jax has no such name. This one exists purely so a
declaration and the transform that satisfies it can find each other, which is
also why a node under two boundaries can say which one it answers to.
"""

from __future__ import annotations

from nodejax.core.node import Node
from nodejax.transforms.transform import transform
from nodejax.core.wrapper import Wrapper


@transform(preserves='param')
def state_reinit(inner: Node, boundary: str = 'episode') -> Node:
    """This subtree's state is RE-INITED at the named boundary, where it
    would otherwise carry on as it does everywhere else.

    The name says both halves because both are the point: it is the STATE that
    goes, params being untouched, and it goes by running the node's own init
    rather than by being zeroed or cleared.

    A recurrent carry that should not remember the previous sequence, a
    feedback register that should not start the next episode mid-swing. The
    state that accumulates on purpose (a covariance, a belief, running
    statistics) says nothing, because carrying is what a carry does.

    It claims its WHOLE subtree, since the walk runs bottom up and this node
    acts last: one above a member overrides what that member decided.

    Structurally free. A Wrapper's state IS the wrapped node's, so this adds
    no level to any tree and no key to any state; it is a node whose only job
    is to be found by the walk that assembles the boundary."""
    def take_init(carried, init, decided):
        return init

    return Wrapper(inner=inner)(
        name=f'state_reinit({inner.name})',
        boundary={boundary: take_init},
    )
