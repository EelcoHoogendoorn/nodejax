"""Named-boundary state reinitialization (`state_reinit`) paired with `scan(..., boundary=...)`.

Sequential carries persist across `scan` calls unless a subtree declares a
named boundary handler. `state_reinit` attaches a handler on `Wrapper` that
replaces the wrapped subtree's carried state with a fresh `init` or `prime`
state whenever an enclosing `scan(..., boundary=name)` claims the matching
label, while leaving parameters and untagged sibling states untouched.
"""

from __future__ import annotations

from nodejax.core.node import Node
from nodejax.core.wrapper import Wrapper
from nodejax.transforms.transform import transform


@transform(preserves='param')
def state_reinit(inner: Node, boundary: str = 'episode') -> Node:
    """Reinitializes `inner`'s state tree at the start of any enclosing `scan(..., boundary=boundary)` call.

    Parameters remain unchanged, and sibling members without a matching
    `boundary` handler continue carrying their live state across calls. Because
    boundary walks run bottom-up, wrapping a composite overrides any conflicting
    `boundary` handlers declared by its descendants.
    """
    def take_init(carried, init, decided):
        return init

    return Wrapper(inner=inner)(
        name=f'state_reinit({inner.name})',
        boundary={boundary: take_init},
    )
