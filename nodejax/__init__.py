"""Public surface for composing and transforming functional stateful Nodes in JAX.

`Node`, `PNode`, and `PSNode` expose the unbound, parameter-bound, and state-bound views of a normalized internal definition (`Def`), paired with authoring helpers (`@node`, `Leaf`, `Composite`, `Wrapper`) and contract-preserving transforms.
"""

from nodejax.struct import Struct
from nodejax.core.types import (PyTree, Param, State, Input, Output,
                                 ParamFn, InitFn, ApplyFn, LossFn, StaticTree)
from nodejax.core.binding import Aux, REQUIRED, split_aux
from nodejax.core.rng import KeyStream
from nodejax.core.composite import Composite
from nodejax.core.node import BaseNode, Node
from nodejax.core.generic import (Generic, is_generic)
from nodejax.core.pnode import (PNode)
from nodejax.core.psnode import (PSNode)
from nodejax.core.wrapper import (Wrapper)
from nodejax.core.authoring import Leaf, derive
from nodejax.paths import replace_by_path, set_by_path
from nodejax.transforms import (
    batch, unbatched, ensemble, reduce, stack, drop_aux, remat, repeat, repeated, iterated, scan, scanned, carried,
    trained, residual, train_step, supervised_ttt, next_step_ttt, reconstruction_ttt, finetune,
    taps, tie, cyclic, externalize, parallel, at, reconstruction, next_step, freeze, tree_freeze, detach,
    tree_detach, sum_junction, state_reinit,
)
from nodejax.transforms.structure import map_members, tree_filter
from nodejax.tree import (
    tile, tree_broadcast_axis, tree_first, tree_last, tree_len, tree_reshape,
    tree_tail,
    tree_swap_axes, tree_take,
    tree_stop_gradient,
)
from nodejax.core.printing import (
    statics_by_path, describe, tree_view, print_tree, summary, print_summary
)
from nodejax.core.ambient import ambient, node
from nodejax.core.compose import serial
from nodejax import control, nn
from nodejax.control import closed_loop, observed_loop
from nodejax.core.spec import spec, spec_of, materialize

__all__ = [
    'Struct',
    'Aux',
    'REQUIRED',
    # types
    'PyTree', 'Param', 'State', 'Input', 'Output',
    'ParamFn', 'InitFn', 'ApplyFn', 'LossFn', 'StaticTree',
    # core
    'BaseNode', 'PNode', 'PSNode', 'Node', 'Generic', 'is_generic',
    'Composite', 'Wrapper', 'KeyStream', 'split_aux',
    # authoring
    'node', 'ambient', 'Leaf', 'derive',
    # submodules
    'control', 'nn',
    # transforms
    'batch', 'unbatched', 'ensemble', 'reduce', 'stack', 'drop_aux', 'remat', 'repeat', 'repeated', 'iterated', 'scan', 'scanned', 'carried', 'residual', 'train_step', 'trained', 'supervised_ttt', 'next_step_ttt', 'reconstruction_ttt',
    'finetune', 'tie', 'taps', 'cyclic', 'externalize', 'parallel', 'sum_junction', 'state_reinit', 'at',
    'next_step', 'reconstruction',
    'freeze', 'tree_freeze', 'detach', 'tree_detach', 'tree_filter', 'map_members', 'statics_by_path', 'describe', 'tile', 'tree_broadcast_axis', 'tree_first', 'tree_last', 'tree_len', 'tree_reshape', 'tree_swap_axes', 'tree_take', 'tree_tail', 'tree_stop_gradient', 'set_by_path', 'replace_by_path', 'closed_loop', 'observed_loop',
    'tree_view', 'print_tree', 'summary', 'print_summary',
    # composition
    'serial',
    # spec layer
    'spec', 'spec_of', 'materialize',
]
