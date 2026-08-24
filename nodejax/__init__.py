"""Composable stateful computations for JAX.

``Node``, ``PNode``, and ``PSNode`` are the public unbound,
parameter-bound, and state-bound views of one internal definition.  Authored
functions are lowered once to a uniform contract; ready-made transforms then
compose those contracts without exposing their storage machinery here.
"""

from nodejax.struct import Struct
from nodejax.types import (PyTree, Param, State, Input, Output,
                                 ParamFn, InitFn, ApplyFn, LossFn, StaticTree)
from nodejax.binding import Aux, REQUIRED, split_aux
from nodejax.rng import KeyStream
from nodejax.composite import Composite
from nodejax.node import BaseNode, Node
from nodejax.generic import (Generic, is_generic)
from nodejax.pnode import (PNode)
from nodejax.psnode import (PSNode)
from nodejax.wrapper import (Wrapper)
from nodejax.authoring import Leaf, derive
from nodejax.paths import replace_by_path, set_by_path
from nodejax.transforms import (
    batch, unbatched, ensemble, reduce, stack, drop_aux, remat, repeat, scan, scanned, carried,
    trained, residual, train_step, supervised_ttt, next_step_ttt, reconstruction_ttt, finetune,
    taps, tie, externalize, parallel, at, reconstruction, next_step, freeze, tree_freeze, detach,
    tree_detach, tree_filter, map_members, tile
)
from nodejax.printing import (
    statics_by_path, describe, tree_view, print_tree, summary, print_summary
)
from nodejax.transforms import sum_junction, state_reinit
from nodejax.ambient import ambient, node
from nodejax.compose import serial
from nodejax import control
from nodejax.control import closed_loop, observed_loop
from nodejax.spec import spec, spec_of, materialize, meta

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
    'control',
    # transforms
    'batch', 'unbatched', 'ensemble', 'reduce', 'stack', 'drop_aux', 'remat', 'repeat', 'scan', 'scanned', 'carried', 'residual', 'train_step', 'trained', 'supervised_ttt', 'next_step_ttt', 'reconstruction_ttt',
    'finetune', 'tie', 'taps', 'externalize', 'parallel', 'sum_junction', 'state_reinit', 'at',
    'next_step', 'reconstruction',
    'freeze', 'tree_freeze', 'detach', 'tree_detach', 'tree_filter', 'map_members', 'statics_by_path', 'describe', 'tile', 'set_by_path', 'replace_by_path', 'closed_loop', 'observed_loop',
    'tree_view', 'print_tree', 'summary', 'print_summary',
    # composition
    'serial',
    # spec layer
    'spec', 'spec_of', 'materialize', 'meta',
]
