"""Node transforms, written once against the contract.

Each transform consumes and produces nodes (closure), so they compose freely
with each other and with pipes. Param-preserving transforms (batch, scan,
repeat, residual, finetune) also accept bound Nodes and rebind;
param-rewriting ones (ensemble, stack) require nodes. All lift over the
static stage: applied to a generic they defer, commuting with
specialization.

One file per transform; shared lifting helpers in common.py.
"""

from nodejax.transforms.batch import batch, unbatched
from nodejax.transforms.ensemble import ensemble, reduce
from nodejax.transforms.stack import stack
from nodejax.transforms.drop_aux import drop_aux
from nodejax.transforms.remat import remat
from nodejax.transforms.repeat import repeat
from nodejax.transforms.scan import scan, scanned, carried
from nodejax.transforms.residual import residual
from nodejax.transforms.train_step import train_step, trained
from nodejax.transforms.ttt import (supervised_ttt, next_step_ttt,
                                    reconstruction_ttt, next_step, reconstruction)
from nodejax.transforms.finetune import finetune
from nodejax.transforms.tie import tie
from nodejax.transforms.taps import taps
from nodejax.transforms.externalize import externalize
from nodejax.transforms.parallel import parallel
from nodejax.transforms.sum_junction import sum_junction
from nodejax.transforms.boundary import state_reinit
from nodejax.transforms.at import at
from nodejax.transforms.freeze import freeze, tree_freeze, detach, tree_detach
from nodejax.transforms.tree_utils import (
    map_members, tile, tree_filter)
from nodejax.printing import (
    describe, statics_by_path, tree_view, print_tree, summary, print_summary)

__all__ = ['batch', 'unbatched', 'ensemble', 'reduce', 'stack', 'drop_aux', 'remat', 'repeat',
           'scan', 'scanned', 'carried', 'residual',
           'train_step', 'trained', 'supervised_ttt', 'next_step_ttt', 'reconstruction_ttt',
           'next_step', 'reconstruction', 'finetune', 'tie', 'taps', 'externalize',
           'parallel', 'sum_junction', 'state_reinit', 'at', 'freeze', 'tree_freeze', 'detach',
           'tree_detach', 'tree_filter', 'map_members', 'statics_by_path', 'describe', 'tile',
           'tree_view', 'print_tree', 'summary', 'print_summary']
