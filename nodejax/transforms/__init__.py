"""Node transforms, written once against the contract.

Each transform consumes and produces defs (closure), so they compose freely
with each other and with pipes. Param-preserving transforms (batch, scan,
repeat, residual, finetune) also accept bound Nodes and rebind;
param-rewriting ones (ensemble, stack) require defs. All lift over the
static stage: applied to a GenericDef they defer, commuting with
specialization.

One file per transform; shared lifting helpers in common.py.
"""

from nodejax.transforms.batch import batch, unbatched
from nodejax.transforms.ensemble import ensemble
from nodejax.transforms.stack import stack
from nodejax.transforms.repeat import repeat
from nodejax.transforms.scan import scan
from nodejax.transforms.residual import residual
from nodejax.transforms.train_step import train_step
from nodejax.transforms.finetune import finetune
from nodejax.transforms.metasgd import metasgd
from nodejax.transforms.tie import tie
from nodejax.transforms.taps import taps
from nodejax.transforms.externalize import externalize
from nodejax.transforms.parallel import parallel
from nodejax.transforms.at import at
from nodejax.transforms.ttt import ttt, reconstruction
from nodejax.transforms.freeze import freeze, tree_freeze, detach, tree_detach
from nodejax.transforms.tree_utils import map_members, tree_filter

__all__ = ['batch', 'unbatched', 'ensemble', 'stack', 'repeat', 'scan', 'residual',
           'train_step', 'finetune', 'metasgd', 'tie', 'taps', 'externalize',
           'parallel', 'at', 'ttt', 'freeze', 'tree_freeze', 'detach',
           'tree_detach', 'tree_filter', 'map_members']
