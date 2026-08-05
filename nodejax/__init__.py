"""nodejax — stateful computation composed the way JAX composes functions.

Every node is three pure functions against one uniform contract:

    param_fn(param_input)                    -> param pytree
    init_fn(param, state_input, input=None)  -> state pytree
    apply_fn(param, state, input)            -> (state, output)

A NodeDef stores the three plus metadata (input specs, flags, methods);
a Node is a def bound to a param pytree, and flattens to exactly those
params. The node lattice (parametric x cyclic):

    N    = Node(def, ())                trivial param, trivial state
    CN   = Node(def, ())                nontrivial state
    PN   = NodeDef(parametric)          parameterize() -> Node
    PCN  = NodeDef(parametric, cyclic)  parameterize() -> Node

What the one contract buys:

1. Transforms are written ONCE against the contract: batch, ensemble,
   stack, scan, train_step, finetune are each a few dozen lines.
2. Closure: transforms and composition consume exactly what they
   produce (def in, def out), so batch(stack(x)), ensemble(a >> b) and
   train_step(pipe) compose freely.
3. Stable pytree identity: one instance class (Node), the def riding as
   aux data. Two bindings of one def have equal treedefs, so jit caches
   hit and optimizer states line up.
4. Composition has no special cases: a pipe's param and state are
   Structs over ALL members, trivial ()s included; those contribute no
   leaves and cost nothing, but keep every code path uniform.
5. train_step generalizes to stateful models for free: model state
   travels inside the trainer state, because every def has a state slot.

Bound calling conventions live on Node: a non-cyclic node is called
apply(input) -> output, a cyclic one apply(state, input) -> (state,
output), and the unbound def answers the same calls with the param
passed explicitly. Params, states and inputs are plain pytrees;
authored functions carry natural signatures, lifted by node_def().

Package layout (one file per concern):

    struct.py      Struct: the immutable record pytree everything speaks
    types.py       type vocabulary: pytree roles and the fn types
    core.py        NodeDef, Node — the only layer that executes
    authoring.py   node_def / derive: natural signatures -> contract form
    wiring.py      the `self` sugar for hand-wired composites
    compose.py     serial / parallel / composite; >> flattening
    spec.py        input specs: declare or resolve, derive the rest
    generic.py     GenericDef: the composable static stage
    ambient.py     dynamic scope for construction arguments
    transforms/    node transforms, one file each
    nn.py          stock neural blocks
    control.py     closed_loop / observed_loop
    actuator/      a motor-control showcase domain
    examples/      example nodes and end-to-end tests
"""

from nodejax.struct import Struct
from nodejax.types import (PyTree, Param, State, Input, Output,
                                 ParamFn, InitFn, ApplyFn, LossFn, StaticTree)
from nodejax.core import Node, NodeDef, split_aux, hoist_rng
from nodejax.authoring import node_def, derive, KeyStream
from nodejax.generic import GenericDef, generic
from nodejax.ambient import ambient
from nodejax.paths import replace_by_path, set_by_path
from nodejax.transforms import batch, ensemble, stack, repeat, scan, residual, train_step, finetune, metasgd, taps, tie, externalize, parallel, at, ttt, reconstruction, freeze, tree_freeze, detach, tree_detach, tree_filter, map_members
from nodejax.compose import serial, serial_generic, composite, composite_init, wrapper
from nodejax.control import closed_loop, observed_loop
from nodejax.spec import spec, spec_of, materialize, initialize, meta

__all__ = [
    'Struct',
    # types
    'PyTree', 'Param', 'State', 'Input', 'Output',
    'ParamFn', 'InitFn', 'ApplyFn', 'LossFn', 'StaticTree',
    # core
    'Node', 'NodeDef', 'split_aux',
    # authoring
    'node_def', 'derive', 'KeyStream',
    # static stage
    'GenericDef', 'generic',
    # transforms
    'batch', 'ensemble', 'stack', 'repeat', 'scan', 'residual', 'train_step',
    'finetune', 'metasgd', 'taps', 'externalize', 'parallel', 'at', 'ttt', 'reconstruction',
    'freeze', 'tree_freeze', 'detach', 'tree_detach', 'tree_filter', 'map_members', 'closed_loop', 'observed_loop', 'set_by_path',
    # composition
    'serial', 'serial_generic', 'composite', 'composite_init', 'wrapper',
    # spec layer
    'spec', 'spec_of', 'materialize', 'initialize', 'meta',
]
