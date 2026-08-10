"""Type vocabulary for the core.

All four data roles are plain pytrees; the aliases document intent, not
structure. The three function types state the contract once.
"""

from __future__ import annotations

from typing import Any, Callable

import jax

from nodejax.struct import Struct

PyTree = Any
Param = PyTree
State = PyTree
Input = PyTree
Output = PyTree

ParamFn = Callable[..., Param]                            # (*args, **kwargs) -> param
InitFn = Callable[..., State]                             # (param, *args, **kwargs) -> state
ApplyFn = Callable[[Param, State, Input], tuple[State, Output]]
LossFn = Callable[[Output, PyTree], jax.Array]            # (output, target) -> scalar loss

StaticTree = dict[str, Any]  # nested static kwargs, keyed by pipe member name

