"""Defines core type aliases for PyTree roles, lifecycle callables, and static dictionaries.

`Param`, `State`, `Input`, and `Output` are semantic aliases for `PyTree` that record which role an argument or return value occupies in the Node contract.
"""

from __future__ import annotations

from typing import Any, Callable

import jax

PyTree = Any
Param = PyTree
State = PyTree
Input = PyTree
Output = PyTree

ParamFn = Callable[..., Param]
InitFn = Callable[..., State]
ApplyFn = Callable[[Param, State, Input], tuple[State, Output]]
LossFn = Callable[..., jax.Array]

StaticTree = dict[str, Any]
