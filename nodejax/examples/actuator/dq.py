"""DQ — the rotor-frame value pair, as a domain-level arithmetic type.
Elementwise operators mean one controller, one
estimator, one set of motor equations run both axes: `i * R + di_dt * L`
reads like the physics. A registered (keyed) pytree, so DQ flows through
state, params, scan stacking and grad like any other value.

Deliberately domain-level: nodejax's Struct stays an inert record; the
arithmetic lives on the type that earns it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class DQ:
    __slots__ = ('d', 'q')

    def __init__(self, d=0.0, q=0.0):
        self.d = d
        self.q = q

    def _zip(self, other, op):
        if isinstance(other, DQ):
            return DQ(op(self.d, other.d), op(self.q, other.q))
        return DQ(op(self.d, other), op(self.q, other))

    def __add__(self, other):
        return self._zip(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._zip(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._zip(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._zip(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._zip(other, lambda a, b: a / b)

    def __neg__(self):
        return DQ(-self.d, -self.q)

    def norm2(self):
        return self.d ** 2 + self.q ** 2

    def norm(self):
        return jnp.sqrt(self.norm2())

    def clamp_norm(self, limit):
        """Scale down to the limit magnitude, direction preserved."""
        return self / jnp.maximum(1.0, self.norm() / limit)

    def __repr__(self):
        return f'DQ(d={self.d!r}, q={self.q!r})'


def _flatten_with_keys(t: float):
    return ((jax.tree_util.GetAttrKey('d'), t.d),
            (jax.tree_util.GetAttrKey('q'), t.q)), None


jax.tree_util.register_pytree_with_keys(
    DQ, _flatten_with_keys, lambda _, c: DQ(*c), lambda t: ((t.d, t.q), None))
