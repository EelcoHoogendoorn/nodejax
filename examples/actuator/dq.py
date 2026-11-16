"""Defines `DQ`, a JAX-registered rotor-frame `(d, q)` arithmetic pair."""

from __future__ import annotations

import jax
import jax.numpy as jnp


class DQ:
    """Represents a direct-quadrature `(d, q)` vector with elementwise arithmetic and smooth norm clamping."""

    __slots__ = ('d', 'q')

    def __init__(self, d=0.0, q=0.0) -> None:
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

    def __neg__(self) -> DQ:
        return DQ(-self.d, -self.q)

    def norm2(self):
        return self.d ** 2 + self.q ** 2

    def norm(self):
        """Computes `sqrt(max(d^2 + q^2, 1e-24))` so gradients remain finite at `(0, 0)`."""
        return jnp.sqrt(jnp.maximum(self.norm2(), 1e-24))

    def clamp_norm(self, limit) -> DQ:
        """Scales `(d, q)` down when `norm()` exceeds `limit`, preserving direction."""
        return self / jnp.maximum(1.0, self.norm() / limit)

    def __repr__(self) -> str:
        return f'DQ(d={self.d!r}, q={self.q!r})'


def _flatten_with_keys(pair: DQ):
    return ((jax.tree_util.GetAttrKey('d'), pair.d),
            (jax.tree_util.GetAttrKey('q'), pair.q)), None


jax.tree_util.register_pytree_with_keys(
    DQ, _flatten_with_keys, lambda _, c: DQ(*c), lambda pair: ((pair.d, pair.q), None))
