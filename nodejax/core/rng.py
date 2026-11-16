"""Provides invocation-local PRNG streams for authored Nodes and T3 transforms.

Authored `param`, `init`, and `apply` functions receive `KeyStream` and draw subkeys via `rng.next()`. T3 transform bodies carry `MaybeKeyStream`, a JAX PyTree that unifies deterministic (`empty`) and stochastic (`keyed`) execution so `child`, `split`, `broadcast`, and `axis` allocate subkeys only when the wrapped `Contract` role declares `takes_rng=True`.
"""

from __future__ import annotations

import jax

from nodejax.core.types import PyTree


class _NoRng:
    def __repr__(self) -> str:
        return '<NO_RNG>'


_NO_RNG = _NoRng()


def _check_raw_key(key: jax.Array, where: str) -> None:
    try:
        data = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{where} expects one raw JAX PRNG key') from exc
    if getattr(data, 'ndim', None) != 1:
        raise TypeError(f'{where} expects one raw JAX PRNG key, not a key axis')


class KeyStream:
    """Supplies sequential PRNG subkeys within one `param`, `init`, or `apply` call."""

    def __init__(self, key: jax.Array):
        self._key = key

    def __repr__(self) -> str:
        return 'KeyStream()'

    def next(self) -> jax.Array:
        """Splits the internal key, advances the stream, and returns one fresh subkey."""
        self._key, key = jax.random.split(self._key)
        return key


class MaybeKeyStream:
    """Wraps an optional `KeyStream` as a JAX PyTree for uniform transform plumbing."""

    def __init__(self, key=_NO_RNG):
        self._stream = (key if type(key) is KeyStream else
                        None if key is _NO_RNG else KeyStream(key))

    def __bool__(self) -> bool:
        return self._stream is not None

    def __repr__(self) -> str:
        return 'MaybeKeyStream(keyed)' if self else 'MaybeKeyStream(empty)'

    def _require(self) -> KeyStream:
        if self._stream is None:
            raise TypeError('random key required, but this RNG is empty')
        return self._stream

    def next(self) -> jax.Array:
        """Draws one subkey, raising `TypeError` when this stream is empty."""
        return self._require().next()

    def child(self, takes_rng: bool) -> MaybeKeyStream:
        """Returns a fresh keyed stream when `takes_rng` is `True`, or an empty stream otherwise."""
        if type(takes_rng) is not bool:
            raise TypeError('MaybeKeyStream.child takes_rng must be a bool')
        return (MaybeKeyStream(self.next()) if takes_rng
                else MaybeKeyStream())

    def split(self, count: int) -> tuple[MaybeKeyStream, int | None]:
        """Splits a keyed stream into `count` axis-mapped streams `(stream, 0)`, or broadcasts `(empty, None)`."""
        if type(count) is not int or count < 0:
            raise TypeError(
                'MaybeKeyStream.split count must be a non-negative int')
        if not self:
            return self.broadcast()
        return MaybeKeyStream(jax.random.split(self.next(), count)), 0

    def broadcast(self) -> tuple[MaybeKeyStream, None]:
        """Pairs this stream with `in_axes=None` so `vmap` broadcasts it unchanged."""
        return self, None

    def axis(self, takes_rng: bool, count: int | None, *,
             split: bool = True) -> tuple[MaybeKeyStream, int | None]:
        """Returns `(stream, in_axis)` for a mapped child role according to `takes_rng` and `split`."""
        if type(takes_rng) is not bool:
            raise TypeError('MaybeKeyStream.axis takes_rng must be a bool')
        if type(split) is not bool:
            raise TypeError('MaybeKeyStream.axis split must be a bool')
        if not takes_rng:
            return MaybeKeyStream().broadcast()
        if not split:
            return self.child(True).broadcast()
        if count is None:
            raise TypeError('a split RNG axis requires a known count')
        return self.split(count)


def _rng_flatten(rng: MaybeKeyStream):
    return ((rng._stream._key,), True) if rng else ((), False)


def _rng_unflatten(keyed: bool, children) -> MaybeKeyStream:
    return MaybeKeyStream(children[0]) if keyed else MaybeKeyStream()


jax.tree_util.register_pytree_node(
    MaybeKeyStream, _rng_flatten, _rng_unflatten)


def _reject_no_rng(tree: PyTree, where: str) -> None:
    capabilities = (KeyStream, MaybeKeyStream)
    leaves = jax.tree.leaves(tree, is_leaf=lambda value:
                            value is _NO_RNG or type(value) in capabilities)
    if any(value is _NO_RNG or type(value) in capabilities for value in leaves):
        raise TypeError(f'{where}: an RNG capability escaped as model data')
