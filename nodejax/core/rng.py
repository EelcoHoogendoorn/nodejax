"""Random-key streams for authored code and transform composition."""

from __future__ import annotations

from typing import Any

import jax


class _NoRng:
    def __repr__(self) -> str:
        return '<NO_RNG>'


_NO_RNG = _NoRng()


def _check_raw_key(key: jax.Array, where: str) -> None:
    """Require one unbatched JAX PRNG key at a public boundary."""
    try:
        data = jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{where} expects one raw JAX PRNG key') from exc
    if getattr(data, 'ndim', None) != 1:
        raise TypeError(f'{where} expects one raw JAX PRNG key, not a key axis')


class KeyStream:
    """The invocation-local mutable RNG stream passed to authored code."""

    def __init__(self, key: Any):
        self._key = key

    def __repr__(self) -> str:
        return 'KeyStream()'

    def next(self):
        """Draw one key and advance the stream."""
        self._key, key = jax.random.split(self._key)
        return key


class MaybeKeyStream:
    """A keyed-or-empty RNG capability used by compiled transform code.

    It keeps deterministic and stochastic canonical calls uniform. ``child``
    and ``axis`` allocate entropy only when the called role declares that it
    needs it. Ordinary authored functions never receive this object; a keyed
    capability is narrowed to :class:`KeyStream` at that boundary.
    """

    def __init__(self, key: Any = _NO_RNG):
        self._stream = (key if type(key) is KeyStream else
                        None if key is _NO_RNG else KeyStream(key))

    def __bool__(self) -> bool:
        return self._stream is not None

    def __repr__(self) -> str:
        return 'MaybeKeyStream(keyed)' if self else 'MaybeKeyStream(empty)'

    def _require(self) -> KeyStream:
        """Narrow a keyed capability for ordinary authored code."""
        if self._stream is None:
            raise TypeError('random key required, but this RNG is empty')
        return self._stream

    def next(self):
        """Draw one key, failing when this capability is empty."""
        return self._require().next()

    def child(self, takes_rng: bool) -> 'MaybeKeyStream':
        """Allocate the capability required by one child call."""
        if type(takes_rng) is not bool:
            raise TypeError('MaybeKeyStream.child takes_rng must be a bool')
        return (MaybeKeyStream(self.next()) if takes_rng
                else MaybeKeyStream())

    def split(self, count: int) -> tuple['MaybeKeyStream', int | None]:
        """Split this capability over an axis.

        Empty capabilities stay empty and broadcast because they have no JAX
        leaves. A keyed capability is consumed as the root of ``count``
        independent axis streams.
        """
        if type(count) is not int or count < 0:
            raise TypeError(
                'MaybeKeyStream.split count must be a non-negative int')
        if not self:
            return self.broadcast()
        return MaybeKeyStream(jax.random.split(self.next(), count)), 0

    def broadcast(self) -> tuple['MaybeKeyStream', None]:
        """Broadcast this capability unchanged over an axis."""
        return self, None

    def axis(self, takes_rng: bool, count: int | None, *,
             split: bool = True) -> tuple['MaybeKeyStream', int | None]:
        """Allocate one child's split or broadcast axis capability."""
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


def _reject_no_rng(tree, where: str) -> None:
    """Keep the private empty marker and RNG capabilities out of model data."""
    capabilities = (KeyStream, MaybeKeyStream)
    leaves = jax.tree.leaves(tree, is_leaf=lambda value:
                            value is _NO_RNG or type(value) in capabilities)
    if any(value is _NO_RNG or type(value) in capabilities for value in leaves):
        raise TypeError(f'{where}: an RNG capability escaped as model data')
