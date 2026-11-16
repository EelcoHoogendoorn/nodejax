"""Re-exports Python 3.15 `frozendict` and provides a pure-Python fallback on older runtimes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodejax.core.types import PyTree


try:
    from builtins import frozendict
except ImportError:
    class frozendict(Mapping):
        """Stores key-value pairs in an immutable insertion-ordered mapping."""

        __slots__ = ('__values',)

        def __init__(self, source=(), /, **values: PyTree) -> None:
            self.__values = dict(source, **values)

        def __getitem__(self, key):
            return self.__values[key]

        def __iter__(self) -> Iterator:
            return iter(self.__values)

        def __len__(self) -> int:
            return len(self.__values)

        def __repr__(self) -> str:
            return f'frozendict({self.__values!r})'

        def __hash__(self) -> int:
            return hash(frozenset(self.__values.items()))

        def __or__(self, other):
            values = dict(self)
            values.update(other)
            return type(self)(values)

        def __ror__(self, other):
            values = dict(other)
            values.update(self)
            return type(self)(values)

        def copy(self) -> frozendict:
            return self

        @classmethod
        def fromkeys(cls, keys, value=None) -> frozendict:
            return cls(dict.fromkeys(keys, value))

        def __reduce__(self):
            return type(self), (dict(self),)


__all__ = ('frozendict',)
