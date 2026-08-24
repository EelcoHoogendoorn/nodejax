"""Python 3.15 ``frozendict`` with a fallback for supported older Pythons."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


try:
    from builtins import frozendict
except ImportError:
    class frozendict(Mapping):
        """An insertion-ordered immutable mapping."""

        __slots__ = ('__values',)

        def __init__(self, source=(), /, **values: Any) -> None:
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

        def copy(self) -> 'frozendict':
            return self

        @classmethod
        def fromkeys(cls, keys, value=None) -> 'frozendict':
            return cls(dict.fromkeys(keys, value))

        def __reduce__(self):
            return type(self), (dict(self),)


__all__ = ('frozendict',)
