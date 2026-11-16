"""Immutable named record registered as a keyed JAX PyTree.

`Struct` stores named fields with attribute, key, and positional-index access while keeping non-dunder dict methods (`keys`, `values`, `items`) off the namespace so arbitrary field names never collide with container methods. JAX PyTree flattening emits `GetAttrKey` path entries so path-addressed tools (`replace_by_path`, `set_by_path`, `optax.multi_transform`) see dotted `.member.field` addresses.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
	from nodejax.core.types import PyTree


class Struct:
	"""Holds named read-only fields as an ordered record and keyed JAX PyTree."""

	def __init__(self, **kwargs: PyTree) -> None:
		object.__setattr__(self, '__keys__', tuple(kwargs.keys()))
		object.__setattr__(self, '__values__', tuple(kwargs.values()))

	def __setattr__(self, name: str, value: PyTree) -> None:
		raise AttributeError(f"'{self.__class__.__name__}' object attribute '{name}' is read-only")

	def __delattr__(self, name: str) -> None:
		raise AttributeError(f"'{self.__class__.__name__}' object attribute '{name}' is read-only")

	def __reduce__(self):
		return _rebuild, (type(self), tuple(self.__items__))

	def __getattribute__(self, name: str) -> PyTree:
		keys = object.__getattribute__(self, '__keys__')
		if name in keys:
			values = object.__getattribute__(self, '__values__')
			return values[keys.index(name)]
		return object.__getattribute__(self, name)

	def __getitem__(self, key: str | int) -> PyTree:
		if key in self.__keys__:
			key = self.__keys__.index(key)
		if type(key) is int:
			return self.__values__[key]
		raise KeyError(f"'{self.__class__.__name__}' object key '{key}' not found.")

	def __contains__(self, key: str) -> bool:
		return key in self.__keys__

	def __len__(self) -> int:
		return len(self.__keys__)

	def __iter__(self) -> Iterator[PyTree]:
		return iter(self.__values__)

	def __repr__(self) -> str:
		fields = ', '.join(f'{k}={repr(v)}' for k, v in self.__as_dict__.items())
		return f'{type(self).__name__}({fields})'

	def __str__(self) -> str:
		return self.__repr__()

	@property
	def __items__(self) -> Iterator[tuple[str, PyTree]]:
		return zip(self.__keys__, self.__values__)
	@property
	def __as_tuple__(self) -> tuple[PyTree, ...]:
		return tuple(self.__values__)
	@property
	def __as_list__(self) -> list[PyTree]:
		return list(self.__values__)
	@property
	def __as_dict__(self) -> dict[str, PyTree]:
		return dict(self.__items__)

	def without(self, *names: str) -> Struct:
		"""Returns a new record omitting any fields present in `names`."""
		return type(self)(**{k: v for k, v in self.__items__ if k not in names})

	def replace(self, *args: PyTree, **kwargs: PyTree) -> Struct:
		"""Returns a new record with positional or keyword fields replaced."""
		if len(args) > len(self.__keys__):
			raise TypeError(f"replace() takes at most {len(self.__keys__)} positional arguments ({len(args)} given)")
		positional = dict(zip(self.__keys__, args))
		return type(self)(**{
			**self.__as_dict__, **positional, **kwargs,
		})

	def merge(self, other: Struct | dict) -> Struct:
		"""Recursively merges `other` (`Struct` or `dict`), giving leaves in `other` precedence."""
		return self._merge_nested(other, preserve_structure=True)

	def _merge_nested(self, other: Struct | dict, preserve_structure: bool = True) -> Struct:
		result = self.__as_dict__
		if issubclass(type(other), Struct):
			other = other.__as_dict__
		if not issubclass(type(other), dict):
			raise TypeError(f"merges to Struct must be a dict-like")

		for key, value in other.items():
			if key in result:
				current = result[key]
				current_is_struct = issubclass(type(current), Struct)
				value_is_struct = issubclass(type(value), Struct)
				value_is_dict = issubclass(type(value), dict)
				value_is_structure = value_is_struct or value_is_dict

				if current_is_struct and value_is_structure:
					value = current._merge_nested(
						value, preserve_structure=preserve_structure)
				elif preserve_structure and current_is_struct != value_is_structure:
					raise TypeError(f"Attempting to merge incompatible structures")
			result[key] = value

		return type(self)(**result)


def _rebuild(cls: type, items: tuple) -> Struct:
	return cls(**dict(items))


def _struct_flatten(tree):
	return tree.__values__, tree.__keys__

def _struct_flatten_with_keys(tree):
	keys = tree.__keys__
	return tuple((jax.tree_util.GetAttrKey(k), v) for k, v in zip(keys, tree.__values__)), keys

def _struct_unflatten(aux_data, children):
	field_names = aux_data
	field_values = children
	return Struct(**dict(zip(field_names, field_values)))

try:
	import jax
	jax.tree_util.register_pytree_with_keys(
		Struct, _struct_flatten_with_keys, _struct_unflatten, _struct_flatten)
except ImportError:
	pass


def register_struct_subtype(cls: type) -> type:
	"""Registers `cls` as its own keyed JAX PyTree so unflatten preserves the subclass."""
	import jax

	def unflatten(keys, values):
		return cls(**dict(zip(keys, values)))

	jax.tree_util.register_pytree_with_keys(
		cls, _struct_flatten_with_keys, unflatten, _struct_flatten)
	return cls
