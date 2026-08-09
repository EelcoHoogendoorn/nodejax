"""

Key design goals:
- Lightweight container for return types  
- Suitable base class for Nodes
- No tuple subclassing (avoids method shadowing issues)
- Hidden __keys__ __values__
- Minimal attrdict + tuple-like interfaces
- JAX pytree registration
"""

from typing import Any, Iterator


class Struct:
	"""
	This is similar to a AnonymousNamedTuple with getitem access
	or perhaps a FrozenAttrOrderedDict
		that being said; only a part of the dict interface is implemented,
		(no keys, values or items)
		since implementing the dict interface would require showing these non-dunder methods
	"""

	def __init__(self, **kwargs):
		# Store fields in hidden attribute as frozendict-like structure
		object.__setattr__(self, '__keys__', tuple(kwargs.keys()))
		object.__setattr__(self, '__values__', tuple(kwargs.values()))

	def __setattr__(self, name: str, value: Any):
		"""Prevent attribute modification after construction."""
		raise AttributeError(f"'{self.__class__.__name__}' object attribute '{name}' is read-only")

	def __delattr__(self, name: str):
		"""Prevent attribute deletion."""
		raise AttributeError(f"'{self.__class__.__name__}' object attribute '{name}' is read-only")

	# === Attribute access interface ===

	def __getattribute__(self, name: str) -> Any:
		"""Prioritize field access over methods."""
		keys = object.__getattribute__(self, '__keys__')
		if name in keys:
			values = object.__getattribute__(self, '__values__')
			return values[keys.index(name)]
		return object.__getattribute__(self, name)

	def __getitem__(self, key) -> Any:
		"""Support both string keys and integer indices."""
		if key in self.__keys__:
			key = self.__keys__.index(key)
		if isinstance(key, int):
			return self.__values__[key]
		raise KeyError(f"'{self.__class__.__name__}' object key '{key}' not found.")

	def __contains__(self, key) -> bool:
		"""Support 'in' operator for field names."""
		return key in self.__keys__

	def __len__(self) -> int:
		"""Number of fields."""
		return len(self.__keys__)

	def __iter__(self) -> Iterator[Any]:
		"""Iterate over field values (enables unpacking). Note, giving precedence to tuple interface over dict interface"""
		return iter(self.__values__)

	# === String representation ===

	def __repr__(self) -> str:
		"""Compact representation."""
		fields = ', '.join(f'{k}={repr(v)}' for k, v in self.__as_dict__.items())
		return f'Struct2({fields})'

	def __str__(self) -> str:
		"""Same as repr for now."""
		return self.__repr__()

	@property
	def __items__(self) -> Iterator[tuple[str, Any]]:
		"""Iterate over (field_name, field_value) pairs."""
		return zip(self.__keys__, self.__values__)
	@property
	def __as_tuple__(self):
		return tuple(self.__values__)
	@property
	def __as_list__(self):
		return list(self.__values__)
	@property
	def __as_dict__(self):
		return dict(self.__items__)

	def without(self, *names: str) -> 'Struct':
		"""A copy without the named fields. Filter semantics: a name that is
		not present is simply not there to remove."""
		return Struct(**{k: v for k, v in self.__items__ if k not in names})

	def replace(self, *args, **kwargs) -> 'Struct':
		"""Create new Struct with updated fields, supporting nested dict-like updates."""
		# Convert positional args to field updates by zipping with our keys
		if len(args) > len(self.__keys__):
			raise TypeError(f"replace() takes at most {len(self.__keys__)} positional arguments ({len(args)} given)")

		# Build uniform update dict: positional args mapped to field names + kwargs
		updates = dict(zip(self.__keys__, args))
		updates.update(kwargs)

		return self._merge_nested(updates, preserve_structure=False)

	def merge(self, other) -> 'Struct':
		"""Merge another struct-like object, with other's fields taking precedence."""
		return self._merge_nested(other, preserve_structure=True)

	def _merge_nested(self, updates: dict, preserve_structure=True) -> 'Struct':
		"""Recursively merge nested dict-like updates."""
		result = self.__as_dict__
		if isinstance(updates, Struct):
			updates = updates.__as_dict__
		if not isinstance(updates, dict):
			raise TypeError(f"merges to Struct must be a dict-like")

		for key, value in updates.items():
			if key in result:
				rvalue = result[key]
				if isinstance(rvalue, Struct):
					if preserve_structure and not isinstance(value, (Struct, dict)):
						raise TypeError(f"Attempting to merge incompatible structures")
					value = rvalue._merge_nested(value)
				else:
					if preserve_structure and isinstance(value, (Struct, dict)):
						raise TypeError(f"Attempting to merge incompatible structures")
			result[key] = value

		return Struct(**result)


class Aux(Struct):
	"""Explicit auxiliary data container for sown metrics, losses, and taps."""
	pass


# === JAX PyTree registration ===

def _struct_flatten(tree):
	"""Flatten Struct for JAX pytree."""
	return tree.__values__, tree.__keys__

def _struct_flatten_with_keys(tree):
	"""Keyed flatten: field names become pytree path entries, so path-aware
	tooling (tree_map_with_path, optax.multi_transform, freeze/decay masks)
	sees '.member.field' paths instead of anonymous indices."""
	keys = tree.__keys__
	return tuple((jax.tree_util.GetAttrKey(k), v) for k, v in zip(keys, tree.__values__)), keys

def _struct_unflatten(aux_data, children):
	"""Unflatten JAX pytree back to Struct."""
	field_names = aux_data
	field_values = children
	return Struct(**dict(zip(field_names, field_values)))

# Register with JAX
try:
	import jax
	jax.tree_util.register_pytree_with_keys(
		Struct, _struct_flatten_with_keys, _struct_unflatten, _struct_flatten)
	def _aux_unflatten(aux_data, children):
		return Aux(**dict(zip(aux_data, children)))
	jax.tree_util.register_pytree_with_keys(
		Aux, _struct_flatten_with_keys, _aux_unflatten, _struct_flatten)
except ImportError:
	pass  # JAX not available
