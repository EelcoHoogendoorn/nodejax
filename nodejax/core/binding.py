"""Carved from core.py: one file per class, the helpers beside their
class. The package __init__ is the public facade."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import jax

from nodejax.struct import Struct, register_struct_subtype
from nodejax.frozendict import frozendict
from nodejax.core.rng import MaybeKeyStream, _check_raw_key
from nodejax.core.types import Output


class _Required:
    """Marker for a required input-bundle field: a constructor parameter with
    no default, whose value the caller must supply. Its shape/dtype is the
    spec's concern; this is the requirement marker."""
    def __repr__(self) -> str:
        return 'REQUIRED'


REQUIRED = _Required()

_UNSET = object()      # a channel not offered to bind: absent, not None


@dataclass(frozen=True, eq=False)
class AxisSpec:
    """A leading axis over one ELEMENT: what an axis transform
    (batch, and the scan family in turn) publishes as its input spec.

    The element is a spec pytree, concrete leaves or a declared bundle;
    count is the axis's declared extent. A fixed mapping learns an omitted
    count from its first binding and checks it thereafter. A variable mapping
    never adopts a runtime extent; scans use that form because sequence length
    may change between calls. The element survives either way, which is the
    point: params build against it and bindings validate against it. Reserved
    RNG never sits under the map; its invocation capability is separate."""
    element: Any
    count: int | None = None
    fixed: bool = True

    def __post_init__(self) -> None:
        if self.count is not None and type(self.count) is not int:
            raise TypeError('AxisSpec.count must be an int or None')
        if self.count is not None and self.count < 0:
            raise ValueError('AxisSpec.count cannot be negative')
        if type(self.fixed) is not bool:
            raise TypeError('AxisSpec.fixed must be a bool')

    def replace(self, *, element: Any = _UNSET, count: Any = _UNSET,
                fixed: Any = _UNSET) -> 'AxisSpec':
        """Return a distinct declaration with the named fields replaced."""
        return type(self)(
            self.element if element is _UNSET else element,
            self.count if count is _UNSET else count,
            fixed=self.fixed if fixed is _UNSET else fixed,
        )

    def __repr__(self) -> str:
        n = ('*' if not self.fixed else
             '?' if self.count is None else self.count)
        return f'axis[{n}]({self.element!r})'

    def __eq__(self, other: Any) -> bool:
        return (type(other) is AxisSpec and other.count == self.count
                and other.element == self.element
                and other.fixed == self.fixed)


def _bundle_spec(parameters: Mapping[str, inspect.Parameter], *, drop: tuple = (),
                 owner: str | None = None,
                 allow_defaults: bool = True) -> Struct:
    """Build one ordered argument declaration from inspected parameters."""
    declaration = {}
    prefix = f'{owner}: ' if owner else ''
    for name, parameter in parameters.items():
        if name in drop:
            continue
        if name == 'bundle':
            raise TypeError(
                f"{prefix}'bundle' is reserved at the call boundary "
                '(apply(bundle=...) hands the input over formed); name the '
                'field for what it carries')
        if name == 'rng' and parameter.default is not inspect.Parameter.empty:
            raise TypeError(f'{prefix}rng never has a default; a stochastic '
                            'function requires its key')
        if (not allow_defaults
                and parameter.default is not inspect.Parameter.empty):
            raise TypeError(
                f"{prefix}input '{name}' cannot have a default; node inputs "
                'are always required')
        declaration[name] = (REQUIRED
                             if parameter.default is inspect.Parameter.empty
                             else parameter.default)
    return Struct(**declaration)


def _bundle_spec_from_sig(fn: Callable, *, drop: tuple = ()) -> Struct:
    """The required input fields of an authored apply function.

    Names in ``drop`` are implementation channels rather than inputs. The
    reserved ``rng`` marker remains only long enough for compilation to record
    the role's entropy requirement; it is never part of the data bundle.
    """
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return Struct()
    params = {
        name: parameter for name, parameter in params.items()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD)
    }
    return _bundle_spec(
        params, drop=drop, owner=fn.__name__, allow_defaults=False)


def _has_rng_deep(tree: Any) -> bool:
    """Whether a state VALUE carries a reserved rng field at any depth:
    the question a stored state answers before it may fill an init slot
    without a boundary key (a stored key never replays; it is replaced)."""
    if issubclass(type(tree), Struct):
        return 'rng' in tree or any(_has_rng_deep(v) for _, v in tree.__items__)
    return False


def _has_rng_field(tree: Any) -> bool:
    """Whether one Struct value has a top-level domain field named ``rng``."""
    return issubclass(type(tree), Struct) and 'rng' in tree


def _spec_resolved(spec: Any) -> bool:
    """Whether an apply input spec is RESOLVED (shapes known) rather than a
    signature-derived marker bundle awaiting binding. None is not resolved;
    a tree containing a REQUIRED marker is not resolved. A AxisSpec is
    resolved when its ELEMENT is: the count is not shape knowledge, and
    nothing that builds params needs it."""
    if spec is None:
        return False
    leaves = jax.tree.leaves(spec, is_leaf=lambda x: type(x) is AxisSpec)
    return all(_spec_resolved(leaf.element) if type(leaf) is AxisSpec
               else leaf is not REQUIRED for leaf in leaves)


def _contains_axis(spec: Any) -> bool:
    """Whether the spec declares any axis."""
    return any(type(leaf) is AxisSpec for leaf in jax.tree.leaves(
        spec, is_leaf=lambda x: type(x) is AxisSpec))


def _counts_unknown(spec: Any) -> bool:
    """Whether any fixed axis still lacks its extent."""
    axes = jax.tree.leaves(
        spec,
        is_leaf=lambda value: type(value) is AxisSpec,
    )
    return any(
        type(axis) is AxisSpec
        and (
            (axis.fixed and axis.count is None)
            or _counts_unknown(axis.element)
        )
        for axis in axes
    )


def _bind_axis(declared: Any, got: Any) -> Any:
    """The spec to store after binding `got` (already reduced to a spec)
    against `declared`: where declared maps an axis, got's leading axis
    becomes the count and the rest the element, so the stored spec keeps
    the declared FORM; everywhere else got stands. Validation is
    validate_input_spec's job; this only records."""
    if type(declared) is AxisSpec:
        if type(got) is AxisSpec:
            count = declared.count
            if declared.fixed and count is None:
                count = got.count
            return declared.replace(
                element=_bind_axis(declared.element, got.element),
                count=count,
            )
        leaves = jax.tree.leaves(got)
        count = leaves[0].shape[0] if leaves and leaves[0].shape else None
        element = jax.tree.map(
            lambda l: jax.ShapeDtypeStruct(l.shape[1:], l.dtype), got)
        bound_count = declared.count
        if declared.fixed and bound_count is None:
            bound_count = count
        return declared.replace(
            element=_bind_axis(declared.element, element),
            count=bound_count,
        )
    if (issubclass(type(declared), Struct)
            and issubclass(type(got), Struct)):
        values = {}
        for key in declared.__keys__:
            if key in got:
                values[key] = _bind_axis(declared[key], got[key])
            elif declared[key] is not REQUIRED:
                values[key] = declared[key]
        for key in got.__keys__:
            if key not in values:
                values[key] = got[key]
        out = Struct(**values)
        return out
    return got


def _spec_sig(tree: Any) -> tuple:
    """A pytree's structural shape signature (treedef + per-leaf shape and
    dtype), taken through spec_of so concrete and spec leaves compare
    alike. Axis declarations compare through one abstract representative,
    never by treating AxisSpec itself as an array. For validating one shape
    against another."""
    from nodejax.core.spec import _axis_probe_spec
    leaves, treedef = jax.tree.flatten(_axis_probe_spec(tree))
    return treedef, tuple((tuple(leaf.shape), leaf.dtype) for leaf in leaves)


def _validate_bundle(
        name: str,
        entry: str,
        form: CallForm,
        bundle: Struct,
        *,
        _boundary: bool = True,
) -> None:
    """One bundle against its declared spec: an unknown field and a missing
    REQUIRED field are both errors, named, at the call that supplied them.

    RNG is a separate framework channel at the call boundary. Complete fields
    remain ordinary domain data even when their value is a Struct; only fields
    explicitly declared as nested construction forms recurse."""
    if not issubclass(type(bundle), Struct):
        raise TypeError(
            f'{name}.{entry}: canonical bundles are Struct values; '
            f'got {type(bundle).__name__}')
    if _boundary and 'rng' in bundle:
        raise TypeError(f'{name}.{entry}: rng is a separate call channel')
    declared = set(form.fields.__keys__)
    given = set(bundle.__keys__)
    unknown = given - declared
    if unknown:
        raise TypeError(f'{name}.{entry}: unknown bundle fields '
                        f'{sorted(unknown)}; the bundle is '
                        f'{sorted(form.fields.__keys__)}')
    missing = [
        field_name for field_name, field in form.fields.__items__
        if (not field.is_nested and field.content is REQUIRED
            and field_name not in bundle)
    ]
    if missing:
        raise TypeError(f'{name}.{entry}: missing required bundle fields {missing}')
    for field_name, field in form.fields.__items__:
        if field.is_nested:
            supplied = (bundle[field_name]
                        if field_name in bundle else Struct())
            _validate_bundle(
                name, f'{entry}.{field_name}', field.content, supplied,
                _boundary=False)


def validate_param_input(node, param_input: Struct) -> None:
    """Check a construction bundle against param_input_spec.

    An absent spec means the node takes no construction input at all, so any
    field is an error. That is not the same as having no params: a composite
    has no construction input of its own and still builds a param, the Struct
    of its members'."""
    call = node._def.calls.param
    if call is None:
        if set(param_input.__keys__):
            raise TypeError(f'{node.name} is not parametric; its param bundle is empty')
        return
    _validate_bundle(node.name, 'contract.param', call.form, param_input)


def validate_state_input(node, state_input: Struct) -> None:
    """Check a state_input bundle against state_input_spec.

    An absent spec means the node has no state to build from, so any field is an
    error: it claims state that does not exist."""
    call = node._def.calls.init
    if call is None:
        if set(state_input.__keys__):
            raise TypeError(f'{node.name} is not cyclic; its state bundle is empty')
        return
    _validate_bundle(node.name, 'contract.init', call.form, state_input)


def validate_input_spec(definition, input_spec: Any) -> None:
    """Check a value against a definition's declared input spec.

    Bundles validate by the bundle rule, the same one every other bundle
    boundary applies: an unknown field is a NAMED conflict, a present field
    must match the spec's shape, and an absent field merely contributes no new
    shape evidence. Runtime requiredness is owned separately by CallForm;
    apply inputs never have defaults. Non-bundle trees compare whole."""
    declared = definition.calls.apply.input_spec
    _validate_spec(definition.name, declared, input_spec)


def _validate_spec(where: str, spec: Any, got: Any) -> None:
    """One spec subtree against what a binding supplies, recursing through
    bundles and declared axes so the error names the exact conflict.

    RNG is not represented here: public binding has already converted the raw
    key into a separate invocation capability."""
    if spec is REQUIRED:
        return
    if type(spec) is AxisSpec:
        if type(got) is AxisSpec:
            if (spec.fixed and spec.count is not None and got.count is not None
                    and spec.count != got.count):
                raise TypeError(f'{where}: axis of {got.count} conflicts '
                                f'with the declared count {spec.count}')
            _validate_spec(f'{where}.element', spec.element, got.element)
            return
        from nodejax.core.spec import spec_of
        gleaves = jax.tree.leaves(spec_of(got))
        if any(not l.shape for l in gleaves):
            raise TypeError(
                f'{where}: the input carries a leading axis and this '
                f'one has a scalar leaf, so there is no axis to map; the '
                f'element is {_spec_sig(spec.element)[1]}')
        counts = {l.shape[0] for l in gleaves}
        if len(counts) > 1:
            raise TypeError(f'{where}: unequal leading axes {sorted(counts)} '
                            'on one input carrying an axis')
        count = counts.pop() if counts else None
        if (spec.fixed and spec.count is not None and count is not None
                and count != spec.count):
            raise TypeError(f'{where}: axis of {count} conflicts with '
                            f'the declared count {spec.count}')
        element = jax.tree.map(
            lambda l: jax.ShapeDtypeStruct(l.shape[1:], l.dtype), spec_of(got))
        _validate_spec(f'{where}.element', spec.element, element)
        return
    if (issubclass(type(spec), Struct)
            and issubclass(type(got), Struct)):
        unknown = set(got.__keys__) - set(spec.__keys__)
        if unknown:
            raise TypeError(f"{where}: unknown input fields {sorted(unknown)}; "
                            f"the spec fields are {sorted(spec.__keys__)}")
        for k in got.__keys__:
            _validate_spec(f'{where}.{k}', spec[k], got[k])
        return
    if _spec_sig(spec) != _spec_sig(got):
        raise TypeError(
            f"{where}: input shape {_spec_sig(got)[1]} conflicts with its "
            f"declared spec {_spec_sig(spec)[1]}")


@register_struct_subtype
class Aux(Struct):
    """The auxiliary channel's marker: sown metrics, losses and taps.

    A named bundle like any other, which is why it subtypes Struct — and a
    distinct TYPE, because that is what split_aux dispatches on. The output
    doctrine spells positional pairs of data as named Structs, so a Struct
    in the second slot is data; only an Aux is aux."""


def split_aux(output: Output) -> tuple[Output, Any]:
    """The aux stream: clean output and auxiliary data separation.

    Returns (clean_output, aux).
    aux is non-None when output is a 2-tuple whose second element is an
    Aux — the explicit marker, and the only one. A Struct there is DATA: the
    output doctrine spells positional pairs of data as named Structs, so
    reading any Struct as aux would take a node's own output away from it.
    """
    if type(output) is tuple and len(output) == 2 and type(output[1]) is Aux:
        return output[0], output[1]
    return output, None


def drop_aux(output: Output) -> Output:
    """Discard Aux from a concrete runtime output."""
    clean, _ = split_aux(output)
    return clean


def _freeze_mapping(values: Mapping) -> frozendict:
    """Freeze a nested mapping after local assembly."""
    return frozendict({
        name: (_freeze_mapping(value)
               if issubclass(type(value), Mapping) else value)
        for name, value in values.items()
    })


def _unflatten_dot_paths(statics: dict) -> frozendict:
    """'member.field' keys nest into member dicts; '*.'-wilds are the
    caller's to strip first."""
    out: dict = {}
    for key, value in statics.items():
        if '.' in key and not key.startswith('*.'):
            parts = key.split('.')
            curr = out
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = value
        elif (issubclass(type(value), Mapping)
              and issubclass(type(out.get(key)), Mapping)):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return _freeze_mapping(out)


def _as_bundle(fields: dict) -> Any:
    """The param/state-boundary sugar: loose kwargs pack into a bundle,
    and a dot-joined key configures by path: parameterize(**{'gain.scale':
    1.0}) nests exactly as parameterize(gain=Struct(scale=1.0)) does, so
    the flat and the nested spelling are the same call. Sub-bundles are
    Structs."""
    def as_struct(tree: Any) -> Any:
        if issubclass(type(tree), Mapping):
            return Struct(**{name: as_struct(value)
                             for name, value in tree.items()})
        return tree

    if any('.' in name for name in fields):
        return as_struct(_unflatten_dot_paths(dict(fields)))
    return Struct(**fields)


def _bind_call(node, args: tuple, kwargs: dict) -> Any:
    """THE calling convention: a node's input signature is a function
    signature. Positional args bind to the declared fields in order,
    kwargs by name; what travels underneath is the input bundle. For a
    closed call, explicit ``bundle=`` selects the declared fields from a
    formed source bundle. An open call receives that bundle whole. Ordinary
    positional and keyword calls remain exact."""
    kwargs = dict(kwargs)
    if 'bundle' in kwargs:
        formed = kwargs.pop('bundle')
        if args or kwargs:
            raise TypeError(f'{node.name}: bundle= is the whole input')
        if type(formed) is dict:
            formed = Struct(**formed)
        if not issubclass(type(formed), Struct):
            raise TypeError(f'{node.name}: bundle= must be a Struct or dict')
        if 'rng' in formed:
            raise TypeError(
                f'{node.name}: pass rng= beside bundle=, not inside it')
        if not node.contract._apply_form.open:
            declaration = node.contract._apply_form.declaration
            selected = {
                name: formed[name]
                for name in declaration.__keys__
                if name in formed
            }
            formed = Struct(**selected)
        return _validate_public_input(node, formed)
    declaration = node.contract._apply_form.declaration
    if node.contract._apply_form.open and not declaration.__keys__:
        if args:
            raise TypeError(
                f'{node.name} declares no fixed input fields; '
                'pass keyword fields or bundle=')
        return _validate_public_input(node, Struct(**kwargs))
    fields = node.contract.apply_fields
    if len(args) > len(fields):
        raise TypeError(f'{node.name} takes {len(fields)} positional input '
                        f'field(s) {fields}; got {len(args)}')
    filled = dict(zip(fields, args))
    twice = set(filled) & set(kwargs)
    if twice:
        raise TypeError(f'{node.name}: {sorted(twice)} given positionally '
                        'and by keyword')
    filled.update(kwargs)
    ordered = {f: filled[f] for f in fields if f in filled}
    ordered.update({f: value for f, value in filled.items() if f not in ordered})
    return _validate_public_input(node, Struct(**ordered))


def _bind_rng(takes_rng: bool, key: Any, where: str) -> MaybeKeyStream:
    """Bind a raw key or adapt a forwarded invocation capability.

    Raw public keys are strict: a stochastic role requires exactly one and a
    deterministic role rejects one. A forwarded ``MaybeKeyStream`` is
    compositional: deterministic children receive an empty capability without
    advancing the parent, while stochastic children consume one parent draw.
    """
    if type(takes_rng) is not bool:
        raise TypeError(f'{where}: takes_rng must be a bool')
    if type(key) is MaybeKeyStream:
        try:
            return key.child(takes_rng)
        except TypeError as exc:
            raise TypeError(f'{where}: {exc}') from exc
    supplied = key is not _UNSET
    if takes_rng and not supplied:
        raise TypeError(f'{where} requires rng=')
    if not takes_rng and supplied:
        raise TypeError(f'{where} does not accept rng=')
    if not supplied:
        return MaybeKeyStream()
    _check_raw_key(key, where)
    return MaybeKeyStream(key)


def _bind_public_call(node, args: tuple,
                      kwargs: dict) -> tuple[Struct, MaybeKeyStream]:
    """Bind one root apply call and its separate RNG execution channel.

    Authored member calls use the same argument binding before their enclosing
    wiring supplies a child capability. Only a public call accepts a raw key.
    """
    kwargs = dict(kwargs)
    key = kwargs.pop('rng', _UNSET)
    bundle = _bind_call(node, args, kwargs)
    rng = _bind_rng(node.contract.apply_takes_rng, key,
                    f'{node.name}: apply')
    return bundle, rng


def _validate_public_input(node, bundle: Struct) -> Struct:
    """Validate one formed public input against its stable call form.

    This is the sole validation boundary for positional, keyword, and
    explicit ``bundle=`` calls. A closed explicit bundle has already selected
    the declaration's fields before reaching this boundary. Internal
    composition already produces formed bundles and enters through the
    compiled contract instead. Resolved value specs are construction and
    transform metadata, not a restriction on the outer shape of every later
    runtime value; resolution validates those shapes at ``with_input`` and
    internal binding sites.
    """
    if 'rng' in bundle:
        raise TypeError(f'{node.name}: rng is a separate call channel')
    declaration = node.contract._apply_form.declaration
    fields = set(bundle.__keys__)
    if not node.contract._apply_form.open:
        unknown = fields - set(declaration.__keys__)
        if unknown:
            raise TypeError(f'{node.name}: unknown input fields {sorted(unknown)}; '
                            f'declared: {tuple(declaration.__keys__)}')
    missing = [name for name in declaration.__keys__
               if declaration[name] is REQUIRED and name not in fields]
    if missing:
        raise TypeError(f'{node.name}: missing required input fields {missing}')
    return bundle


# names that are CHANNELS in a method signature, injected by the view
# that binds the method
_METHOD_CHANNELS = frozenset({'param', 'state', 'node', 'rng'})


def _bind_method(fn: Callable, *, node = None,
                 param = None, state = None,
                 rng = None) -> Callable:
    """Bind a node method to a view: the reserved parameters (node, param,
    state, rng) fill from the view at call time, every other parameter is
    a call argument, and an explicit keyword beats the view.

    Each keyword here is a zero-arg callable, evaluated per call rather
    than captured, so a state read after a member step sees the advance."""
    names = list(inspect.signature(fn).parameters)
    fields = [n for n in names if n not in _METHOD_CHANNELS]
    reserved = [n for n in names if n in _METHOD_CHANNELS]

    def bound(*args: Any, **kwargs: Any) -> Any:
        if len(args) > len(fields):
            raise TypeError(f'{fn.__name__}() takes {len(fields)} call argument(s) '
                            f'{fields}; the reserved names {reserved} are injected')
        kw = dict(zip(fields, args))
        doubled = kw.keys() & kwargs.keys()
        if doubled:
            raise TypeError(f'{fn.__name__}() got multiple values for {sorted(doubled)}')
        kw.update(kwargs)
        if 'rng' in kw:
            kw['rng'] = _bind_rng(
                True, kw['rng'], f'{fn.__name__}: rng')._require()
        for nm, get in dict(node=node, param=param, state=state, rng=rng).items():
            if nm in reserved and nm not in kw and get is not None:
                value = get()
                kw[nm] = (value._require()
                          if nm == 'rng' and type(value) is MaybeKeyStream
                          else value)
        return fn(**kw)

    return bound


# === the form ===
