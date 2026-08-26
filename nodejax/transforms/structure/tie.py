"""Path-based parameter sharing.

`tie` is not a fully compositional Node transform. It rewrites an assembled
member tree, then reconstructs prior sharing and repairs the sparse parameter
layout itself. Parameter identity should instead be declared as definition
data. It remains open whether sharing should then survive every structural
transform or be consumed once at an explicit lowering boundary.
"""

from __future__ import annotations

from nodejax.core.ambient import node
from nodejax.core.compose import _probe_rng, _step_spec, param_call
from nodejax.core.contract import ContractCalls
from nodejax.core.definition import Captures, Def, Layout
from nodejax.frozendict import frozendict
from nodejax.core.generic import Generic
from nodejax.core.node import BaseNode, Node
from nodejax.core.spec import materialize
from nodejax.struct import Struct


def _parameter_sources(definition: Def) -> frozendict:
    """Map aliases in nested tie definitions to their active param member."""
    if definition.layout.kind != 'tie':
        return frozendict()
    arguments = definition.construction.arguments
    sources = dict(_parameter_sources(arguments.pipe._def))
    source = sources.get(arguments.source, arguments.source)
    for alias in arguments.aliases:
        sources[alias] = source
    for alias in tuple(sources):
        target = sources[alias]
        seen = set()
        while target in sources and target not in seen:
            seen.add(target)
            target = sources[target]
        sources[alias] = target
    return frozendict(sources)


@node
def _tie(pipe: Node, source: str, aliases: tuple[str, ...]) -> Node:
    if not pipe.members:
        raise TypeError(f"tie rewires members and '{pipe.name}' has none")
    missing = ({source} | set(aliases)) - set(pipe.members.__keys__)
    if missing:
        raise TypeError(f'tie: unknown members {sorted(missing)}')

    parameter_sources = dict(_parameter_sources(pipe._def))
    shared_source = parameter_sources.get(source, source)
    for alias in aliases:
        parameter_sources[alias] = shared_source
    for alias in tuple(parameter_sources):
        target = parameter_sources[alias]
        seen = set()
        while target in parameter_sources and target not in seen:
            seen.add(target)
            target = parameter_sources[target]
        parameter_sources[alias] = target

    def expand(param):
        return param.replace(**{
            alias: getattr(param, shared_source) for alias in aliases})

    inherited_param_members = pipe._def.layout.param_members
    active_param_members = frozenset(
        name for name, member in pipe.members.__items__
        if member.parametric
        and name not in aliases
        and (inherited_param_members is None
             or name in inherited_param_members))
    active_captures = Captures(
        param=frozendict({
            name: value
            for name, value in pipe._def.captures.param.items()
            if name not in aliases
        }),
        state=pipe._def.captures.state,
    )
    param_takes_rng = any(
        name not in active_captures.param
        and member.contract.param_takes_rng
        for name, member in pipe.members.__items__
        if name in active_param_members)

    def current_pipe(definition):
        return pipe._def.copy(
            members=definition.members,
            captures=definition.captures)

    def for_input(member, input_spec):
        if input_spec is None:
            return member
        return member.contract._resolve_def(input_spec)

    def param_impl(definition, formed_input, rng):
        supplied_aliases = set(aliases) & set(formed_input.__keys__)
        if supplied_aliases:
            raise TypeError(
                f'tied members {sorted(supplied_aliases)} share {source!r}')
        current = current_pipe(definition)
        spec = definition.contract.input_spec
        if spec is not None:
            spec = definition.contract.intake(spec)
        values = {}

        def build_param(name, input_spec):
            owner = parameter_sources.get(name, name)
            if owner in values:
                return values[owner]
            if owner not in active_param_members:
                return ()
            member = getattr(current.members, owner)
            if owner in definition.captures.param:
                values[owner] = formed_input[owner]
            else:
                bundle = (formed_input[owner]
                          if owner in formed_input else Struct())
                resolved = for_input(member, input_spec)
                child = rng.child(resolved.contract.param_takes_rng)
                values[owner] = resolved.contract.param(bundle, child)
            return values[owner]

        for name, member in current.members.__items__:
            walk_param = build_param(name, spec)
            if spec is not None:
                resolved = for_input(member, spec)
                child = _probe_rng(resolved.contract.init_takes_rng)
                state = (resolved.contract.prime(
                    walk_param, Struct(), materialize(spec), child)
                    if resolved.contract.init_requires_input else
                    resolved.contract.init(walk_param, Struct(), child))
                spec = _step_spec(
                    resolved, walk_param, state, spec, name)
        return Struct(**values)

    if pipe.contract.init_requires_input:
        def init_impl(definition, param, formed_input, input, rng):
            return current_pipe(definition).contract.prime(
                expand(param), formed_input, input, rng)
    else:
        def init_impl(definition, param, formed_input, rng):
            current = current_pipe(definition)
            shape = definition.contract.input_spec
            resolved = (current if shape is None else
                        current.contract._resolve_def(
                            shape, bundled=True))
            return resolved.contract.init(
                expand(param), formed_input, rng)

    def apply_impl(definition, param, state, formed_input, rng):
        return current_pipe(definition).contract.apply(
            expand(param), state, formed_input, rng)

    param_form = (pipe._def.calls.param.form.without(*aliases)
                  if pipe.parametric else None)
    init_role = (pipe._def.calls.init.copy(impl=init_impl)
                 if pipe._def.calls.init is not None else None)
    calls = ContractCalls(
        param=(param_call(
            param_impl,
            param_form,
            takes_rng=param_takes_rng)
               if pipe.parametric else None),
        init=init_role,
        apply=pipe._def.calls.apply.copy(impl=apply_impl),
    )
    definition = Def(
        name=f'tie({pipe.name})', calls=calls,
        members=Struct(**{
            name: member._def for name, member in pipe.members.__items__}),
        captures=active_captures,
        layout=Layout(
            kind='tie',
            param_members=active_param_members,
        ),
    )

    def bind(replacements):
        rebuilt_pipe = pipe._def.bind_members(replacements)
        return _tie(Node(rebuilt_pipe), source, aliases)._def

    return Node(definition.copy(tree=bind))


def tie(pipe: BaseNode | Generic, source: str,
        *aliases: str) -> BaseNode | Generic:
    aliases = tuple(aliases)
    if pipe.generic:
        return _tie(pipe, source, aliases)
    tied = _tie(pipe.node, source, aliases)
    if not pipe.bound:
        return tied

    param = (pipe.param.without(*aliases)
             if pipe.parametric else pipe.param)
    if pipe.state_bound:
        return tied.bind(param, state=pipe.state)
    return tied.bind(param)
