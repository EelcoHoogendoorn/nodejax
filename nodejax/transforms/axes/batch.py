from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.contract import Contract
from nodejax.core.node import Node
from nodejax.core.spec import element_spec, add_axis, axis_count
from nodejax.struct import Struct
from nodejax.tree import tree_first
from nodejax.transforms.transform import transform, vmap_apply, vmap_prime
from nodejax.core.wrapper import Wrapper


def _state_axes(inner: Contract):
    """Map ordinary state and broadcast ``single_batch_state`` members."""
    return inner.state_tree(
        lambda member: None if 'single_batch_state' in member.tags else 0)


def _batched_states(inner: Contract, single_state, count: int):
    """Tile state rows while retaining explicitly shared state."""
    def split_or_tile(tree):
        if type(tree) is Struct:
            return Struct(**{
                name: (jax.random.split(value, count) if name == 'rng'
                       else split_or_tile(value))
                for name, value in tree.__items__})
        return jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf), (count,) + jnp.shape(leaf)),
            tree,
        )

    return inner.map_state(
        single_state,
        lambda member, m_state: m_state
        if 'single_batch_state' in member.tags
        else split_or_tile(m_state))


@transform(preserves='param')
def batch(sample: Node, n: int | None = None,
          axis: str = 'batch') -> Node:
    """vmap over the input axis: params broadcast, input/output/state batched.

    Type-preserving; accepts nodes and bound nodes (param meaning unchanged).
    For cyclic nodes the per-element state is tiled to the batch size: read
    from the bound batched input shape, or from the static n=<batch size>
    here when the node inits without a shape (a shape-free, constant state).

    The vmap axis is named ``batch`` by default, so members can use JAX
    collectives over it. Nodes tagged with
    'single_batch_state' retain a single unbatched state across the batch.
    """
    def apply_fn(contract, param, state, input, rng):
        """State axes are both in_axes and out_axes: a 'single_batch_state'
        member broadcasts in and comes back unmapped, no slicing after
        the fact."""
        inner = contract.members.sample
        state_axes = _state_axes(inner)
        return vmap_apply(
            inner, param, state, input, rng,
            param_axis=None, state_axis=state_axes, input_axis=0,
            axis_name=axis)

    def init_context(contract):
        """Resolve the sample shape and the batch extent at this boundary."""
        inner = contract.members.sample
        batched = contract.input_spec
        rows = axis_count(batched)
        if rows is not None and n is not None and n != rows:
            raise TypeError(
                f'batch({inner.name}): n={n} conflicts with the bound '
                f'batched axis {rows}')
        count = rows if rows is not None else n
        if count is None:
            raise TypeError(
                f'batch({inner.name}).init needs a bound batched input '
                'shape (with_input(<batched spec>)), or batch(node, '
                'n=<batch size>) for a shape-free node')
        element_shape = (None if batched is None else element_spec(batched))
        return inner, element_shape, count

    def init_fn(contract, param, state_input, rng):
        """One state per element: a random init draws per element, as prime
        does; a deterministic init builds one state and tiles it."""
        inner, element_shape, count = init_context(contract)
        current = (inner if element_shape is None else
                   inner._resolve_def(
                       element_shape, bundled=True).contract)
        rngs, rng_axis = rng.axis(current.init_takes_rng, count)
        if rng_axis is None:
            return _batched_states(inner, current.init(param, state_input, rngs), count)
        return jax.vmap(
            lambda child_rng: current.init(param, state_input, child_rng),
            in_axes=rng_axis,
            out_axes=_state_axes(inner),
            axis_name=axis,
            axis_size=count,
        )(rngs)

    def prime_fn(contract, param, state_input, input, rng):
        inner, _, count = init_context(contract)
        return vmap_prime(
            inner, param, state_input, input, rng,
            count=count, param_axis=None, input_axis=0,
            state_axis=_state_axes(inner), axis_name=axis)

    def param_fn(contract, param_input, rng):
        """Parameterize one shared sample tree while binding the batch axis."""
        inner = contract.members.sample
        batched = contract.input_spec
        current = (inner if batched is None else
                   inner._resolve_def(
                       element_spec(batched), bundled=True).contract)
        count = axis_count(batched)
        count = n if count is None else count
        if count is None:
            return current.param(param_input, rng)
        return jax.vmap(
            lambda unused: current.param(param_input, rng),
            out_axes=None,
            axis_name=axis,
        )(jnp.arange(count))

    batched = add_axis(sample.contract.input_spec, n)
    return Wrapper(sample=sample).roles(
        name=f'batch({sample.name})',
        destructurable_state=False,   # state tiles per element; params stay flat
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        input_spec=batched,
    )


@transform(preserves='param')
def unbatched(inner: Node, axis: str = 'batch') -> Node:
    """Run one sample while binding the named axis at size one.

    The temporary axis is added before the inner call and removed from its
    state and output afterward.
    """
    def lead(tree):
        return jax.tree.map(lambda value: jnp.asarray(value)[None], tree)

    def sample_contract(contract):
        sample = contract.members.sample
        input_spec = contract.input_spec
        return (sample if input_spec is None else
                sample._resolve_def(input_spec, bundled=True).contract)

    def param_fn(contract, param_input, rng):
        sample = sample_contract(contract)
        return jax.vmap(
            lambda unused: sample.param(param_input, rng),
            out_axes=None,
            axis_name=axis,
        )(jnp.arange(1))

    def init_fn(contract, param, state_input, rng):
        sample = sample_contract(contract)
        state = jax.vmap(
            lambda unused: sample.init(param, state_input, rng),
            out_axes=0,
            axis_name=axis,
        )(jnp.arange(1))
        return tree_first(state)

    def prime_fn(contract, param, state_input, input, rng):
        sample = sample_contract(contract)
        state = vmap_prime(
            sample, param, state_input, lead(input), rng,
            count=1, param_axis=None, input_axis=0,
            state_axis=0, axis_name=axis,
        )
        return tree_first(state)

    def apply_fn(contract, param, state, input, rng):
        sample = contract.members.sample
        batched_input = lead(input)
        new_state, out = vmap_apply(
            sample, param, lead(state), batched_input, rng,
            param_axis=None, state_axis=0, input_axis=0,
            axis_name=axis, count=1)
        strip = tree_first
        return strip(new_state), strip(out)

    return Wrapper(sample=inner).roles(
        name=f'unbatched({inner.name})',
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
    )
