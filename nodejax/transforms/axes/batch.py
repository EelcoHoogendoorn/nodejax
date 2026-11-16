"""Data-parallel `batch` and single-sample `unbatched` axis transforms.

`batch` maps a single-sample `Node` across a leading input axis via `jax.vmap`
while broadcasting shared parameters, tiling or per-row initializing state, and
binding a named axis (`'batch'` by default) so members such as `BatchNorm` can
run JAX collectives across the batch. Members carrying the
`'single_batch_state'` tag keep one unmapped state shared across all batch
elements. `unbatched` provides the inverse size-1 named-axis frame so a Node
that expects a named `'batch'` axis can run on a single unbatched sample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax.core.contract import Contract
from nodejax.core.node import Node
from nodejax.core.spec import add_axis, axis_count, element_spec
from nodejax.core.wrapper import Wrapper
from nodejax.struct import Struct
from nodejax.transforms.transform import transform, vmap_apply, vmap_prime
from nodejax.tree import tree_first


def _state_axes(inner: Contract):
    return inner.state_tree(
        lambda member: None if 'single_batch_state' in member.tags else 0)


def _batched_states(inner: Contract, single_state, count: int):
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
    """Maps `sample` across leading input axis 0 with shared parameters and per-element state.

    Unlike `ensemble`, `batch` preserves the parameter tree unchanged
    (`preserves='param'`). Cyclic state acquires a leading batch axis read from
    bound batched input shapes or static `n`, except on members tagged
    `'single_batch_state'`, which broadcast in and return unmapped along the
    named `vmap` axis `axis`.
    """
    def apply_fn(contract, param, state, input, rng):
        # Shared `'single_batch_state'` leaves use `None` in both `in_axes` and `out_axes`.
        inner = contract.members.sample
        state_axes = _state_axes(inner)
        return vmap_apply(
            inner, param, state, input, rng,
            param_axis=None, state_axis=state_axes, input_axis=0,
            axis_name=axis)

    def init_context(contract):
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
        # Deterministic empty-bundle init builds one state and tiles it; otherwise `vmap` runs per row.
        inner, element_shape, count = init_context(contract)
        current = (inner if element_shape is None else
                   inner._resolve_def(
                       element_shape, bundled=True).contract)
        rngs, rng_axis = rng.axis(current.init_takes_rng, count)
        if rng_axis is None and not jax.tree.leaves(state_input):
            return _batched_states(inner, current.init(param, state_input, rngs), count)
        mapped = dict(out_axes=_state_axes(inner), axis_name=axis, axis_size=count)
        if rng_axis is None:
            return jax.vmap(
                lambda bundle: current.init(param, bundle, rngs), in_axes=0, **mapped,
            )(state_input)
        return jax.vmap(
            lambda bundle, child_rng: current.init(param, bundle, child_rng),
            in_axes=(0, rng_axis), **mapped,
        )(state_input, rngs)

    def prime_fn(contract, param, state_input, input, rng):
        inner, _, count = init_context(contract)
        return vmap_prime(
            inner, param, state_input, input, rng,
            count=count, param_axis=None, input_axis=0,
            state_axis=_state_axes(inner), axis_name=axis)

    def param_fn(contract, param_input, rng):
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
        destructurable_state=False,
        param=param_fn,
        init=init_fn,
        prime=prime_fn,
        apply=apply_fn,
        input_spec=batched,
    )


@transform(preserves='param')
def unbatched(inner: Node, axis: str = 'batch') -> Node:
    """Evaluates `inner` on a single unbatched sample inside a size-1 named `vmap` frame `axis`.

    Inserts a temporary leading axis of length 1 before calling `inner` and
    strips that axis from the resulting state and output afterward so
    collective-aware Nodes (`BatchNorm`) work on unbatched inputs.
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
