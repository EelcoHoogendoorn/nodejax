from __future__ import annotations

import jax

from nodejax.struct import Struct
from nodejax.core import NodeDef, Composite
from nodejax.generic import _over_generic


@_over_generic
def tie(pipe: NodeDef, source: str, *aliases: str) -> NodeDef:
    """Share one member's params with others — sharing as
    reparameterization. The composite param carries only `source` (alias
    slots hold ()); expand() inserts that subtree at each alias slot before
    init/apply (aliases must expect the same param structure). Gradients from
    all uses accumulate automatically; member STATE stays separate.

    consumers: tied autoencoders (encoder/decoder weights),
    tied embeddings (embed/unembed in a language model)."""
    if not isinstance(pipe, Composite):     # blessed(2): structural dispatch
        raise TypeError(f"tie rewires member param slots and '{pipe.name}' has none")
    missing = ({source} | set(aliases)) - set(pipe.members)
    if missing:
        raise TypeError(f"tie: {sorted(missing)} name no member of '{pipe.name}'")

    def expand(param):
        return param.replace(**{alias: param[source] for alias in aliases})

    def param_fn(ndef, param_input=Struct()):
        supplied = set(aliases) & set(param_input.__keys__)
        if supplied:
            raise TypeError(f"tied members {sorted(supplied)} share '{source}'; "
                            'do not parameterize them separately')
        key = param_input.rng if 'rng' in param_input else None
        parts = {}
        for nm, d in pipe.members.items():
            if nm in aliases:
                parts[nm] = ()               # filled from `source` by expand()
                continue
            if not d.parametric:
                parts[nm] = ()
                continue
            bundle = param_input[nm] if nm in param_input else Struct()
            if key is not None and 'rng' in d.param_input_spec and 'rng' not in bundle:
                key, sub_ = jax.random.split(key)
                bundle = bundle.replace(rng=sub_)
            parts[nm] = d.build_param(bundle)
        return Struct(**parts)

    def init_fn(ndef, param, state_input=Struct(), input=None):
        d = pipe if ndef.apply_input_spec is None else pipe.with_input(ndef.apply_input_spec)
        return d.init_fn(expand(param), state_input, input)

    def apply_fn(nd, param, state, input):
        return pipe.apply_fn(expand(param), state, input)

    # members ride along for introspection; serial stays False: flattening a
    # tied pipe across >> would splat the members and lose the tie, so a tied
    # def is atomic.
    tied_param_spec = (pipe.param_input_spec.without(*aliases)
                       if pipe.parametric else None)
    return Composite(f'tie({pipe.name})', param_fn, init_fn, apply_fn,
                     pipe.parametric, pipe.cyclic, members=pipe.members,
                     apply_input_spec=pipe.apply_input_spec,
                     param_input_spec=tied_param_spec,
                     state_input_spec=pipe.state_input_spec if pipe.cyclic else None)
