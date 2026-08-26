"""Configuring a deep composition, the equinox column.

Written the way you would write it to be REUSABLE: the architecture is
three classes, each taking the configuration it needs and handing the
rest down. No globals are read inside a module, because a module that
reads its width off a global is not an architecture, it is one model.

That leaves the threading in plain sight, which is the measurement.
Committee takes five configuration parameters and consumes exactly one
(members); the other four exist to be handed to Tower unchanged. Tower
takes temperature for no reason of its own. And `in_dim` is threaded
from the caller through both levels, because an equinox Linear must be
told its fan-in; the nodejax column never mentions a fan-in anywhere,
since a constructor there reads it off the wiring.

The temperature is a STATIC field, which it must be: a bare float on an
equinox Module is a pytree leaf, and a pytree leaf gets traced and
optimized. That correctness fix is what makes exercise 2 expensive; see
`retempered`.

Run directly:  python -m examples.comparisons.generics.generics_equinox
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from examples.comparisons.generics.generics_common import (
    CONFIGS, IN_DIM, LR, PARAM_KEY, RETEMPERED, TRAIN_STEPS, make_data, report)


class Block(eqx.Module):
    """The leaf: a same-width linear, then a tanh cooled by temperature."""
    linear: eqx.nn.Linear
    temperature: float = eqx.field(static=True)

    def __init__(self, width: int, temperature: float, *, key):
        self.linear = eqx.nn.Linear(width, width, key=key)
        self.temperature = temperature

    def __call__(self, row: jax.Array) -> jax.Array:
        return jnp.tanh(self.linear(row) / self.temperature)


class Tower(eqx.Module):
    """entry -> depth blocks -> readout. `temperature` appears in this
    signature for one reason: to reach Block."""
    entry: eqx.nn.Linear
    blocks: list
    readout: eqx.nn.Linear

    def __init__(self, in_dim: int, width: int, depth: int,
                 temperature: float, *, key):
        keys = jax.random.split(key, depth + 2)
        self.entry = eqx.nn.Linear(in_dim, width, key=keys[0])
        self.blocks = [Block(width, temperature, key=block_key)
                       for block_key in keys[1:-1]]
        self.readout = eqx.nn.Linear(width, 1, key=keys[-1])

    def __call__(self, row: jax.Array) -> jax.Array:
        carried = self.entry(row)
        for block in self.blocks:
            carried = block(carried)
        return self.readout(carried)


class Committee(eqx.Module):
    """`members` towers, averaged. Four of the five configuration
    parameters below are pure threading."""
    towers: list

    def __init__(self, in_dim: int, width: int, depth: int, members: int,
                 temperature: float, *, key):
        self.towers = [Tower(in_dim, width, depth, temperature, key=tower_key)
                       for tower_key in jax.random.split(key, members)]

    def __call__(self, row: jax.Array) -> jax.Array:
        return jnp.mean(jnp.stack([tower(row) for tower in self.towers]), axis=0)


# Tower.temperature, and Committee's in_dim, width, depth, temperature:
# five parameters whose only job is to hand a value to a deeper level
THREADING_TAX = 5


def configured(config: dict, key) -> Committee:
    """Exercise 1. The config cannot be handed over as a config: the
    constructor chain takes positional values, so the caller unpacks
    the dict here and every level re-states what it is passing on."""
    return Committee(IN_DIM, config['width'], config['depth'],
                     config['members'], config['temperature'], key=key)


@eqx.filter_jit
def fit(model: Committee, rows: jax.Array, targets: jax.Array) -> tuple:
    """The whole run as one scan, the module riding as the carry: its
    array leaves are the loop state and its static fields sit in the
    treedef, which lax.scan requires to be identical in and out. That
    holds here, and it is worth naming why: the configuration cannot
    change during a run, so a config value that lives in the treedef
    can be carried. The same fact is what makes it immovable AFTER the
    run; see `retempered`."""
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def step(carry, _):
        model, opt_state = carry

        def objective(model):
            return jnp.mean((jax.vmap(model)(rows) - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(objective)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        return (eqx.apply_updates(model, updates), opt_state), loss

    (model, _), losses = jax.lax.scan(step, (model, opt_state),
                                      length=TRAIN_STEPS)
    return model, losses


def retempered(model: Committee, config: dict, temperature: float,
               key) -> Committee:
    """Exercise 2, and the price of a static field.

    Two doors are shut before the third opens. eqx.tree_at cannot reach
    the field: `tree_at(lambda m: m.temperature, ...)` raises
    'Operation undefined, 1.0 is not a leaf of the pytree', because a
    static field is not in the pytree at all. So the architecture is
    REBUILT at the new temperature (throwing away a fresh weight draw)
    and the trained weights are moved onto it. But eqx.combine refuses
    that too: the old and new skeletons differ in exactly the metadata
    we changed, and combine compares metadata, reporting
    '(Missing, 1.0)' against '(Missing, 2.0)'.

    What is left is the transplant by hand: flatten the trained arrays
    to a list and unflatten that list into the new skeleton, trusting
    LEAF ORDER to line the two architectures up. It works here because
    the flip changed no shape. It is also the moment the config must be
    alive at the call site, since the model cannot say what it was
    built with."""
    fresh = configured({**config, 'temperature': temperature}, key)
    trained_arrays, _ = eqx.partition(model, eqx.is_array)
    fresh_arrays, skeleton = eqx.partition(fresh, eqx.is_array)
    grafted = jax.tree.unflatten(jax.tree.structure(fresh_arrays),
                                 jax.tree.leaves(trained_arrays))
    return eqx.combine(grafted, skeleton)


def main() -> None:
    rows, targets = make_data()
    rows, targets = jnp.asarray(rows), jnp.asarray(targets)
    key = jax.random.PRNGKey(PARAM_KEY)

    reported, first_trained = [], None
    for config in CONFIGS:
        model = configured(config, key)
        model, losses = fit(model, rows, targets)
        parameters = sum(leaf.size for leaf in
                         jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        reported.append((config, parameters, float(losses[0]), float(losses[-1])))
        first_trained = first_trained or model

    before = jax.vmap(first_trained)(rows)
    flipped = retempered(first_trained, CONFIGS[0], RETEMPERED, key)
    shift = float(jnp.mean(jnp.abs(jax.vmap(flipped)(rows) - before)))

    # exercise 3: there is no address book. The module can be PRINTED
    # (eqx.tree_pformat shows static fields), but a repr is not data you
    # can hand back to a builder, and nothing ties it to the config dict
    # the caller kept: the two can drift and nothing notices.
    print('[equinox] the configuration, hand-walked off the model:')
    print(f'    members = {len(flipped.towers)}')
    print(f'    depth = {len(flipped.towers[0].blocks)}')
    print(f'    width = {flipped.towers[0].entry.out_features}')
    print(f'    temperature = {flipped.towers[0].blocks[0].temperature}')

    report('equinox', reported, shift, THREADING_TAX)


if __name__ == '__main__':
    main()
