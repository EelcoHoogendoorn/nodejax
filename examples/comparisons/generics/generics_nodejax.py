"""Deferred configuration of a deep composition in NodeJAX.

`committee()` defines an ensemble of stacked residual-style towers without taking configuration arguments: `nn.Linear()`, `stack(Block())`, and `ensemble(tower)` leave `n_out`, `n`, `temperature`, and committee size unbound in the `Generic` definition. `configured(config)` resolves all four statics in one `.specialize(...)` call, fan-in width derives automatically from dataflow at `.with_input(rows)`, and post-training retempering re-specializes `*.temperature` while keeping the learned parameter tree intact (`THREADING_TAX = 0`).

Run directly with `python -m examples.comparisons.generics.generics_nodejax`.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    KeyStream, Leaf, Node, Struct, ensemble, nn, node, reduce,
    serial, split_aux, stack, tile, train_step, trained,
)

from examples.comparisons.generics.generics_common import (
    CONFIGS, LR, PARAM_KEY, RETEMPERED, TRAIN_STEPS, make_data, report,
)


@node
def Block(temperature: float) -> Node:
    """Applies a same-width linear projection followed by temperature-scaled `tanh`."""
    def param(node, rng: KeyStream) -> Struct:
        width = node.input.shape[-1]
        return Struct(
            w=jax.random.normal(rng.next(), (width, width)) / jnp.sqrt(width),
            b=jnp.zeros(width),
        )

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        return jnp.tanh((input @ param.w + param.b) / temperature)

    return Leaf(apply, param=param, name='block')


def committee() -> Node:
    """Assembles the generic committee with all four hyperparameters left unbound."""
    tower = serial(entry=nn.Linear(), blocks=stack(Block()), readout=nn.Linear(1))
    return ensemble(tower) >> reduce(jnp.mean)


ADDRESSES = dict(
    members='ensemble.n',
    width='ensemble.member.entry.n_out',
    depth='ensemble.member.blocks.n',
    temperature='ensemble.member.blocks.layer.temperature',
)
THREADING_TAX = 0


def configured(config: dict) -> Node:
    return committee().specialize(**{
        ADDRESSES[knob]: value for knob, value in config.items()
    })


def mean_squared(output: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((output - target) ** 2)


def fit(model, rows: jax.Array, targets: jax.Array) -> tuple:
    trainer = train_step(model, mean_squared, optax.adam(LR))
    done, aux = trained(trainer).apply(
        input=tile(rows, TRAIN_STEPS), target=tile(targets, TRAIN_STEPS)
    )
    return done, aux.loss


def predictions(model, rows: jax.Array) -> jax.Array:
    clean, _ = split_aux(model.apply(rows))
    return clean


def main() -> None:
    rows_np, targets_np = make_data()
    rows, targets = jnp.asarray(rows_np), jnp.asarray(targets_np)

    rows_out, trained_first = [], None
    for config in CONFIGS:
        model = configured(config).with_input(rows).parameterize(
            rng=jax.random.PRNGKey(PARAM_KEY)
        ).initialize()
        done, losses = fit(model, rows, targets)
        parameters = sum(leaf.size for leaf in jax.tree.leaves(done.param))
        rows_out.append((config, parameters, float(losses[0]), float(losses[-1])))
        trained_first = trained_first or done

    before = predictions(trained_first.pnode, rows)
    retempered = trained_first.specialize(
        **{'*.temperature': RETEMPERED}
    ).bind(trained_first.param)
    shift = float(jnp.mean(jnp.abs(predictions(retempered, rows) - before)))

    print('[nodejax] static configuration by path:')
    for path, value in retempered.statics_by_path().items():
        print(f'    {path} = {value!r}')

    report('nodejax', rows_out, shift, THREADING_TAX)


if __name__ == '__main__':
    main()
