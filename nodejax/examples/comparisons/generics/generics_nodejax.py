"""Configuring a deep composition, the nodejax side of the comparison.

The architecture is defined once, deciding nothing: a committee of
towers, each an entry linear, a stack of blocks, and a readout. Read
`committee` below and notice what is absent. No function in it takes a
configuration argument, because there is no level to thread one
through: an @node factory called without its statics yields the same
description with those arguments unbound, and the composition carries
them upward as data. `statics_by_path` prints exactly the addresses
`specialize` accepts, so configuring the whole tree is one call with a
flat dict, and the rival columns' constructor chains have no
counterpart here.

The block's WIDTH is not even a knob: its constructor reads the width
off whatever it is wired to, so the model's width reaches every block
without being passed to anything. Equinox, NNX and Torch carry that value
through their constructor chains. Keras infers it from the Functional graph,
which is a genuine strength; its difference from this column is the arbitrary
unresolved statics above that shape-derived layer.

One structural difference to name rather than hide: `stack` compiles
ONE block and scans its stacked params over depth, where Equinox, NNX and
Torch unroll a list of `depth` modules. The Keras column maps its committee
through a local `stateless_call` lift, but still unrolls depth. All build the
same parameter count and train to the same place; the cost of reusable lifting
is measured directly in the lift and tower comparisons, not here.

Run directly:  python -m nodejax.examples.comparisons.generics.generics_nodejax
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (node, nn, Leaf, Node, PNode, Struct, KeyStream,
                     stack, ensemble, reduce, train_step, trained, tile,
                     split_aux)

from nodejax.examples.comparisons.generics.generics_common import (
    CONFIGS, PARAM_KEY, TRAIN_STEPS, LR, RETEMPERED, make_data, report)


@node
def Block(temperature: float) -> Node:
    """One tower block: a same-width linear, then a tanh cooled by
    `temperature`."""
    def param(node, rng: KeyStream) -> Struct:
        width = node.input.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (width, width)) / jnp.sqrt(width),
                      b=jnp.zeros(width))

    def apply(param, input: jax.Array) -> jax.Array:
        return jnp.tanh((input @ param.w + param.b) / temperature)

    return Leaf(apply, param=param, name='block')


def committee() -> Node:
    """The architecture, deciding nothing: every knob left unbound, so
    the whole tree is a generic until one call fills it."""
    tower = nn.Linear() >> stack(Block()) >> nn.Linear(1)
    return ensemble(tower) >> reduce(jnp.mean)


# where each knob lives in the record, which is where specialize takes
# it. This dict IS the configuration surface: the architecture never
# mentions these names, and nothing between the top and a block has a
# parameter for them.
ADDRESSES = dict(
    members='ensemble.n',
    width='ensemble.member.linear_stack.linear.n_out',
    depth='ensemble.member.linear_stack.stack.n',
    temperature='ensemble.member.linear_stack.stack.layer.temperature',
)
THREADING_TAX = 0                       # no function forwards a config value


def configured(config: dict) -> Node:
    """Exercise 1: one config, one architecture, one call."""
    return committee().specialize(**{ADDRESSES[knob]: value
                                     for knob, value in config.items()})


def mean_squared(output, target: jax.Array) -> jax.Array:
    """The committee sows its member population to aux; the objective
    scores the mean it emitted."""
    clean, _ = split_aux(output)
    return jnp.mean((clean - target) ** 2)


def fit(model, rows: jax.Array, targets: jax.Array) -> tuple:
    """Train to completion, returning the trained model and its loss
    trace."""
    trainer = train_step(model, mean_squared, optax.adam(LR))
    done, aux = trained(trainer).apply(input=tile(rows, TRAIN_STEPS),
                                       target=tile(targets, TRAIN_STEPS))
    return done, aux.loss


def predictions(model, rows: jax.Array) -> jax.Array:
    clean, _ = split_aux(model.apply(rows))
    return clean


def main() -> None:
    rows, targets = make_data()
    rows, targets = jnp.asarray(rows), jnp.asarray(targets)

    rows_out, trained_first = [], None
    for config in CONFIGS:
        model = configured(config).with_input(rows).parameterize(
            rng=jax.random.PRNGKey(PARAM_KEY)).initialize()
        done, losses = fit(model, rows, targets)
        parameters = sum(leaf.size for leaf in jax.tree.leaves(done.param))
        rows_out.append((config, parameters, float(losses[0]), float(losses[-1])))
        trained_first = trained_first or done

    # exercise 2: the deepest knob flips on the TRAINED model, the
    # weights carried verbatim. specialize re-runs the record, so
    # anything the change implied downstream re-derives with it
    before = predictions(trained_first.pnode, rows)
    retempered = trained_first.specialize(
        **{'*.temperature': RETEMPERED}).bind(trained_first.param)
    shift = float(jnp.mean(jnp.abs(predictions(retempered, rows) - before)))

    # exercise 3: the configuration, read back off the model itself
    print('[nodejax] the configuration, as data:')
    for path, value in retempered.statics_by_path().items():
        print(f'    {path} = {value!r}')

    report('nodejax', rows_out, shift, THREADING_TAX)


if __name__ == '__main__':
    main()
