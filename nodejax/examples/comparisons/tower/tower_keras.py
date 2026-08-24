"""Deeply nested composition, in Keras 3 on the JAX backend: the same
one-tree tower as `tower_nodejax.py`, a residual stacked RNN committee
adapted per task inside the meta-training loop.

Same model, task, and budget. Keras 3 is a wrapper over several backends, and
separating what it contributes from what its backend contributes is the point
of reading this file.

`keras.layers.RNN` lowers time to a JAX scan, and
`keras.ops.vectorized_map` maps the independently parameterized members. The
depth stack remains ordinary Layer composition, and the committee is built as
ordinary Python structure before its variables are stacked for the mapped
call. This keeps the model visible as Keras Layers instead of manually packing
the architectural axes into bespoke weight tensors.

Keras contributes the object model and `stateless_call`, which applies a Layer
at explicit variable values. JAX supplies the differentiation required by the
differentiable inner optimization. The result is a valid Keras/JAX program,
but the reusable lift from member Layers to a mapped committee is local code
rather than a stock Layer-to-Layer transform.

Run directly:  python -m nodejax.examples.comparisons.tower.tower_keras
"""

import os

# Keras picks its backend at import and defaults to TensorFlow.
os.environ.setdefault('KERAS_BACKEND', 'jax')

import jax
import keras

from nodejax.examples.comparisons.tower.tower_common import (
    HIDDEN, LAYERS, MEMBERS, META_STEPS, INNER_LR, OUTER_LR, MOMENTUM,
    make_tasks,
)


class ResidualCell(keras.Layer):
    """One recurrent depth layer."""

    def build(self, _):
        self.wx = self.add_weight(
            shape=(HIDDEN,), name='wx',
            initializer=keras.initializers.RandomNormal(stddev=0.5))
        self.wh = self.add_weight(
            shape=(HIDDEN, HIDDEN), name='wh',
            initializer=keras.initializers.RandomNormal(
                stddev=0.3 / HIDDEN ** 0.5))
        self.b = self.add_weight(
            shape=(HIDDEN,), name='b', initializer='zeros')
        self.built = True

    def call(self, hidden, signal):
        advanced = keras.ops.tanh(
            self.wx * signal + keras.ops.matmul(hidden, self.wh) + self.b)
        return advanced, advanced


class TowerCell(keras.Layer):
    """Project, normalize, traverse the residual depth stack, and read out."""

    output_size = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cells = [
            ResidualCell(name=f'depth_{index}')
            for index in range(LAYERS)
        ]

    @property
    def state_size(self):
        return [HIDDEN, HIDDEN] + [HIDDEN] * LAYERS

    def build(self, _):
        self.up_w = self.add_weight(
            shape=(HIDDEN,), name='up_w',
            initializer=keras.initializers.RandomNormal(stddev=0.5))
        self.ro_w = self.add_weight(
            shape=(HIDDEN,), name='ro_w',
            initializer=keras.initializers.RandomNormal(stddev=0.1))
        self.ro_b = self.add_weight(
            shape=(), name='ro_b', initializer='zeros')
        for cell in self.cells:
            cell.build(None)
        self.built = True

    def call(self, input, states):
        mean, variance = states[0], states[1]
        hiddens = list(states[2:])

        signal = self.up_w * input
        normalized = (signal - mean) / keras.ops.sqrt(variance + 1e-5)
        next_mean = (1 - MOMENTUM) * mean + MOMENTUM * signal
        next_variance = (
            (1 - MOMENTUM) * variance
            + MOMENTUM * (signal - mean) ** 2)

        signal = normalized
        for depth, cell in enumerate(self.cells):
            hiddens[depth], contribution = cell(hiddens[depth], signal)
            signal = signal + contribution

        prediction = keras.ops.matmul(signal, self.ro_w) + self.ro_b
        return (
            keras.ops.reshape(prediction, (-1, 1)),
            [next_mean, next_variance] + hiddens,
        )


def build_tower() -> keras.layers.RNN:
    tower = keras.layers.RNN(TowerCell(), return_sequences=True)
    tower.build((None, None, 1))
    return tower


def initial_state(batch: int) -> list:
    return (
        [keras.ops.zeros((batch, HIDDEN)),
         keras.ops.ones((batch, HIDDEN))]
        + [keras.ops.zeros((batch, HIDDEN))] * LAYERS
    )


def stack_members(members: list) -> list:
    variable_rows = [member.trainable_variables for member in members]

    def stack_row(*row):
        return keras.ops.stack([
            keras.ops.convert_to_tensor(weight) for weight in row
        ])

    return keras.tree.map_structure(stack_row, *variable_rows)


def committee_prediction(template, stacked: list, xs):
    sequence = keras.ops.reshape(xs, (1, -1, 1))

    def one_member(variables):
        output, _ = template.stateless_call(
            variables, [], sequence, initial_state=initial_state(1))
        return keras.ops.reshape(output, (-1,))

    population = keras.ops.vectorized_map(one_member, stacked)
    return keras.ops.mean(population, axis=0)


def meta_loss(template, stacked: list, sup_x, sup_y, qry_x, qry_y):
    def per_task(support_x, support_y, query_x, query_y):
        fast = stacked
        for step in range(support_x.shape[0]):
            def support_loss(weights):
                prediction = committee_prediction(
                    template, weights, support_x[step])
                return keras.ops.mean((prediction - support_y[step]) ** 2)

            gradients = jax.grad(support_loss)(fast)
            fast = [
                weight - INNER_LR * gradient
                for weight, gradient in zip(fast, gradients)
            ]
        prediction = committee_prediction(template, fast, query_x)
        return keras.ops.mean((prediction - query_y) ** 2)

    return keras.ops.mean(keras.ops.vectorized_map(
        lambda task: per_task(*task), (sup_x, sup_y, qry_x, qry_y)))


def main() -> float:
    keras.utils.set_random_seed(0)
    members = [build_tower() for _ in range(MEMBERS)]
    template = members[0]
    stacked = stack_members(members)

    optimizer = keras.optimizers.Adam(learning_rate=OUTER_LR)
    optimizer_variables = [keras.Variable(weight) for weight in stacked]
    optimizer.build(optimizer_variables)
    optimizer_state = [
        keras.ops.convert_to_tensor(variable)
        for variable in optimizer.variables
    ]
    tasks = make_tasks(jax.random.PRNGKey(2))

    @jax.jit
    def meta_step(stacked: list, optimizer_state: list, tasks):
        loss, gradients = jax.value_and_grad(meta_loss, argnums=1)(
            template, stacked, *tasks)
        advanced, next_state = optimizer.stateless_apply(
            optimizer_state, gradients, stacked)
        return advanced, next_state, loss

    losses = []
    for _ in range(META_STEPS):
        stacked, optimizer_state, loss = meta_step(
            stacked, optimizer_state, tasks)
        losses.append(float(loss))

    print(f'tower loss {losses[0]:.4f} -> {losses[-1]:.4f}')
    assert losses[-1] < 0.3 * losses[0]
    return losses[-1]


if __name__ == '__main__':
    main()
