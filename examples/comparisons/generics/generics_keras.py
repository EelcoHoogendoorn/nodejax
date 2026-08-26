"""Configuring a deep composition with Keras 3 on the JAX backend.

Keras gets input-derived construction right. ``keras.Input`` carries shape
through the Functional graph, and ``Block.build(input_shape)`` creates its
square weights from the signal that reaches it. The input width is stated once
at the graph boundary and never forwarded through a constructor.

The remaining configuration has a different lifetime. ``depth``, ``members``
and ``temperature`` decide which Python layer graph to construct, so they must
be known when ``committee`` runs. Before that, Keras has a reusable Python
builder; it does not have an unresolved Model that can itself be composed and
specialized later. This column is therefore both a genuine Keras win and a
useful boundary on what NodeJAX's Generics add beyond shape inference.

Keras also has a real configuration data model. Retempering uses the standard
``clone_model`` hook to rebuild every Block at the new static value, then
``set_weights`` transfers the trained values after Keras has checked their
shapes. No mutable flag or private model surgery is needed.

The committee is a real JAX ``vmap``, not a Python loop over calls.
``Ensemble`` stacks corresponding variables from independently initialized
member Models and maps one member's pure ``stateless_call``. Keras supplies
both halves of that implementation; what it does not supply is the small
Model-to-Model lift that puts them together and reconnects updated state.
This is enough for the generics experiment; the separate lift comparison is
where arbitrary losses/aux and self-composition belong.

Run directly:

    KERAS_BACKEND=jax python -m \
        examples.comparisons.generics.generics_keras
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'jax')

import keras
import numpy as np
from keras import ops

from examples.comparisons.generics.generics_common import (
    CONFIGS, IN_DIM, LR, PARAM_KEY, RETEMPERED, TRAIN_STEPS, make_data, report)


if keras.backend.backend() != 'jax':
    raise RuntimeError(
        'the Keras comparison requires KERAS_BACKEND=jax to be set before '
        'Keras is imported')


@keras.saving.register_keras_serializable(package='nodejax_comparison')
class Block(keras.layers.Layer):
    """A same-width linear whose fan-in and fan-out come from its input."""

    def __init__(self, temperature: float, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def build(self, input_shape):
        width = int(input_shape[-1])
        self.kernel = self.add_weight(
            name='kernel', shape=(width, width), initializer='glorot_uniform')
        self.bias = self.add_weight(
            name='bias', shape=(width,), initializer='zeros')
        super().build(input_shape)

    def call(self, rows):
        return ops.tanh((ops.matmul(rows, self.kernel) + self.bias)
                        / self.temperature)

    def get_config(self):
        return {**super().get_config(), 'temperature': self.temperature}


def tower(width: int, depth: int, temperature: float, *, name: str):
    """entry -> depth same-width blocks -> readout."""
    return keras.Sequential([
        keras.layers.Dense(width),
        *[Block(temperature) for _ in range(depth)],
        keras.layers.Dense(1),
    ], name=name)


@keras.saving.register_keras_serializable(package='nodejax_comparison')
class Ensemble(keras.layers.Layer):
    """Map one Model over independently initialized params and state."""

    def __init__(self, members, **kwargs):
        super().__init__(**kwargs)
        self.members = list(members)

    def build(self, input_shape):
        for member in self.members:
            if not member.built:
                member.build(input_shape)
        super().build(input_shape)

    def call(self, rows, training=False):
        template = self.members[0]
        params = [
            ops.stack([member.trainable_variables[i]
                       for member in self.members])
            for i in range(len(template.trainable_variables))
        ]
        state = [
            ops.stack([member.non_trainable_variables[i]
                       for member in self.members])
            for i in range(len(template.non_trainable_variables))
        ]

        def apply_member(values):
            member_params, member_state = values
            return template.stateless_call(
                member_params, member_state, rows, training=training)

        outputs, next_state = ops.vectorized_map(
            apply_member, (params, state))

        # stateless_call includes BatchNorm statistics, RNG counters, and any
        # other non-trainable Variables. Reconnect its mapped successors to
        # the ordinary stateful Keras view.
        for member_index, member in enumerate(self.members):
            for state_index, variable in enumerate(
                    member.non_trainable_variables):
                variable.assign(next_state[state_index][member_index])

        return ops.mean(outputs, axis=0)

    def get_config(self):
        return {
            **super().get_config(),
            'members': [keras.saving.serialize_keras_object(member)
                        for member in self.members],
        }

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config['members'] = [
            keras.saving.deserialize_keras_object(member)
            for member in config['members']
        ]
        return cls(**config)


def committee(width: int, depth: int, members: int, temperature: float):
    """Build the Functional graph; input shape reaches every tower itself."""
    rows = keras.Input(shape=(IN_DIM,), name='rows')
    towers = [tower(width, depth, temperature, name=f'tower_{i}')
              for i in range(members)]
    mean = Ensemble(towers, name='committee_vmap')(rows)
    return keras.Model(rows, mean, name='committee')


# committee forwards width, depth and temperature to tower; tower forwards
# temperature to Block. Keras removes the fifth slot in the other OO columns:
# input fan-in follows the graph rather than becoming a constructor argument.
THREADING_TAX = 4


def configured(config: dict, seed: int):
    """Exercise 1: build one closed Functional graph from a late config."""
    keras.utils.set_random_seed(seed)
    return committee(config['width'], config['depth'], config['members'],
                     config['temperature'])


def fit(model, rows: np.ndarray, targets: np.ndarray) -> tuple:
    """Keras's high-level path: compile the captured graph and fit it."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss='mean_squared_error',
    )
    history = model.fit(
        rows,
        targets,
        batch_size=len(rows),
        epochs=TRAIN_STEPS,
        shuffle=False,
        verbose=0,
    )
    return model, np.asarray(history.history['loss'])


def retempered(model, temperature: float):
    """Exercise 2: rewrite config, rebuild, then transfer compatible values."""
    def rewrite(layer):
        if type(layer) is Ensemble:
            members = [
                keras.models.clone_model(
                    member, clone_function=rewrite, recursive=True)
                for member in layer.members
            ]
            config = layer.get_config()
            del config['members']
            return Ensemble(members, **config)

        config = layer.get_config()
        if type(layer) is Block:
            config['temperature'] = temperature
        return layer.__class__.from_config(config)

    flipped = keras.models.clone_model(
        model, clone_function=rewrite, recursive=True)
    flipped.set_weights(model.get_weights())
    return flipped


def configuration(model) -> dict:
    """Exercise 3: summarize facts read from Keras's model/layer configs."""
    ensemble = next(layer for layer in model.layers
                    if type(layer) is Ensemble)
    towers = ensemble.members
    first = towers[0]
    blocks = [layer for layer in first.layers if type(layer) is Block]
    return dict(
        members=len(towers),
        width=first.layers[0].get_config()['units'],
        depth=len(blocks),
        temperature=blocks[0].get_config()['temperature'],
    )


def main() -> None:
    rows, targets = make_data()

    reported, first_trained = [], None
    for config in CONFIGS:
        model = configured(config, PARAM_KEY)
        model, losses = fit(model, rows, targets)
        reported.append((config, model.count_params(),
                         float(losses[0]), float(losses[-1])))
        if first_trained is None:
            first_trained = model

    before = np.asarray(first_trained(rows, training=False))
    flipped = retempered(first_trained, RETEMPERED)
    after = np.asarray(flipped(rows, training=False))
    shift = float(np.mean(np.abs(after - before)))

    print('[keras] the configuration, recovered from model config data:')
    for name, value in configuration(flipped).items():
        print(f'    {name} = {value!r}')

    report('keras', reported, shift, THREADING_TAX)


if __name__ == '__main__':
    main()
