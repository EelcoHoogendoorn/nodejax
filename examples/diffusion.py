"""Implements deterministic DDIM reverse diffusion over arbitrary sample PyTrees."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nodejax import (
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    PyTree,
    Struct,
    node,
    nn,
    repeat,
    serial,
)


def _validate_schedule_values(steps: int, floor: float) -> None:
    if steps < 1:
        raise ValueError('a diffusion schedule needs at least one step')
    if not 0.0 < floor <= 1.0:
        raise ValueError('a diffusion schedule floor must be in (0, 1]')


def linear_alpha_bar(steps: int, floor: float) -> tuple[float, ...]:
    """Decreases cumulative alpha linearly from `1.0` to `floor` across `steps` intervals."""
    _validate_schedule_values(steps, floor)
    values = np.linspace(1.0, floor, steps + 1)
    return tuple(float(value) for value in values)


def cosine_alpha_bar(steps: int, floor: float) -> tuple[float, ...]:
    """Computes a cosine cumulative-alpha schedule clipped below at `floor`."""
    _validate_schedule_values(steps, floor)
    time = np.arange(steps + 1) / steps
    offset = 0.008
    values = np.cos((time + offset) / (1.0 + offset) * np.pi / 2.0) ** 2
    values = np.clip(values / values[0], floor, 1.0)
    return tuple(float(value) for value in values)


def _assert_matching_shapes(
    reference: PyTree,
    candidate: PyTree,
    label: str,
) -> None:
    """Asserts that every leaf in `candidate` matches the shape of its counterpart in `reference`."""
    matches = jax.tree.map(
        lambda reference_leaf, candidate_leaf: (
            jnp.shape(reference_leaf) == jnp.shape(candidate_leaf)
        ),
        reference,
        candidate,
    )
    assert all(jax.tree.leaves(matches)), (
        f'{label} must preserve every sample leaf shape'
    )


def _validate_alpha_bar(alpha_bar: tuple[float, ...]) -> None:
    values = np.asarray(alpha_bar, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError('alpha_bar needs at least a clean and noisy endpoint')
    if not np.all(np.isfinite(values)):
        raise ValueError('alpha_bar values must be finite')
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError('alpha_bar values must be in (0, 1]')
    if np.any(np.diff(values) > 0.0):
        raise ValueError('alpha_bar must be non-increasing')


@node
def DenoisingStep(
    predictor: BaseNode,
    clean: BaseNode,
    alpha_bar: tuple[float, ...],
) -> Node | PNode:
    """Applies one deterministic epsilon-prediction DDIM update over a sample PyTree."""
    _validate_alpha_bar(alpha_bar)
    schedule = jnp.asarray(alpha_bar, dtype=jnp.float32)
    steps = len(alpha_bar) - 1
    members = Composite(predictor=predictor, clean=clean)

    def apply(self, input) -> Struct:
        assert input.reverse.shape == ()
        time = input.reverse.astype(jnp.float32) / steps
        current_alpha = schedule[input.reverse]
        previous_alpha = schedule[input.reverse - 1]
        predicted_noise = self.predictor(Struct(
            condition=input.condition,
            sample=input.sample,
            time=time,
            alpha_bar=current_alpha,
        ))
        _assert_matching_shapes(input.sample, predicted_noise, 'predictor output')

        clean_sample = jax.tree.map(
            lambda sample, noise: (
                sample - jnp.sqrt(1.0 - current_alpha) * noise
            ) / jnp.sqrt(current_alpha),
            input.sample,
            predicted_noise,
        )
        clean_sample = self.clean(clean_sample)
        _assert_matching_shapes(input.sample, clean_sample, 'clean output')

        next_sample = jax.tree.map(
            lambda clean_value, noise: (
                jnp.sqrt(previous_alpha) * clean_value
                + jnp.sqrt(1.0 - previous_alpha) * noise
            ),
            clean_sample,
            predicted_noise,
        )
        return input.replace(
            sample=next_sample,
            reverse=input.reverse - 1,
        )

    return members(apply)


@node
def Denoiser(
    predictor: BaseNode,
    alpha_bar: tuple[float, ...],
    clean: BaseNode = nn.identity,
) -> Node | PNode:
    """Iterates deterministic epsilon-prediction DDIM from an initial noisy sample.

    Consumes `Struct(condition=<pytree>, sample=<pytree>)` and passes `Struct(condition, sample, time, alpha_bar)` to `predictor` at each reverse step. Both `predictor` and `clean` must preserve the leaf shapes of `sample`.
    """
    _validate_alpha_bar(alpha_bar)
    steps = len(alpha_bar) - 1

    def countdown(input) -> Struct:
        return Struct(
            condition=input.condition,
            sample=input.sample,
            reverse=jnp.asarray(steps, dtype=jnp.int32),
        )

    return serial(
        countdown=Leaf(countdown, name='countdown'),
        iterations=repeat(
            DenoisingStep(predictor, clean, alpha_bar),
            n=steps,
        ),
        sample=Leaf(lambda input: input.sample, name='sample'),
    )
