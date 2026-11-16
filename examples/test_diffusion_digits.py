"""Deterministic DDIM reverse diffusion applied to handwritten digits.

`Denoiser` iterates an injected noise predictor across a cumulative-alpha schedule without fixing
the sample PyTree layout, conditioning representation, or network architecture.

`digit_predictor` pairs `DigitFeatures` with a two-hidden-layer MLP predicting Gaussian noise from
a corrupted `8x8` digit, a one-hot class vector, and Fourier time features. Training samples one
random forward-noising step per example, and evaluation binds the learned predictor into `Denoiser`
to reconstruct held-out digits from the schedule endpoint.

Running the module directly additionally trains a second predictor on a near-zero-floor schedule
(`floor=0.02`) and generates class-conditioned digit samples from standard normal noise.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import load_digits

from nodejax import (
    Aux,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    node,
    nn,
    serial,
    train_step,
    trained,
    tree_first,
)
from examples.diffusion import Denoiser, linear_alpha_bar


IMAGE = 8
PIXELS = IMAGE * IMAGE
CLASSES = 10
HIDDEN = 256
FOURIER_FEATURES = 6
DIFFUSION_STEPS = 30
BATCH = 64
UPDATES = 1_000
LEARNING_RATE = 2e-3
TRAIN_IMAGES = 1_400
TEST_IMAGES = 200


ALPHA_BAR = linear_alpha_bar(DIFFUSION_STEPS, floor=0.6)


@node
def PixelRange() -> PNode:
    """Clips clean digit estimates to the normalized `[-1.0, 1.0]` pixel interval."""
    return Leaf(lambda input: jnp.clip(input, -1.0, 1.0))


@node
def DigitFeatures() -> PNode:
    """Flattens one noisy `8x8` digit and appends one-hot class and Fourier time features."""
    frequencies = 2.0 ** jnp.arange(FOURIER_FEATURES)

    def apply(input) -> jax.Array:
        assert input.sample.shape == (IMAGE, IMAGE)
        assert input.condition.shape == (CLASSES,)
        assert input.time.shape == ()
        assert input.alpha_bar.shape == ()
        phase = jnp.pi * input.time * frequencies
        time_features = jnp.concatenate((
            input.time[None],
            jnp.sin(phase),
            jnp.cos(phase),
        ))
        return jnp.concatenate((
            input.sample.reshape(PIXELS),
            input.condition,
            time_features,
        ))

    return Leaf(
        apply,
        apply_input_spec=Struct(
            condition=jnp.zeros(CLASSES),
            sample=jnp.zeros((IMAGE, IMAGE)),
            time=jnp.zeros(()),
            alpha_bar=jnp.ones(()),
        ),
    )


def digit_predictor() -> Node:
    """Builds the digit noise-prediction MLP for injection into `Denoiser`."""
    return serial(
        features=DigitFeatures(),
        hidden1=nn.Linear(HIDDEN),
        activation1=nn.silu,
        hidden2=nn.Linear(HIDDEN),
        activation2=nn.silu,
        noise=nn.Linear(
            PIXELS,
            weight_init=jax.nn.initializers.zeros,
        ),
        image=nn.Reshape((IMAGE, IMAGE)),
    )


def digit_data() -> Struct:
    """Loads a deterministic train and test split of normalized `8x8` digits from `sklearn`."""
    digits = load_digits()
    images = jnp.asarray(digits.images / 8.0 - 1.0, dtype=jnp.float32)
    labels = jnp.asarray(digits.target, dtype=jnp.int32)
    permutation = np.random.RandomState(0).permutation(len(images))
    images = images[permutation]
    labels = labels[permutation]
    return Struct(
        train=Struct(
            images=images[:TRAIN_IMAGES],
            labels=labels[:TRAIN_IMAGES],
        ),
        test=Struct(
            images=images[TRAIN_IMAGES:],
            labels=labels[TRAIN_IMAGES:],
        ),
    )


def forward_noise_data(
    images: jax.Array,
    labels: jax.Array,
    updates: int,
    batch_size: int,
    rng: jax.Array,
    alpha_bar: tuple[float, ...] = ALPHA_BAR,
) -> Struct:
    """Draws update-major epsilon-prediction training batches at random diffusion steps."""
    assert images.ndim == 3
    assert images.shape[1:] == (IMAGE, IMAGE)
    assert labels.shape == images.shape[:1]
    index_key, time_key, noise_key = jax.random.split(rng, 3)
    indices = jax.random.randint(
        index_key,
        (updates, batch_size),
        0,
        images.shape[0],
    )
    steps = len(alpha_bar) - 1
    reverse = jax.random.randint(
        time_key,
        (updates, batch_size),
        1,
        steps + 1,
    )
    schedule = jnp.asarray(alpha_bar, dtype=images.dtype)
    current_alpha = schedule[reverse]
    clean = images[indices]
    noise = jax.random.normal(noise_key, clean.shape, dtype=clean.dtype)
    noisy = (
        jnp.sqrt(current_alpha)[..., None, None] * clean
        + jnp.sqrt(1.0 - current_alpha)[..., None, None] * noise
    )
    condition = jax.nn.one_hot(labels[indices], CLASSES)
    time = reverse.astype(images.dtype) / steps

    assert noisy.shape == (updates, batch_size, IMAGE, IMAGE)
    assert condition.shape == (updates, batch_size, CLASSES)
    assert time.shape == (updates, batch_size)
    return Struct(
        input=Struct(
            condition=condition,
            sample=noisy,
            time=time,
            alpha_bar=current_alpha,
        ),
        target=noise,
    )


def noise_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Computes unweighted mean squared error between predicted and sampled Gaussian noise."""
    assert prediction.shape == target.shape
    return jnp.mean((prediction - target) ** 2)


def trained_digit_predictor(
    updates: int = UPDATES,
    alpha_bar: tuple[float, ...] = ALPHA_BAR,
) -> tuple[PNode, Aux]:
    """Trains a batched digit noise predictor across random diffusion steps."""
    data = digit_data()
    training = forward_noise_data(
        data.train.images,
        data.train.labels,
        updates,
        BATCH,
        jax.random.PRNGKey(1),
        alpha_bar,
    )
    predictor = digit_predictor()
    model = batch(predictor).with_input(
        tree_first(training.input),
    ).parameterize(rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, noise_mse, optax.adam(LEARNING_RATE))
    final, aux = trained(trainer).apply(
        input=training.input,
        target=training.target,
    )
    return predictor.bind(final.param), aux


def held_out_denoising(
    predictor: PNode,
    rng: jax.Array,
    alpha_bar: tuple[float, ...] = ALPHA_BAR,
) -> Struct:
    """Corrupts held-out digits to the schedule endpoint and reconstructs them with `Denoiser`."""
    data = digit_data()
    labels = data.test.labels[:TEST_IMAGES]
    clean = data.test.images[:TEST_IMAGES]
    condition = jax.nn.one_hot(labels, CLASSES)
    noise = jax.random.normal(rng, clean.shape, dtype=clean.dtype)
    endpoint = jnp.asarray(alpha_bar[-1], dtype=clean.dtype)
    corrupted = (
        jnp.sqrt(endpoint) * clean
        + jnp.sqrt(1.0 - endpoint) * noise
    )
    input = Struct(condition=condition, sample=corrupted)
    denoiser = batch(
        Denoiser(
            predictor,
            alpha_bar,
            clean=PixelRange(),
        ),
    ).with_input(input).parameterize()
    reconstructed = jax.jit(denoiser.apply)(input)

    class_means = jnp.stack(tuple(
        jnp.mean(
            data.train.images[data.train.labels == digit],
            axis=0,
        )
        for digit in range(CLASSES)
    ))
    template = class_means[labels]
    assert reconstructed.shape == clean.shape
    assert jnp.all(jnp.isfinite(reconstructed))
    return Struct(
        labels=labels,
        clean=clean,
        corrupted=corrupted,
        template=template,
        reconstructed=reconstructed,
        noisy_mse=jnp.mean((corrupted - clean) ** 2),
        reconstructed_mse=jnp.mean((reconstructed - clean) ** 2),
        template_mse=jnp.mean((template - clean) ** 2),
    )


def generated_digits(
    predictor: PNode,
    rng: jax.Array,
    alpha_bar: tuple[float, ...],
    per_class: int = 1,
) -> jax.Array:
    """Generates `per_class` digit samples per class from standard normal noise using `Denoiser`.

    Requires a predictor trained on the same near-zero-floor `alpha_bar` schedule so the forward
    process endpoint matches the standard normal prior.
    """
    labels = jnp.tile(jnp.arange(CLASSES), per_class)
    condition = jax.nn.one_hot(labels, CLASSES)
    noise = jax.random.normal(rng, (per_class * CLASSES, IMAGE, IMAGE))
    input = Struct(condition=condition, sample=noise)
    denoiser = batch(
        Denoiser(
            predictor,
            alpha_bar,
            clean=PixelRange(),
        ),
    ).with_input(input).parameterize()
    samples = jax.jit(denoiser.apply)(input)
    assert jnp.all(jnp.isfinite(samples))
    return samples


def render_digits(history: Aux, evaluation: Struct,
                  generated: jax.Array) -> str:
    """Renders clean, corrupted, class-mean, denoised, and generated digits to a PNG grid."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = np.asarray(evaluation.labels)
    selected = np.asarray([
        np.flatnonzero(labels == digit)[0]
        for digit in range(CLASSES)
    ])
    samples = np.asarray(generated).reshape(-1, CLASSES, IMAGE, IMAGE)
    rows = (
        ('clean', np.asarray(evaluation.clean)[selected]),
        ('corrupted', np.asarray(evaluation.corrupted)[selected]),
        ('class mean', np.asarray(evaluation.template)[selected]),
        ('denoised', np.asarray(evaluation.reconstructed)[selected]),
    ) + tuple(
        ('generated', row) for row in samples
    )
    figure, axes = plt.subplots(len(rows), CLASSES,
                                figsize=(11.0, 1.2 * len(rows)))
    for row_index, (row_name, images) in enumerate(rows):
        for digit in range(CLASSES):
            axis = axes[row_index, digit]
            axis.imshow(images[digit], cmap='gray', vmin=-1.0, vmax=1.0)
            axis.set_xticks(())
            axis.set_yticks(())
            if row_index == 0:
                axis.set_title(str(digit))
            if digit == 0:
                axis.set_ylabel(row_name)
    figure.suptitle(
        'Held-out digit denoising  '
        f'loss {float(history.loss[0]):.3f} to {float(history.loss[-1]):.3f}  '
        f'MSE {float(evaluation.noisy_mse):.3f} to '
        f'{float(evaluation.reconstructed_mse):.3f}',
    )
    figure.tight_layout()
    directory = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'diffusion_digits.png')
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def test_denoiser_is_generic_over_sample_pytrees() -> None:
    """Verifies that `Denoiser` preserves arbitrary sample PyTree structure and leaf shapes."""
    predictor = Leaf(
        lambda input: jax.tree.map(jnp.zeros_like, input.sample),
        name='zero_noise',
    )
    input = Struct(
        condition=Struct(label=jnp.asarray(3)),
        sample=Struct(
            image=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
            context=jnp.asarray((1.0, -2.0)),
        ),
    )
    alpha_bar = (1.0, 0.75, 0.5, 0.25)
    output = Denoiser(predictor, alpha_bar).apply(input)
    expected = jax.tree.map(lambda value: 2.0 * value, input.sample)

    assert jax.tree.structure(output) == jax.tree.structure(input.sample)
    assert jax.tree.all(jax.tree.map(jnp.allclose, output, expected))


def test_digit_predictor_has_a_separate_visible_network() -> None:
    """Verifies that `digit_predictor` parameters remain addressable inside `Denoiser`."""
    input = Struct(
        condition=jax.nn.one_hot(3, CLASSES),
        sample=jnp.zeros((IMAGE, IMAGE)),
        time=jnp.asarray(0.5),
        alpha_bar=jnp.asarray(0.6),
    )
    predictor = digit_predictor().with_input(input).parameterize(
        rng=jax.random.PRNGKey(0),
    )
    output = predictor.apply(input)
    sampler_input = Struct(
        condition=input.condition,
        sample=input.sample,
    )
    denoiser = Denoiser(
        predictor,
        ALPHA_BAR[:4],
        clean=PixelRange(),
    ).with_input(sampler_input).parameterize()

    assert output.shape == (IMAGE, IMAGE)
    assert predictor.param.__keys__ == ('hidden1', 'hidden2', 'noise')
    assert predictor.param.noise.w.shape == (HIDDEN, PIXELS)
    assert jnp.array_equal(predictor.param.noise.w, jnp.zeros((HIDDEN, PIXELS)))
    assert denoiser.param.iterations.__keys__ == ('predictor',)
    assert jax.tree.all(jax.tree.map(
        jnp.array_equal,
        denoiser.param.iterations.predictor,
        predictor.param,
    ))


def test_denoises_held_out_digits() -> None:
    """Trains `digit_predictor` and verifies held-out DDIM reconstruction MSE and pixel bounds."""
    predictor, history = trained_digit_predictor()
    evaluation = held_out_denoising(
        predictor,
        jax.random.PRNGKey(2),
    )

    assert jnp.all(jnp.isfinite(history.loss))
    assert jnp.mean(history.loss[-50:]) < 0.7 * jnp.mean(history.loss[:50])
    assert evaluation.reconstructed_mse < 0.4 * evaluation.noisy_mse
    assert evaluation.reconstructed_mse < 0.8 * evaluation.template_mse
    assert jnp.min(evaluation.reconstructed) >= -1.0
    assert jnp.max(evaluation.reconstructed) <= 1.0


if __name__ == '__main__':
    predictor, history = trained_digit_predictor()
    evaluation = held_out_denoising(
        predictor,
        jax.random.PRNGKey(2),
    )

    # Pure-noise generation trains a dedicated predictor on a near-zero alpha floor (`0.02`)
    # matching the standard normal prior without over-amplifying small-alpha prediction error.
    full_schedule = linear_alpha_bar(DIFFUSION_STEPS, floor=0.02)
    generative, generative_history = trained_digit_predictor(
        updates=20 * UPDATES,
        alpha_bar=full_schedule,
    )
    per_class = 10
    generated = generated_digits(
        generative,
        jax.random.PRNGKey(3),
        full_schedule,
        per_class,
    )
    data = digit_data()
    class_means = jnp.stack(tuple(
        jnp.mean(data.train.images[data.train.labels == digit], axis=0)
        for digit in range(CLASSES)
    ))
    distances = jnp.sum(
        (generated[:, None] - class_means[None]) ** 2,
        axis=(-2, -1),
    )
    nearest_mean_accuracy = jnp.mean(
        jnp.argmin(distances, axis=1)
        == jnp.tile(jnp.arange(CLASSES), per_class),
    )

    path = render_digits(history, evaluation, generated[:3 * CLASSES])
    print(
        f'noise loss {float(history.loss[0]):.3f} -> '
        f'{float(history.loss[-1]):.3f}',
    )
    print(
        f'held-out MSE: corrupted {float(evaluation.noisy_mse):.3f}, '
        f'denoised {float(evaluation.reconstructed_mse):.3f}, '
        f'class mean {float(evaluation.template_mse):.3f}',
    )
    print(
        f'generation: noise loss {float(generative_history.loss[-1]):.3f}, '
        f'nearest class mean matches the asked class '
        f'{float(nearest_mean_accuracy):.0%}',
    )
    print(path)
