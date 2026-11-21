"""Implements Energy-Based Reasoners (Chen et al., 2026) with `nodejax` transforms.

Separates four orthogonal choices: the scalar energy model (`energy_model` built as a
`serial` pipe over a caller-supplied `backbone` and token activation), the single-step
latent transition (`EnergyStep` along `-grad_z E(z; x)` or `DirectStep` for direct
looped recurrence), zero-masked dynamic halting (`Masked`), and the segment-wise
training regime (`TrainingSegment` for single-trajectory supervision or `SXMSegment`
for state-based explorative modeling over `K` persistent trajectories).

Truncated backpropagation through time (`descent_segment`) is assembled declaratively
from `tie`, `detach`, `repeat`, `at`, `repeated`, and `remat`: `warmup` runs without
parameter or carry gradients, `cut` stops gradients on `latent` while leaving `cond`
live for problem-embedding gradients, and `tail` checkpoints and stacks the final `H`
latent states under one shared parameter tree. `eval_reasoner` mirrors `reasoner`'s
parameter layout so inference binds trained weights directly via `.bind(param)` and
lifts to Best-of-N energy selection with `batch` and `reduce`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from nodejax import (
    Aux,
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    PyTree,
    Struct,
    Wrapper,
    at,
    batch,
    detach,
    externalize,
    nn,
    node,
    reduce,
    remat,
    repeat,
    repeated,
    residual,
    serial,
    tie,
    train_step,
    tree_stop_gradient,
)


def rms_project(latent: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Projects each token vector along the trailing axis onto the unit-RMS sphere."""
    mean_square = jnp.mean(jnp.square(latent), axis=-1, keepdims=True)
    return latent * jax.lax.rsqrt(mean_square + eps)


def safe_norm(
    array: jax.Array,
    axis: int | tuple[int, ...] = (-2, -1),
    eps: float = 1e-12,
) -> jax.Array:
    """Computes `sqrt(sum(array ** 2) + eps)` with a finite zero gradient at the origin."""
    return jnp.sqrt(jnp.sum(jnp.square(array), axis=axis) + eps)


def tangent_grad_norm(
    latent: jax.Array,
    grad: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Computes the Frobenius norm of `grad` projected tangent to the per-token unit-RMS sphere."""
    radial = (
        jnp.sum(grad * latent, axis=-1, keepdims=True)
        / jnp.maximum(jnp.sum(jnp.square(latent), axis=-1, keepdims=True), eps)
    )
    return safe_norm(grad - radial * latent, axis=(-2, -1))


def stablemax_xent(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Computes token-averaged stablemax cross-entropy against integer `targets`."""
    # Double-where keeps the negative branch denominator >= 1 so reverse-mode AD never hits 1 / 0 at logits == 1.
    safe_negative = jnp.where(logits < 0.0, logits, 0.0)
    scaled = jnp.where(logits >= 0.0, logits + 1.0, 1.0 / (1.0 - safe_negative))
    log_probs = jnp.log(scaled) - jnp.log(jnp.sum(scaled, axis=-1, keepdims=True))
    picked = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked, axis=-1)


def convergence_residual(
    latent_tail: jax.Array,
    window: int = 2,
    eps: float = 1e-12,
) -> jax.Array:
    """Averages relative step changes `||z_t - z_{t-1}|| / ||z_{t-1}||` over the last `window` tail transitions."""
    diffs = latent_tail[..., -window:, :, :] - latent_tail[..., -window - 1:-1, :, :]
    previous = latent_tail[..., -window - 1:-1, :, :]
    return jnp.mean(safe_norm(diffs, eps=eps) / safe_norm(previous, eps=eps), axis=-1)


def lowest_energy(candidates: Struct, axis: int = 0) -> Struct:
    """Selects the candidate record along `axis` with the minimum scalar `energy`."""
    best = jnp.argmin(candidates.energy, axis=axis)
    return jax.tree.map(lambda leaf: leaf[best], candidates)


def select_sxm(
    end_logits: jax.Array,
    targets: jax.Array,
    valid_mask: jax.Array,
    used: jax.Array,
    reuse_threshold: float = 0.75,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Picks the lowest-loss `(candidate, target)` pair per row while deferring already-used targets until `reuse_threshold` coverage."""
    cross_entropy = stablemax_xent(end_logits[:, :, None, :, :], targets[:, None, :, :])
    valid_count = jnp.maximum(
        jnp.sum(valid_mask.astype(jnp.float32), axis=-1, keepdims=True),
        1.0,
    )
    used_fraction = (
        jnp.sum((used & valid_mask).astype(jnp.float32), axis=-1, keepdims=True)
        / valid_count
    )
    blocked = (~valid_mask) | (used & (used_fraction < reuse_threshold))
    masked_ce = jnp.where(blocked[:, None, :], jnp.inf, cross_entropy)
    flat_index = jnp.argmin(masked_ce.reshape(cross_entropy.shape[0], -1), axis=-1)
    num_targets = targets.shape[1]
    best_candidate = flat_index // num_targets
    best_target = flat_index % num_targets
    next_used = used.at[jnp.arange(cross_entropy.shape[0]), best_target].set(True)
    return best_candidate, best_target, next_used


def ebr_block(width: int, heads: int, ratio: int) -> Node:
    """Assembles a post-norm Transformer block with residual self-attention and SwiGLU."""
    return serial(
        attn=residual(nn.Attention(heads)) >> nn.RMSNorm(),
        mlp=residual(nn.SwiGLU(width, ratio)) >> nn.RMSNorm(),
    )


def energy_model(
    backbone: BaseNode,
    width: int,
    activation: BaseNode = nn.identity,
) -> Node:
    """Builds scalar energy `sum_i activation(w^T SiLU(W backbone(tokens)_i + b))` as a `serial` pipe."""
    return serial(
        backbone=backbone,
        head=nn.Linear(width) >> nn.silu >> nn.Linear(1, bias=False) >> activation,
        total=Leaf(lambda input: jnp.sum(input), name='sum'),
    )


@node
def EnergyStep(potential: BaseNode) -> Node | PNode:
    """Applies one projected Langevin descent step on `potential(noisy + input.cond)` and sows `energy` and `grad_norm`."""
    def apply(self, input, rng) -> tuple[Struct, Aux]:
        noisy = rms_project(input.latent) + input.noise * jax.random.normal(
            rng.next(), input.latent.shape,
        )
        # Lambda avoids `inspect` dunder probes on the lazy-shape member proxy before first call.
        energy, grad = jax.value_and_grad(lambda tokens: self.potential(tokens))(
            noisy + input.cond,
        )
        next_latent = rms_project(noisy - input.alpha * grad)
        return input.replace(latent=next_latent), Aux(
            energy=energy,
            grad_norm=tangent_grad_norm(noisy, grad),
        )

    return Wrapper(potential=potential)(apply)


@node
def DirectStep(transition: BaseNode) -> Node | PNode:
    """Applies one direct projected recurrence step `transition(noisy + input.cond)` without an energy gradient."""
    def apply(self, input, rng) -> Struct:
        noisy = rms_project(input.latent) + input.noise * jax.random.normal(
            rng.next(), input.latent.shape,
        )
        next_latent = rms_project(self.transition(noisy + input.cond))
        return input.replace(latent=next_latent)

    return Wrapper(transition=transition)(apply)


@node
def Masked(step: BaseNode) -> Node | PNode:
    """Freezes `input.latent` via a zero-masked residual update whenever `input.active` is zero."""
    def apply(self, input) -> Struct:
        candidate = self.step(input)
        return candidate.replace(
            latent=input.latent + input.active * (candidate.latent - input.latent),
        )

    return Wrapper(step=step)(apply)


@node
def Encode(embed: BaseNode) -> Node | PNode:
    """Applies `embed` to `input.problem` and attaches the result as `input.cond`."""
    def apply(self, input) -> Struct:
        return input.replace(cond=self.embed(input.problem))

    return Wrapper(embed=embed)(apply)


@node
def Decode(readout: BaseNode) -> Node | PNode:
    """Applies `readout` to `input.latent` and attaches class predictions as `input.logits`."""
    def apply(self, input) -> Struct:
        return input.replace(logits=self.readout(input.latent))

    return Wrapper(readout=readout)(apply)


@node
def EvalCheck(potential: BaseNode, *, tau: float) -> Node | PNode:
    """Evaluates clean endpoint energy and tangent gradient norm, zero-masking `input.active` once `grad_norm < tau`."""
    def apply(self, input) -> Struct:
        energy, grad = jax.value_and_grad(lambda tokens: self.potential(tokens))(
            input.latent + input.cond,
        )
        grad_norm = tangent_grad_norm(input.latent, grad)
        next_active = input.active * (grad_norm >= tau).astype(input.active.dtype)
        return input.replace(active=next_active, energy=energy, grad_norm=grad_norm)

    return Wrapper(potential=potential)(apply)


stop_latent = at(
    Leaf(lambda input: tree_stop_gradient(input), name='stop_grad'),
    'latent',
)


def descent_segment(step: BaseNode, *, warmup: int, tail: int) -> Node:
    """Runs `warmup` detached steps followed by `tail` checkpointed, differentiated steps under one tied parameter tree."""
    return tie(
        serial(
            warmup=detach(repeat(step, n=warmup)),
            cut=stop_latent,
            tail=repeated(remat(step), n=tail),
        ),
        'tail',
        'warmup',
    )


def reasoner(
    embed: BaseNode,
    step: BaseNode,
    readout: BaseNode,
    *,
    warmup: int,
    tail: int,
) -> Node:
    """Composes problem encoding, truncated descent `segment`, and tail `readout` into one acyclic per-sample `Node`."""
    return serial(
        encode=Encode(embed),
        segment=descent_segment(step, warmup=warmup, tail=tail),
        decode=Decode(batch(readout, n=tail)),
    )


def eval_reasoner(
    embed: BaseNode,
    potential: BaseNode,
    readout: BaseNode,
    *,
    steps: int,
    segments: int,
    tau: float,
) -> Node:
    """Assembles multi-segment inference with tangent-gradient halting and the same parameter layout as `reasoner`."""
    step = Masked(EnergyStep(potential))
    one_segment = tie(
        serial(
            tail=repeat(step, n=steps),
            check=EvalCheck(potential, tau=tau),
        ),
        'tail',
        'check',
    )
    return serial(
        encode=Encode(embed),
        segment=repeat(one_segment, n=segments),
        decode=Decode(readout),
    )


def eval_direct_reasoner(
    embed: BaseNode,
    transition: BaseNode,
    readout: BaseNode,
    *,
    steps: int,
) -> Node:
    """Assembles fixed-budget direct looped inference with the same parameter layout as `reasoner`."""
    step = Masked(DirectStep(transition))
    return serial(
        encode=Encode(embed),
        segment=serial(tail=repeat(step, n=steps)),
        decode=Decode(readout),
    )


def best_of_n(solver: BaseNode, *, n_samples: int) -> Node | PNode:
    """Runs `n_samples` independent trajectories per puzzle and selects the lowest-energy solution."""
    return serial(
        candidates=batch(solver, n=n_samples),
        select=reduce(lowest_energy),
    )


def ebr_loss(
    *,
    delta: float = 0.05,
    kappa: float = 0.05,
    conv_weight: float = 0.1,
) -> Callable:
    """Builds the linearly weighted tail stablemax cross-entropy plus convergence-supervision objective."""
    def loss_fn(trajectory: Struct, target: jax.Array) -> tuple[jax.Array, Aux]:
        active = trajectory.active[:, 0]
        tail_steps = trajectory.logits.shape[1]
        step_weights = jnp.arange(1, tail_steps + 1, dtype=jnp.float32)
        step_weights = step_weights / jnp.sum(step_weights)
        step_ce = stablemax_xent(trajectory.logits, target[:, None, :])
        cross_entropy = jnp.sum(step_ce * step_weights[None, :], axis=-1)
        residual_val = convergence_residual(trajectory.latent)
        correct = jnp.all(
            jnp.argmax(trajectory.logits[:, -1], axis=-1) == target,
            axis=-1,
        ).astype(jnp.float32)
        conv_loss = optax.sigmoid_binary_cross_entropy(
            (delta - residual_val) / kappa,
            correct,
        )
        per_row = active * (cross_entropy + conv_weight * conv_loss)
        denom = jnp.maximum(jnp.sum(active), 1.0)
        return jnp.sum(per_row) / denom, Aux(
            ce=jnp.mean(cross_entropy),
            residual=jnp.mean(residual_val),
            acc=jnp.mean(correct),
        )

    return loss_fn


@node
def TrainingSegment(trainer: BaseNode, *, delta: float) -> Node:
    """Advances `trainer` on one segment, carries detached final `latent`, and zero-masks converged rows."""
    def apply(self, input) -> tuple[Struct, Aux]:
        trajectory = self.trainer(input=input.state, target=input.target)
        residual_val = convergence_residual(trajectory.latent)
        next_active = input.state.active * (residual_val >= delta).astype(
            input.state.active.dtype,
        )
        next_state = input.state.replace(
            latent=tree_stop_gradient(trajectory.latent[:, -1]),
            active=next_active,
        )
        return input.replace(state=next_state), Aux(
            residual=jnp.mean(residual_val),
            active_frac=jnp.mean(next_active),
        )

    return Wrapper(trainer=trainer)(apply)


@node
def SXMSegment(explorer: BaseNode, trainer: BaseNode, *, delta: float) -> Node:
    """Explores `K` detached trajectories via `explorer`, trains `trainer` on the best `(candidate, target)` pair, and carries all `K` endpoints."""
    def apply(self, input) -> tuple[Struct, Aux]:
        candidates = self.explorer(
            input=input.state,
            weights=tree_stop_gradient(self.trainer.params()),
        )
        best_candidate, best_target, next_used = select_sxm(
            candidates.logits[:, :, -1],
            input.targets,
            input.valid_mask,
            input.used,
        )
        row_indices = jnp.arange(best_candidate.shape[0])
        selected_start = jax.tree.map(
            lambda leaf: leaf[row_indices, best_candidate],
            input.state,
        )
        self.trainer(
            input=selected_start,
            target=input.targets[row_indices, best_target],
        )
        residual_val = convergence_residual(candidates.latent)
        next_active = input.state.active * (residual_val >= delta).astype(
            input.state.active.dtype,
        )
        next_state = input.state.replace(
            latent=tree_stop_gradient(candidates.latent[:, :, -1]),
            active=next_active,
        )
        return input.replace(state=next_state, used=next_used), Aux(
            residual=jnp.mean(residual_val),
            active_frac=jnp.mean(next_active),
        )

    return Composite(explorer=explorer, trainer=trainer)(apply)


def ebr_episode(
    model: Node,
    sample_state: PyTree,
    *,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    batch_size: int,
    segments: int,
    delta: float,
) -> Node:
    """Assembles a multi-segment single-trajectory training episode from a per-sample `model`."""
    batched_sample = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size,) + leaf.shape),
        sample_state,
    )
    batched_model = batch(model, n=batch_size).with_input(batched_sample)
    trainer = train_step(batched_model, loss_fn, optimizer)
    return repeat(TrainingSegment(trainer, delta=delta), n=segments)


def sxm_episode(
    model: Node,
    sample_state: PyTree,
    *,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    batch_size: int,
    n_candidates: int,
    segments: int,
    delta: float,
) -> Node:
    """Assembles a multi-segment SXM training episode sharing one parameter tree between `explorer` and `trainer`."""
    batched_sample = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size,) + leaf.shape),
        sample_state,
    )
    batched_model = batch(model, n=batch_size).with_input(batched_sample)
    trainer = train_step(batched_model, loss_fn, optimizer)
    explorer = externalize(
        batch(batch(model, n=n_candidates), n=batch_size),
        field='weights',
    )
    return repeat(SXMSegment(explorer, trainer, delta=delta), n=segments)
