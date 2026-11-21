"""Verifies Energy-Based Reasoner composition, truncated BPTT, zero-mask halting, SXM, and Graph k-Coloring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nodejax import (
    Struct,
    batch,
    nn,
    remat,
    repeat,
    repeated,
    serial,
    split_aux,
    stack,
)
from examples.ebr import (
    Decode,
    Encode,
    EnergyStep,
    Masked,
    best_of_n,
    ebr_block,
    ebr_loss,
    energy_model,
    eval_reasoner,
    reasoner,
    rms_project,
    select_sxm,
    stablemax_xent,
)
from examples.ebr.coloring import (
    CPU_CONFIG,
    evaluate_coloring,
    make_coloring_dataset,
    split_dataset,
    train_coloring,
)


WIDTH, HEADS, RATIO, DEPTH = 8, 2, 2, 1
TOKENS, FEATURE_DIM, CLASSES = 4, 8, 3
WARMUP, TAIL = 2, 3


def _tiny_components(activation=nn.identity) -> Struct:
    embed = nn.Linear(WIDTH) >> nn.PosEmbed()
    backbone = stack(ebr_block(WIDTH, HEADS, RATIO), n=DEPTH)
    potential = energy_model(backbone, WIDTH, activation=activation)
    readout = nn.Linear(CLASSES, bias=False)
    return Struct(embed=embed, backbone=backbone, potential=potential, readout=readout)


def _tiny_state(
    *,
    noise: float = 0.05,
    active: float = 1.0,
) -> Struct:
    return Struct(
        latent=rms_project(jax.random.normal(jax.random.PRNGKey(2), (TOKENS, WIDTH))),
        problem=jax.random.normal(jax.random.PRNGKey(3), (TOKENS, FEATURE_DIM)),
        alpha=jnp.asarray(1.0, dtype=jnp.float32),
        noise=jnp.asarray(noise, dtype=jnp.float32),
        active=jnp.asarray(active, dtype=jnp.float32),
    )


def test_energy_model_activation_polymorphism() -> None:
    """Verifies that swapping token activation `nn.identity` vs. `nn.softplus` preserves parameters and enforces positivity."""
    backbone = stack(ebr_block(WIDTH, HEADS, RATIO), n=DEPTH)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (TOKENS, WIDTH))

    linear_energy = energy_model(backbone, WIDTH, activation=nn.identity).with_input(
        tokens,
    ).parameterize(rng=jax.random.PRNGKey(0))
    positive_energy = energy_model(backbone, WIDTH, activation=nn.softplus).with_input(
        tokens,
    ).parameterize(rng=jax.random.PRNGKey(0))

    linear_leaves = jax.tree.leaves(linear_energy.param)
    positive_leaves = jax.tree.leaves(positive_energy.param)
    assert [leaf.shape for leaf in linear_leaves] == [leaf.shape for leaf in positive_leaves]
    assert linear_energy(tokens).shape == ()
    assert float(positive_energy(tokens)) > 0.0


def test_descent_segment_ties_weights_and_truncates_bptt_exactly() -> None:
    """Verifies single-copy parameter tying and bit-exact gradient isolation across the warmup/tail boundary."""
    components = _tiny_components(activation=nn.softplus)
    step = Masked(EnergyStep(components.potential))
    model = reasoner(
        components.embed,
        step,
        components.readout,
        warmup=WARMUP,
        tail=TAIL,
    )
    state = _tiny_state(noise=0.0)
    bound = model.with_input(state).parameterize(rng=jax.random.PRNGKey(0))
    assert list(bound.param.segment.__keys__) == ['tail']

    target = jnp.arange(TOKENS) % CLASSES

    def full_objective(param) -> jax.Array:
        trajectory, _ = split_aux(model.apply(param, state, rng=jax.random.PRNGKey(5)))
        return jnp.mean(stablemax_xent(trajectory.logits, target[None, :]))

    # Compute reference gradients by running warmup outside `jax.grad` and freezing its endpoint.
    warmup_pipe = serial(encode=Encode(components.embed), warmup=repeat(step, n=WARMUP))
    warmed, _ = split_aux(
        warmup_pipe.apply(
            Struct(encode=bound.param.encode, warmup=bound.param.segment.tail),
            state,
            rng=jax.random.PRNGKey(5),
        ),
    )
    frozen_latent = jax.lax.stop_gradient(warmed.latent)
    tail_pipe = serial(
        encode=Encode(components.embed),
        tail=repeated(remat(step), n=TAIL),
        decode=Decode(batch(components.readout, n=TAIL)),
    )

    def tail_only_objective(param) -> jax.Array:
        tail_param = Struct(encode=param.encode, tail=param.segment.tail, decode=param.decode)
        trajectory, _ = split_aux(
            tail_pipe.apply(
                tail_param,
                state.replace(latent=frozen_latent),
                rng=jax.random.PRNGKey(5),
            ),
        )
        return jnp.mean(stablemax_xent(trajectory.logits, target[None, :]))

    full_grad = jax.grad(full_objective)(bound.param)
    ref_grad = jax.grad(tail_only_objective)(bound.param)
    diffs = jax.tree.map(lambda left, right: float(jnp.max(jnp.abs(left - right))), full_grad, ref_grad)
    assert max(jax.tree.leaves(diffs)) < 1e-6


def test_zero_mask_halting_freezes_state_and_zeros_gradients() -> None:
    """Verifies that `active == 0` freezes `latent` and produces exact zero gradients without NaNs."""
    components = _tiny_components()
    step = Masked(EnergyStep(components.potential))
    model = reasoner(
        components.embed,
        step,
        components.readout,
        warmup=WARMUP,
        tail=TAIL,
    )
    batch_size = 2
    active_sample = _tiny_state(active=1.0)
    halted_sample = _tiny_state(active=0.0)
    batched_active = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size,) + leaf.shape),
        active_sample,
    )
    batched_halted = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size,) + leaf.shape),
        halted_sample,
    )
    batched = batch(model, n=batch_size).with_input(batched_active).parameterize(
        rng=jax.random.PRNGKey(0),
    )
    targets = jnp.zeros((batch_size, TOKENS), dtype=jnp.int32)
    loss_fn = ebr_loss(delta=0.12, kappa=0.05, conv_weight=0.1)

    halted_out, _ = split_aux(batched(batched_halted, rng=jax.random.PRNGKey(1)))
    assert jnp.allclose(halted_out.latent[:, -1], batched_halted.latent)

    def objective(param, batch_state) -> jax.Array:
        trajectory, _ = split_aux(batched.node.apply(param, batch_state, rng=jax.random.PRNGKey(1)))
        loss_val, _ = loss_fn(trajectory, targets)
        return loss_val

    grad_fn = jax.jit(jax.grad(objective))
    active_grad = grad_fn(batched.param, batched_active)
    halted_grad = grad_fn(batched.param, batched_halted)

    for member in ('encode', 'segment', 'decode'):
        active_max = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(getattr(active_grad, member)))
        halted_max = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(getattr(halted_grad, member)))
        assert active_max > 0.0
        assert halted_max == 0.0


def test_eval_reasoner_binds_directly_halts_and_selects_lowest_energy() -> None:
    """Verifies zero-drill `.bind(param)`, tangent-gradient halting, and `best_of_n` minimum-energy selection."""
    components = _tiny_components(activation=nn.softplus)
    step = Masked(EnergyStep(components.potential))
    train_model = reasoner(
        components.embed,
        step,
        components.readout,
        warmup=WARMUP,
        tail=TAIL,
    )
    state = _tiny_state()
    bound = train_model.with_input(state).parameterize(rng=jax.random.PRNGKey(0))

    # A very large `tau` halts trajectories immediately after the first segment.
    halting_solver = eval_reasoner(
        components.embed,
        components.potential,
        components.readout,
        steps=2,
        segments=2,
        tau=1e3,
    ).bind(bound.param)
    eval_state = state.replace(
        energy=jnp.asarray(0.0, dtype=jnp.float32),
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )
    halted_result, _ = split_aux(halting_solver(eval_state, rng=jax.random.PRNGKey(7)))
    assert float(halted_result.active) == 0.0
    assert halted_result.logits.shape == (TOKENS, CLASSES)

    n_samples = 5
    solver = eval_reasoner(
        components.embed,
        components.potential,
        components.readout,
        steps=3,
        segments=2,
        tau=0.01,
    ).bind(bound.param)
    bon = best_of_n(solver, n_samples=n_samples)
    eval_batch = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (n_samples,) + leaf.shape),
        eval_state,
    )
    winner, aux = split_aux(jax.jit(bon)(eval_batch, rng=jax.random.PRNGKey(9)))
    assert winner.logits.shape == (TOKENS, CLASSES)
    assert jnp.isclose(winner.energy, jnp.min(aux.select.population.energy))


def test_stablemax_xent_gradient_is_finite_at_one() -> None:
    """Verifies that reverse-mode AD through `stablemax_xent` stays finite when logits equal `1.0`."""
    logits = jnp.array([[1.0, 0.0, -1.0], [1.0, 1.0, 0.0]], dtype=jnp.float32)
    targets = jnp.array([0, 1], dtype=jnp.int32)
    grad = jax.grad(lambda values: jnp.mean(stablemax_xent(values, targets)))(logits)
    assert jnp.all(jnp.isfinite(grad))


def test_select_sxm_defers_used_solutions_until_threshold() -> None:
    """Verifies that `select_sxm` masks invalid targets and avoids reusing solutions until 75% coverage."""
    # One graph, 2 candidates, 4 targets (3 valid, 1 padded invalid).
    targets = jnp.array([[[0, 0], [1, 1], [2, 2], [0, 0]]], dtype=jnp.int32)
    valid_mask = jnp.array([[True, True, True, False]])
    # Candidate 0 prefers target 0; candidate 1 prefers target 1.
    end_logits = jnp.array([[[
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ], [
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
    ]]], dtype=jnp.float32)

    used_none = jnp.zeros((1, 4), dtype=jnp.bool_)
    cand_first, targ_first, used_one = select_sxm(end_logits, targets, valid_mask, used_none)
    assert int(targ_first[0]) in (0, 1)
    assert bool(used_one[0, int(targ_first[0])])

    # With target 0 already used (1/3 < 0.75), target 0 is deferred in favor of target 1.
    used_zero = jnp.array([[True, False, False, False]])
    cand_second, targ_second, _ = select_sxm(end_logits, targets, valid_mask, used_zero)
    assert int(cand_second[0]) == 1
    assert int(targ_second[0]) == 1

    # Once all 3 valid targets are used (3/3 >= 0.75), targets unlock while the invalid entry stays masked.
    used_all_valid = jnp.array([[True, True, True, False]])
    _, targ_reused, _ = select_sxm(end_logits, targets, valid_mask, used_all_valid)
    assert int(targ_reused[0]) in (0, 1, 2)


@pytest.mark.parametrize('mode', ('sxm', 'ebr', 'direct'))
def test_coloring_trains_and_evaluates_across_modes(mode: str) -> None:
    """Verifies end-to-end Graph 3-Coloring training (`carried`) and evaluation across `sxm`, `ebr`, and `direct`."""
    config = CPU_CONFIG.replace(
        train_graphs=8,
        test_graphs=4,
        width=8,
        heads=2,
        depth=1,
        warmup=1,
        tail=3,
        segments=2,
        eval_steps=2,
        eval_segments=2,
        batch_size=4,
        rounds=2,
        eval_samples=2,
    )
    dataset = make_coloring_dataset(
        config.train_graphs + config.test_graphs,
        config.vertices,
        config.colors,
        edge_prob=config.edge_prob,
        max_solutions=config.max_solutions,
        seed=0,
    )
    train_split, test_split = split_dataset(dataset, config.train_graphs)

    outcome = train_coloring(config, train_split, mode=mode, seed=0)
    assert outcome.history.loss.shape == (config.rounds,)
    assert jnp.all(jnp.isfinite(outcome.history.loss))

    metrics = evaluate_coloring(
        outcome.param,
        outcome.components,
        test_split,
        config,
        mode=mode,
        seed=101,
    )
    assert 0.0 <= float(metrics.pass_at_1) <= 1.0
    assert 0.0 <= float(metrics.best_of_n) <= 1.0
    assert 0.0 <= float(metrics.oracle_at_n) <= 1.0
    assert 0.0 <= float(metrics.coverage) <= 1.0
    if mode != 'direct':
        assert jnp.isfinite(metrics.valid_energy) or jnp.isfinite(metrics.invalid_energy)
