"""Trains and evaluates Energy-Based Reasoners on multi-solution Graph k-Coloring.

Each graph instance supplies one token per vertex carrying its adjacency row together
with a one-hot indicator for one pinned seed vertex, and pairs that input with up to
`max_solutions` enumerated valid colorings. Because every k-colorable graph admits
multiple valid completions, the task exercises single-trajectory energy descent
(`ebr`), tail-anchored state-based explorative modeling (`sxm`), direct looped
recurrence (`direct`), shared-per-puzzle noisy solution initialization, EMA evaluation
weights, zero-masked halting, and energy-ranked Best-of-N selection under one shared
assembly.

`CPU_CONFIG` solves 5-vertex 3-coloring on a laptop CPU in seconds, while `GPU_CONFIG`
(8-vertex) and `GPU10_CONFIG` (10-vertex) scale the same composition to a single GPU.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (
    BaseNode,
    Node,
    PyTree,
    Struct,
    Wrapper,
    batch,
    carried,
    nn,
    node,
    split_aux,
    stack,
    tree_stop_gradient,
)
from examples.ebr.core import (
    DirectStep,
    EnergyStep,
    Masked,
    best_of_n,
    ebr_block,
    ebr_episode,
    ebr_loss,
    ebr_muon_optimizer,
    energy_model,
    eval_direct_reasoner,
    eval_reasoner,
    noisy_solution_latent,
    reasoner,
    rms_project,
    sxm_episode,
    tail_reasoner,
)


CPU_CONFIG = Struct(
    vertices=5,
    colors=3,
    edge_prob=0.72,
    max_solutions=8,
    train_graphs=280,
    test_graphs=80,
    width=32,
    heads=4,
    mlp_ratio=2,
    depth=2,
    warmup=3,
    tail=4,
    segments=2,
    eval_steps=7,
    eval_segments=4,
    n_candidates=2,
    batch_size=32,
    rounds=600,
    alpha_min=1.2,
    alpha_max=1.8,
    eval_alpha=1.5,
    noise=0.05,
    pure_noise_prob=1.0,
    delta=0.12,
    kappa=0.05,
    conv_weight=0.1,
    tau=0.2,
    ema_decay=0.99,
    optimizer='adamw',
    learning_rate=2e-3,
    weight_decay=1e-3,
    eval_samples=8,
)

GPU_CONFIG = Struct(
    vertices=8,
    colors=3,
    edge_prob=0.58,
    max_solutions=16,
    train_graphs=4096,
    test_graphs=256,
    width=128,
    heads=4,
    mlp_ratio=2,
    depth=2,
    warmup=8,
    tail=7,
    segments=3,
    eval_steps=15,
    eval_segments=4,
    n_candidates=2,
    batch_size=256,
    rounds=1500,
    alpha_min=180.0,
    alpha_max=240.0,
    eval_alpha=220.0,
    noise=0.3,
    pure_noise_prob=1.0,
    delta=0.25,
    kappa=0.05,
    conv_weight=0.25,
    tau=0.2,
    ema_decay=0.995,
    optimizer='muon',
    learning_rate=1e-3,
    weight_decay=1e-2,
    eval_samples=20,
)

GPU10_CONFIG = GPU_CONFIG.replace(
    vertices=10,
    edge_prob=0.50,
    max_solutions=24,
    rounds=1800,
)


def enumerate_colorings(
    adjacency: np.ndarray,
    num_colors: int,
    *,
    pinned: tuple[int, int] | None = None,
    limit: int = 16,
) -> list[np.ndarray]:
    """Enumerates up to `limit` valid `num_colors`-colorings of `adjacency`, honoring optional `pinned` `(vertex, color)`."""
    num_vertices = adjacency.shape[0]
    assigned = np.full(num_vertices, -1, dtype=np.int32)
    if pinned is not None:
        assigned[pinned[0]] = pinned[1]
    solutions = []
    neighbors = [np.flatnonzero(adjacency[vertex]) for vertex in range(num_vertices)]

    def backtrack(vertex: int) -> None:
        if len(solutions) >= limit:
            return
        if vertex == num_vertices:
            solutions.append(assigned.copy())
            return
        if assigned[vertex] != -1:
            for neighbor in neighbors[vertex]:
                if assigned[neighbor] == assigned[vertex]:
                    return
            backtrack(vertex + 1)
            return
        used = {assigned[neighbor] for neighbor in neighbors[vertex] if assigned[neighbor] != -1}
        for color in range(num_colors):
            if color not in used:
                assigned[vertex] = color
                backtrack(vertex + 1)
                assigned[vertex] = -1

    backtrack(0)
    return solutions


def make_coloring_dataset(
    num_graphs: int,
    num_vertices: int,
    num_colors: int,
    *,
    edge_prob: float,
    max_solutions: int,
    seed: int = 0,
) -> Struct:
    """Generates `num_graphs` distinct k-colorable graphs paired with padded valid solution sets."""
    rng = np.random.RandomState(seed)
    adjacencies = []
    features = []
    targets = []
    valid_masks = []
    seen = set()

    while len(adjacencies) < num_graphs:
        planted = rng.randint(0, num_colors, size=num_vertices)
        adjacency = np.zeros((num_vertices, num_vertices), dtype=np.float32)
        for src in range(num_vertices):
            for dst in range(src + 1, num_vertices):
                if planted[src] != planted[dst] and rng.rand() < edge_prob:
                    adjacency[src, dst] = 1.0
                    adjacency[dst, src] = 1.0
        if adjacency.sum() < 2 * num_vertices:
            continue
        seed_color = int(planted[0])
        signature = (adjacency.tobytes(), seed_color)
        if signature in seen:
            continue
        solutions = enumerate_colorings(
            adjacency,
            num_colors,
            pinned=(0, seed_color),
            limit=max_solutions,
        )
        if len(solutions) < 2:
            continue
        seen.add(signature)
        vertex_features = np.zeros(
            (num_vertices, num_vertices + 1 + num_colors),
            dtype=np.float32,
        )
        vertex_features[:, :num_vertices] = adjacency
        vertex_features[0, num_vertices] = 1.0
        vertex_features[0, num_vertices + 1 + seed_color] = 1.0

        solution_table = np.zeros((max_solutions, num_vertices), dtype=np.int32)
        solution_mask = np.zeros((max_solutions,), dtype=np.bool_)
        for solution_index, coloring in enumerate(solutions):
            solution_table[solution_index] = coloring
            solution_mask[solution_index] = True

        adjacencies.append(adjacency)
        features.append(vertex_features)
        targets.append(solution_table)
        valid_masks.append(solution_mask)

    return Struct(
        adjacency=jnp.asarray(np.stack(adjacencies)),
        features=jnp.asarray(np.stack(features)),
        targets=jnp.asarray(np.stack(targets)),
        valid_mask=jnp.asarray(np.stack(valid_masks)),
    )


def split_dataset(dataset: Struct, train_graphs: int) -> tuple[Struct, Struct]:
    """Splits `dataset` along axis 0 into `(train_split, test_split)`."""
    train_split = jax.tree.map(lambda leaf: leaf[:train_graphs], dataset)
    test_split = jax.tree.map(lambda leaf: leaf[train_graphs:], dataset)
    return train_split, test_split


def valid_coloring(
    predictions: jax.Array,
    adjacency: jax.Array,
    features: jax.Array,
    *,
    num_vertices: int,
) -> jax.Array:
    """Checks whether `predictions` satisfy all edge inequalities and match the pinned vertex color."""
    same_color = predictions[..., :, None] == predictions[..., None, :]
    conflicts = jnp.sum(adjacency * same_color, axis=(-2, -1)) / 2.0
    pinned_color = jnp.argmax(features[..., 0, num_vertices + 1:], axis=-1)
    return (conflicts == 0) & (predictions[..., 0] == pinned_color)


def solution_coverage(
    predictions: jax.Array,
    targets: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Computes Cov@N (fraction of distinct valid solutions recovered across `N` samples, capped by `N`)."""
    matches = jnp.all(
        predictions[:, :, None, :] == targets[:, None, :, :],
        axis=-1,
    )
    found = jnp.any(matches, axis=1) & valid_mask
    num_samples = predictions.shape[1]
    denominator = jnp.minimum(jnp.sum(valid_mask, axis=-1), num_samples)
    return jnp.sum(found) / jnp.maximum(jnp.sum(denominator), 1)


def coloring_components(config: Struct) -> Struct:
    """Builds the shared `embed`, `backbone`, `potential`, and `readout` Nodes for Graph k-Coloring."""
    embed = nn.Linear(config.width) >> nn.PosEmbed()
    backbone = stack(
        ebr_block(config.width, config.heads, config.mlp_ratio),
        n=config.depth,
    )
    potential = energy_model(backbone, config.width, activation=nn.identity)
    readout = nn.Linear(config.colors, bias=False)
    return Struct(
        embed=embed,
        backbone=backbone,
        potential=potential,
        readout=readout,
    )


def sample_coloring_state(config: Struct, sample_problem: jax.Array) -> Struct:
    """Constructs a single-graph prototype `Struct` used for shape discovery at bind time."""
    return Struct(
        latent=jnp.zeros((config.vertices, config.width), dtype=jnp.float32),
        problem=sample_problem,
        alpha=jnp.asarray(config.eval_alpha, dtype=jnp.float32),
        noise=jnp.asarray(config.noise, dtype=jnp.float32),
        active=jnp.asarray(1.0, dtype=jnp.float32),
    )


@node
def SingleTrajectoryRound(
    episode: BaseNode,
    dataset: Struct,
    config: Struct,
) -> Node:
    """Draws one minibatch of graphs and initial latents and advances a single-trajectory `episode`."""
    num_graphs = dataset.features.shape[0]
    latent_shape = (config.batch_size, config.vertices, config.width)

    def apply(self, tick, rng) -> Struct:
        del tick
        index_key, latent_key, alpha_key = jax.random.split(rng.next(), 3)
        indices = jax.random.randint(index_key, (config.batch_size,), 0, num_graphs)
        targets = dataset.targets[indices, 0]
        readout_weight = tree_stop_gradient(self.episode.layer.trainer.params().decode.w)
        latent = noisy_solution_latent(
            latent_key,
            readout_weight,
            targets,
            latent_shape,
            pure_noise_prob=config.pure_noise_prob,
        )
        alpha = jax.random.uniform(
            alpha_key,
            (config.batch_size,),
            minval=config.alpha_min,
            maxval=config.alpha_max,
        )
        state = Struct(
            latent=latent,
            problem=dataset.features[indices],
            alpha=alpha,
            noise=jnp.full((config.batch_size,), config.noise, dtype=jnp.float32),
            active=jnp.ones((config.batch_size,), dtype=jnp.float32),
        )
        return self.episode(Struct(state=state, target=targets))

    return Wrapper(episode=episode)(apply)


@node
def ExplorativeRound(
    episode: BaseNode,
    dataset: Struct,
    config: Struct,
) -> Node:
    """Draws one minibatch of graphs with `n_candidates` initial latents per graph and advances an SXM `episode`."""
    num_graphs = dataset.features.shape[0]
    feature_width = dataset.features.shape[-1]
    batch_shape = (config.batch_size, config.n_candidates)
    latent_shape = batch_shape + (config.vertices, config.width)

    def apply(self, tick, rng) -> Struct:
        del tick
        index_key, latent_key, alpha_key = jax.random.split(rng.next(), 3)
        indices = jax.random.randint(index_key, (config.batch_size,), 0, num_graphs)
        # Seed each candidate from a distinct valid solution at a shared per-row noise level.
        seed_targets = dataset.targets[indices, :config.n_candidates]
        readout_weight = tree_stop_gradient(self.episode.layer.trainer.params().decode.w)
        latent = noisy_solution_latent(
            latent_key,
            readout_weight,
            seed_targets,
            latent_shape,
            pure_noise_prob=config.pure_noise_prob,
        )
        alpha = jax.random.uniform(
            alpha_key,
            batch_shape,
            minval=config.alpha_min,
            maxval=config.alpha_max,
        )
        problems = jnp.broadcast_to(
            dataset.features[indices, None],
            batch_shape + (config.vertices, feature_width),
        )
        state = Struct(
            latent=latent,
            problem=problems,
            alpha=alpha,
            noise=jnp.full(batch_shape, config.noise, dtype=jnp.float32),
            active=jnp.ones(batch_shape, dtype=jnp.float32),
        )
        return self.episode(Struct(
            state=state,
            targets=dataset.targets[indices],
            valid_mask=dataset.valid_mask[indices],
            used=jnp.zeros((config.batch_size, config.max_solutions), dtype=jnp.bool_),
        ))

    return Wrapper(episode=episode)(apply)


def _build_optimizer(config: Struct) -> optax.GradientTransformation:
    total_updates = max(config.rounds * config.segments, 2)
    warmup_updates = min(100, total_updates // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=warmup_updates,
        decay_steps=total_updates,
        end_value=config.learning_rate * 0.1,
    )
    if config.optimizer == 'muon':
        return ebr_muon_optimizer(schedule, weight_decay=config.weight_decay)
    if config.optimizer == 'adamw':
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=config.weight_decay),
        )
    raise ValueError(f"optimizer must be 'muon' or 'adamw', got {config.optimizer!r}")


def coloring_program(
    components: Struct,
    train_split: Struct,
    config: Struct,
    *,
    mode: str = 'sxm',
) -> Node:
    """Assembles the complete Graph k-Coloring training program (`carried` over `rounds`) for `'sxm'`, `'ebr'`, or `'direct'`."""
    if mode not in ('sxm', 'ebr', 'direct'):
        raise ValueError(f"mode must be 'sxm', 'ebr', or 'direct', got {mode!r}")

    step = (
        Masked(DirectStep(components.backbone))
        if mode == 'direct'
        else Masked(EnergyStep(components.potential))
    )
    explore_model = reasoner(
        components.embed,
        step,
        components.readout,
        warmup=config.warmup,
        tail=config.tail,
    )
    sample_state = sample_coloring_state(config, train_split.features[0])
    optimizer = _build_optimizer(config)
    loss_fn = ebr_loss(
        delta=config.delta,
        kappa=config.kappa,
        conv_weight=config.conv_weight,
    )

    if mode == 'sxm':
        replay_model = tail_reasoner(
            components.embed,
            step,
            components.readout,
            tail=config.tail,
        )
        episode = sxm_episode(
            explore_model,
            replay_model,
            sample_state,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=config.batch_size,
            n_candidates=config.n_candidates,
            segments=config.segments,
            delta=config.delta,
            ema_decay=config.ema_decay,
        )
        round_step = ExplorativeRound(episode, train_split, config)
    else:
        episode = ebr_episode(
            explore_model,
            sample_state,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=config.batch_size,
            segments=config.segments,
            delta=config.delta,
            ema_decay=config.ema_decay,
        )
        round_step = SingleTrajectoryRound(episode, train_split, config)

    return carried(round_step, n=config.rounds)


def train_coloring(
    config: Struct,
    train_split: Struct,
    *,
    mode: str = 'sxm',
    seed: int = 0,
) -> Struct:
    """Runs `coloring_program` in one JIT-compiled call and returns EMA `param`, `raw_param`, `components`, and history."""
    components = coloring_components(config)
    program = coloring_program(components, train_split, config, mode=mode)
    parameter_key = jax.random.fold_in(jax.random.PRNGKey(0), seed)
    training_key = jax.random.fold_in(jax.random.PRNGKey(100), seed)

    bound = program.parameterize(rng=parameter_key)
    final_round, aux = split_aux(
        jax.jit(bound.apply)(jnp.arange(config.rounds), rng=training_key),
    )
    raw_param = final_round.episode.layer.trainer.trained().sample.param
    ema_param = final_round.episode.layer.ema.state
    return Struct(
        param=ema_param,
        raw_param=raw_param,
        components=components,
        history=Struct(
            loss=jnp.mean(aux.trainer.loss, axis=-1),
            residual=aux.residual[:, -1],
            active_frac=aux.active_frac[:, -1],
        ),
    )


def evaluate_coloring(
    trained_param: PyTree,
    components: Struct,
    dataset: Struct,
    config: Struct,
    *,
    mode: str = 'sxm',
    seed: int = 101,
) -> Struct:
    """Evaluates `trained_param` from pure Gaussian noise across `config.eval_samples` independent trajectories per graph."""
    num_graphs = dataset.features.shape[0]
    n_samples = config.eval_samples
    feature_width = dataset.features.shape[-1]
    latent_key, run_key = jax.random.split(jax.random.PRNGKey(seed))

    latent = rms_project(
        jax.random.normal(
            latent_key,
            (num_graphs, n_samples, config.vertices, config.width),
        ),
    )
    problems = jnp.broadcast_to(
        dataset.features[:, None],
        (num_graphs, n_samples, config.vertices, feature_width),
    )
    eval_state = Struct(
        latent=latent,
        problem=problems,
        alpha=jnp.full((num_graphs, n_samples), config.eval_alpha, dtype=jnp.float32),
        noise=jnp.full((num_graphs, n_samples), config.noise, dtype=jnp.float32),
        active=jnp.ones((num_graphs, n_samples), dtype=jnp.float32),
    )

    if mode == 'direct':
        solver = eval_direct_reasoner(
            components.embed,
            components.backbone,
            components.readout,
            steps=config.eval_steps * config.eval_segments,
        ).bind(trained_param)
        batched_solver = batch(batch(solver, n=n_samples), n=num_graphs)
        output = jax.jit(batched_solver)(eval_state, rng=run_key)
        all_predictions = jnp.argmax(output.logits, axis=-1)
        all_valid = valid_coloring(
            all_predictions,
            dataset.adjacency[:, None],
            dataset.features[:, None],
            num_vertices=config.vertices,
        )
        return Struct(
            pass_at_1=jnp.mean(all_valid[:, 0]),
            best_of_n=jnp.mean(all_valid[:, 0]),
            oracle_at_n=jnp.mean(jnp.any(all_valid, axis=-1)),
            coverage=solution_coverage(all_predictions, dataset.targets, dataset.valid_mask),
            valid_energy=jnp.asarray(jnp.nan),
            invalid_energy=jnp.asarray(jnp.nan),
            halted_frac=jnp.asarray(0.0),
        )

    solver = eval_reasoner(
        components.embed,
        components.potential,
        components.readout,
        steps=config.eval_steps,
        segments=config.eval_segments,
        tau=config.tau,
    ).bind(trained_param)
    eval_state = eval_state.replace(
        energy=jnp.zeros((num_graphs, n_samples), dtype=jnp.float32),
        grad_norm=jnp.zeros((num_graphs, n_samples), dtype=jnp.float32),
    )
    batched_bon = batch(best_of_n(solver, n_samples=n_samples), n=num_graphs)
    winner, aux = split_aux(jax.jit(batched_bon)(eval_state, rng=run_key))

    population = aux.select.population
    all_predictions = jnp.argmax(population.logits, axis=-1)
    all_valid = valid_coloring(
        all_predictions,
        dataset.adjacency[:, None],
        dataset.features[:, None],
        num_vertices=config.vertices,
    )
    winner_valid = valid_coloring(
        jnp.argmax(winner.logits, axis=-1),
        dataset.adjacency,
        dataset.features,
        num_vertices=config.vertices,
    )
    energies = population.energy
    valid_energy = jnp.sum(energies * all_valid) / jnp.maximum(jnp.sum(all_valid), 1.0)
    invalid_energy = jnp.sum(energies * (~all_valid)) / jnp.maximum(jnp.sum(~all_valid), 1.0)
    return Struct(
        pass_at_1=jnp.mean(all_valid[:, 0]),
        best_of_n=jnp.mean(winner_valid),
        oracle_at_n=jnp.mean(jnp.any(all_valid, axis=-1)),
        coverage=solution_coverage(all_predictions, dataset.targets, dataset.valid_mask),
        valid_energy=valid_energy,
        invalid_energy=invalid_energy,
        halted_frac=1.0 - jnp.mean(population.active),
    )


def main() -> None:
    """Trains and evaluates Graph k-Coloring at `cpu`, `gpu`, or `gpu10` scale."""
    parser = argparse.ArgumentParser(description='Train an Energy-Based Reasoner on Graph k-Coloring.')
    parser.add_argument('--scale', choices=('cpu', 'gpu', 'gpu10'), default='cpu')
    parser.add_argument('--mode', choices=('sxm', 'ebr', 'direct'), default='sxm')
    parser.add_argument('--optimizer', choices=('muon', 'adamw'), default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    configs = {'cpu': CPU_CONFIG, 'gpu': GPU_CONFIG, 'gpu10': GPU10_CONFIG}
    config = configs[args.scale]
    if args.rounds is not None:
        config = config.replace(rounds=args.rounds)
    if args.optimizer is not None:
        config = config.replace(optimizer=args.optimizer)

    dataset = make_coloring_dataset(
        config.train_graphs + config.test_graphs,
        config.vertices,
        config.colors,
        edge_prob=config.edge_prob,
        max_solutions=config.max_solutions,
        seed=args.seed,
    )
    train_split, test_split = split_dataset(dataset, config.train_graphs)

    start_time = time.perf_counter()
    outcome = train_coloring(config, train_split, mode=args.mode, seed=args.seed)
    jax.block_until_ready(outcome.param)
    elapsed = time.perf_counter() - start_time

    metrics = evaluate_coloring(
        outcome.param,
        outcome.components,
        test_split,
        config,
        mode=args.mode,
        seed=args.seed + 101,
    )
    print(
        f'[{args.scale}/{args.mode}/{config.optimizer}] {config.rounds} rounds in {elapsed:.1f}s | '
        f'loss {float(outcome.history.loss[0]):.3f} -> {float(outcome.history.loss[-1]):.3f}',
    )
    print(
        f'Held-out ({config.test_graphs} graphs, N={config.eval_samples}): '
        f'Pass@1={float(metrics.pass_at_1) * 100:.1f}%, '
        f'Best-of-N={float(metrics.best_of_n) * 100:.1f}%, '
        f'Oracle@N={float(metrics.oracle_at_n) * 100:.1f}%, '
        f'Cov@N={float(metrics.coverage) * 100:.1f}%',
    )
    if args.mode != 'direct':
        print(
            f'Energy separation: E(valid)={float(metrics.valid_energy):.3f} vs '
            f'E(invalid)={float(metrics.invalid_energy):.3f} | '
            f'halted={float(metrics.halted_frac) * 100:.1f}%',
        )


if __name__ == '__main__':
    main()
