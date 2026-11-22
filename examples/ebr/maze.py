"""Trains and evaluates Energy-Based Reasoners on multi-route Shortest-Path Mazes.

Each `side x side` grid is tokenized into `side * side` cells (`0 = wall`, `1 = open`,
`2 = start`, `3 = goal`), and the model predicts the binary shortest-path mask from
start to goal. Knocking down a fraction of interior walls creates loops and longer
detours, directly exercising `energy_model`'s injectable token activation
(`nn.softplus` vs. `nn.identity`, Appendix A.1) alongside energy descent vs. direct
looped recurrence (`ebr` vs. `direct`) and lowest-energy Best-of-N selection.

Evaluation reports both exact BFS-label match and any-shortest-path validity
(Appendix D), plus detour rate on connected non-shortest paths.
"""

from __future__ import annotations

import argparse
from collections import deque
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
    reasoner,
    rms_project,
)


WALL, OPEN, START, GOAL = 0, 1, 2, 3
_DIRECTIONS = ((1, 0), (0, -1), (0, 1), (-1, 0))  # D < L < R < U (Appendix D)

MAZE_CPU_CONFIG = Struct(
    side=7,
    loop_prob=0.18,
    min_distance=7,
    train_mazes=256,
    test_mazes=64,
    width=32,
    heads=4,
    mlp_ratio=2,
    depth=2,
    warmup=3,
    tail=4,
    segments=2,
    eval_steps=7,
    eval_segments=4,
    batch_size=32,
    rounds=400,
    alpha_min=1.5,
    alpha_max=2.5,
    eval_alpha=2.0,
    noise=0.05,
    delta=0.25,
    kappa=0.05,
    conv_weight=0.25,
    tau=0.2,
    ema_decay=0.99,
    optimizer='adamw',
    learning_rate=2e-3,
    weight_decay=1e-3,
    eval_samples=5,
)

MAZE_GPU_CONFIG = Struct(
    side=7,
    loop_prob=0.18,
    min_distance=7,
    train_mazes=2048,
    test_mazes=256,
    width=128,
    heads=4,
    mlp_ratio=2,
    depth=2,
    warmup=4,
    tail=7,
    segments=3,
    eval_steps=11,
    eval_segments=6,
    batch_size=128,
    rounds=1000,
    alpha_min=180.0,
    alpha_max=240.0,
    eval_alpha=220.0,
    noise=0.4,
    delta=0.75,
    kappa=0.05,
    conv_weight=0.25,
    tau=0.2,
    ema_decay=0.995,
    optimizer='muon',
    learning_rate=1e-3,
    weight_decay=1e-2,
    eval_samples=5,
)


def _carve_maze(side: int, loop_prob: float, rng: np.random.RandomState) -> np.ndarray:
    grid = np.zeros((side, side), dtype=np.int32)
    start_row = 2 * rng.randint(0, (side + 1) // 2)
    start_col = 2 * rng.randint(0, (side + 1) // 2)
    grid[start_row, start_col] = OPEN
    stack_cells = [(start_row, start_col)]
    while stack_cells:
        row, col = stack_cells[-1]
        candidates = []
        for delta_row, delta_col in _DIRECTIONS:
            next_row = row + 2 * delta_row
            next_col = col + 2 * delta_col
            if 0 <= next_row < side and 0 <= next_col < side and grid[next_row, next_col] == WALL:
                candidates.append((next_row, next_col, row + delta_row, col + delta_col))
        if not candidates:
            stack_cells.pop()
            continue
        chosen = candidates[rng.randint(0, len(candidates))]
        grid[chosen[2], chosen[3]] = OPEN
        grid[chosen[0], chosen[1]] = OPEN
        stack_cells.append((chosen[0], chosen[1]))

    # Open a fraction of interior walls between open cells so alternate detours exist.
    for row in range(side):
        for col in range(side):
            if grid[row, col] != WALL:
                continue
            open_neighbors = 0
            for delta_row, delta_col in _DIRECTIONS:
                next_row, next_col = row + delta_row, col + delta_col
                if 0 <= next_row < side and 0 <= next_col < side and grid[next_row, next_col] == OPEN:
                    open_neighbors += 1
            if open_neighbors >= 2 and rng.rand() < loop_prob:
                grid[row, col] = OPEN
    return grid


def _bfs_shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    side = grid.shape[0]
    parent = {start: None}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            path = []
            cursor = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return path
        for delta_row, delta_col in _DIRECTIONS:
            next_row, next_col = row + delta_row, col + delta_col
            if (
                0 <= next_row < side
                and 0 <= next_col < side
                and grid[next_row, next_col] != WALL
                and (next_row, next_col) not in parent
            ):
                parent[(next_row, next_col)] = (row, col)
                queue.append((next_row, next_col))
    return None


def make_maze_dataset(
    num_mazes: int,
    side: int,
    *,
    loop_prob: float = 0.18,
    min_distance: int = 7,
    seed: int = 0,
) -> Struct:
    """Generates `num_mazes` distinct `side x side` multi-route mazes paired with BFS shortest-path masks."""
    rng = np.random.RandomState(seed)
    grids = []
    targets = []
    lengths = []
    seen = set()

    while len(grids) < num_mazes:
        base = _carve_maze(side, loop_prob, rng)
        open_cells = list(zip(*np.nonzero(base == OPEN)))
        if len(open_cells) < min_distance + 1:
            continue
        start_idx, goal_idx = rng.choice(len(open_cells), size=2, replace=False)
        start = (int(open_cells[start_idx][0]), int(open_cells[start_idx][1]))
        goal = (int(open_cells[goal_idx][0]), int(open_cells[goal_idx][1]))
        path = _bfs_shortest_path(base, start, goal)
        if path is None or len(path) - 1 < min_distance:
            continue
        grid = base.copy()
        grid[start] = START
        grid[goal] = GOAL
        signature = grid.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        mask = np.zeros((side, side), dtype=np.int32)
        for cell in path:
            mask[cell] = 1
        grids.append(grid.reshape(-1))
        targets.append(mask.reshape(-1))
        lengths.append(len(path))

    return Struct(
        problem=jnp.asarray(np.stack(grids), dtype=jnp.int32),
        target=jnp.asarray(np.stack(targets), dtype=jnp.int32),
        shortest_cells=jnp.asarray(lengths, dtype=jnp.int32),
    )


def split_maze_dataset(dataset: Struct, train_mazes: int) -> tuple[Struct, Struct]:
    """Splits `dataset` along axis 0 into `(train_split, test_split)`."""
    train_split = jax.tree.map(lambda leaf: leaf[:train_mazes], dataset)
    test_split = jax.tree.map(lambda leaf: leaf[train_mazes:], dataset)
    return train_split, test_split


def verify_maze_prediction(
    grid_flat: np.ndarray,
    pred_flat: np.ndarray,
    shortest_cells: int,
    side: int,
) -> tuple[bool, bool, int]:
    """Checks whether `pred_flat` is a valid connected S-to-G path, whether it is shortest, and its detour cells."""
    grid = grid_flat.reshape(side, side)
    pred = pred_flat.reshape(side, side)
    starts = list(zip(*np.nonzero(grid == START)))
    goals = list(zip(*np.nonzero(grid == GOAL)))
    if not starts or not goals:
        return False, False, 0
    start, goal = starts[0], goals[0]
    if pred[start] != 1 or pred[goal] != 1:
        return False, False, 0
    if np.any((pred == 1) & (grid == WALL)):
        return False, False, 0
    # Verify 4-connectivity from start to goal through predicted cells and no disconnected islands.
    visited = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in _DIRECTIONS:
            next_row, next_col = row + delta_row, col + delta_col
            if (
                0 <= next_row < side
                and 0 <= next_col < side
                and pred[next_row, next_col] == 1
                and (next_row, next_col) not in visited
            ):
                visited.add((next_row, next_col))
                queue.append((next_row, next_col))
    total_predicted = int(np.sum(pred == 1))
    is_valid = goal in visited and len(visited) == total_predicted
    is_shortest = is_valid and total_predicted == int(shortest_cells)
    detour = max(total_predicted - int(shortest_cells), 0) if is_valid else 0
    return is_valid, is_shortest, detour


def maze_components(
    config: Struct,
    *,
    activation: BaseNode = nn.softplus,
) -> Struct:
    """Builds the shared `embed`, `backbone`, `potential`, and `readout` Nodes for Shortest-Path Maze with 2D RoPE."""
    embed = nn.OneHot(4) >> nn.Linear(config.width, bias=False) >> nn.RMSNorm()
    backbone = stack(
        ebr_block(
            config.width,
            config.heads,
            config.mlp_ratio,
            grid_shape=(config.side, config.side),
        ),
        n=config.depth,
    )
    potential = energy_model(backbone, config.width, activation=activation)
    readout = nn.Linear(2, bias=False)
    return Struct(
        embed=embed,
        backbone=backbone,
        potential=potential,
        readout=readout,
    )


def sample_maze_state(config: Struct, sample_problem: jax.Array) -> Struct:
    """Constructs a single-maze prototype `Struct` used for shape discovery at bind time."""
    tokens = config.side * config.side
    return Struct(
        latent=jnp.zeros((tokens, config.width), dtype=jnp.float32),
        problem=sample_problem,
        alpha=jnp.asarray(config.eval_alpha, dtype=jnp.float32),
        noise=jnp.asarray(config.noise, dtype=jnp.float32),
        active=jnp.asarray(1.0, dtype=jnp.float32),
    )


@node
def MazeRound(
    episode: BaseNode,
    dataset: Struct,
    config: Struct,
) -> Node:
    """Draws one minibatch of mazes and pure-noise latents and advances `episode` across segments."""
    num_mazes = dataset.problem.shape[0]
    tokens = config.side * config.side

    def apply(self, tick, rng) -> Struct:
        del tick
        index_key, latent_key, alpha_key = jax.random.split(rng.next(), 3)
        indices = jax.random.randint(index_key, (config.batch_size,), 0, num_mazes)
        latent = rms_project(
            jax.random.normal(latent_key, (config.batch_size, tokens, config.width)),
        )
        alpha = jax.random.uniform(
            alpha_key,
            (config.batch_size,),
            minval=config.alpha_min,
            maxval=config.alpha_max,
        )
        state = Struct(
            latent=latent,
            problem=dataset.problem[indices],
            alpha=alpha,
            noise=jnp.full((config.batch_size,), config.noise, dtype=jnp.float32),
            active=jnp.ones((config.batch_size,), dtype=jnp.float32),
        )
        return self.episode(Struct(state=state, target=dataset.target[indices]))

    return Wrapper(episode=episode)(apply)


def maze_program(
    components: Struct,
    train_split: Struct,
    config: Struct,
    *,
    mode: str = 'ebr',
) -> Node:
    """Assembles the Shortest-Path Maze training program (`carried` over `rounds`) for `'ebr'` or `'direct'`."""
    if mode not in ('ebr', 'direct'):
        raise ValueError(f"mode must be 'ebr' or 'direct', got {mode!r}")

    step = (
        Masked(DirectStep(components.backbone))
        if mode == 'direct'
        else Masked(EnergyStep(components.potential))
    )
    model = reasoner(
        components.embed,
        step,
        components.readout,
        warmup=config.warmup,
        tail=config.tail,
    )
    total_updates = max(config.rounds * config.segments, 2)
    warmup_updates = min(100, total_updates // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=warmup_updates,
        decay_steps=total_updates,
        end_value=config.learning_rate * 0.1,
    )
    optimizer = (
        ebr_muon_optimizer(schedule, weight_decay=config.weight_decay)
        if config.optimizer == 'muon'
        else optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=config.weight_decay),
        )
    )
    loss_fn = ebr_loss(
        delta=config.delta,
        kappa=config.kappa,
        conv_weight=config.conv_weight,
    )
    episode = ebr_episode(
        model,
        sample_maze_state(config, train_split.problem[0]),
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=config.batch_size,
        segments=config.segments,
        delta=config.delta,
        ema_decay=config.ema_decay,
    )
    return carried(MazeRound(episode, train_split, config), n=config.rounds)


def train_maze(
    config: Struct,
    train_split: Struct,
    *,
    mode: str = 'ebr',
    activation: BaseNode = nn.softplus,
    seed: int = 0,
) -> Struct:
    """Runs `maze_program` in one JIT-compiled call and returns EMA `param`, `raw_param`, `components`, and history."""
    components = maze_components(config, activation=activation)
    program = maze_program(components, train_split, config, mode=mode)
    parameter_key = jax.random.fold_in(jax.random.PRNGKey(0), seed)
    training_key = jax.random.fold_in(jax.random.PRNGKey(100), seed)

    bound = program.parameterize(rng=parameter_key)
    final_round, aux = split_aux(
        jax.jit(bound.apply)(jnp.arange(config.rounds), rng=training_key),
    )
    return Struct(
        param=final_round.episode.layer.ema.state,
        raw_param=final_round.episode.layer.trainer.trained().sample.param,
        components=components,
        history=Struct(
            loss=jnp.mean(aux.trainer.loss, axis=-1),
            residual=aux.residual[:, -1],
            active_frac=aux.active_frac[:, -1],
        ),
    )


def evaluate_maze(
    trained_param: PyTree,
    components: Struct,
    dataset: Struct,
    config: Struct,
    *,
    mode: str = 'ebr',
    seed: int = 101,
) -> Struct:
    """Evaluates `trained_param` on held-out mazes, reporting exact match, any-shortest-path, valid-path, and detour rates."""
    num_mazes = dataset.problem.shape[0]
    n_samples = config.eval_samples
    tokens = config.side * config.side
    latent_key, run_key = jax.random.split(jax.random.PRNGKey(seed))

    latent = rms_project(
        jax.random.normal(latent_key, (num_mazes, n_samples, tokens, config.width)),
    )
    problems = jnp.broadcast_to(dataset.problem[:, None], (num_mazes, n_samples, tokens))
    eval_state = Struct(
        latent=latent,
        problem=problems,
        alpha=jnp.full((num_mazes, n_samples), config.eval_alpha, dtype=jnp.float32),
        noise=jnp.full((num_mazes, n_samples), config.noise, dtype=jnp.float32),
        active=jnp.ones((num_mazes, n_samples), dtype=jnp.float32),
    )

    if mode == 'direct':
        solver = eval_direct_reasoner(
            components.embed,
            components.backbone,
            components.readout,
            steps=config.eval_steps * config.eval_segments,
        ).bind(trained_param)
        output = jax.jit(batch(batch(solver, n=n_samples), n=num_mazes))(eval_state, rng=run_key)
        all_preds = np.asarray(jnp.argmax(output.logits, axis=-1))
        chosen_preds = all_preds[:, 0]
        halted_frac = 0.0
    else:
        solver = eval_reasoner(
            components.embed,
            components.potential,
            components.readout,
            steps=config.eval_steps,
            segments=config.eval_segments,
            tau=config.tau,
        ).bind(trained_param)
        eval_state = eval_state.replace(
            energy=jnp.zeros((num_mazes, n_samples), dtype=jnp.float32),
            grad_norm=jnp.zeros((num_mazes, n_samples), dtype=jnp.float32),
        )
        winner, aux = split_aux(
            jax.jit(batch(best_of_n(solver, n_samples=n_samples), n=num_mazes))(
                eval_state,
                rng=run_key,
            ),
        )
        all_preds = np.asarray(jnp.argmax(aux.select.population.logits, axis=-1))
        chosen_preds = np.asarray(jnp.argmax(winner.logits, axis=-1))
        halted_frac = float(1.0 - jnp.mean(aux.select.population.active))

    grids_np = np.asarray(dataset.problem)
    targets_np = np.asarray(dataset.target)
    lengths_np = np.asarray(dataset.shortest_cells)

    pass1_shortest = []
    bon_shortest = []
    bon_valid = []
    bon_detours = []
    oracle_shortest = []
    exact_bon = []

    for row in range(num_mazes):
        _, short_first, _ = verify_maze_prediction(
            grids_np[row], all_preds[row, 0], lengths_np[row], config.side,
        )
        valid_best, short_best, detour_best = verify_maze_prediction(
            grids_np[row], chosen_preds[row], lengths_np[row], config.side,
        )
        any_short = any(
            verify_maze_prediction(grids_np[row], all_preds[row, sample_idx], lengths_np[row], config.side)[1]
            for sample_idx in range(n_samples)
        )
        pass1_shortest.append(short_first)
        bon_shortest.append(short_best)
        bon_valid.append(valid_best)
        if valid_best:
            bon_detours.append(detour_best)
        oracle_shortest.append(any_short)
        exact_bon.append(bool(np.array_equal(chosen_preds[row], targets_np[row])))

    return Struct(
        pass_at_1=jnp.asarray(float(np.mean(pass1_shortest))),
        best_of_n=jnp.asarray(float(np.mean(bon_shortest))),
        exact_match=jnp.asarray(float(np.mean(exact_bon))),
        valid_path=jnp.asarray(float(np.mean(bon_valid))),
        oracle_at_n=jnp.asarray(float(np.mean(oracle_shortest))),
        mean_detour=jnp.asarray(float(np.mean(bon_detours)) if bon_detours else 0.0),
        halted_frac=jnp.asarray(halted_frac),
    )


def main() -> None:
    """Trains and evaluates Shortest-Path Maze at `cpu` or `gpu` scale."""
    parser = argparse.ArgumentParser(description='Train an Energy-Based Reasoner on Shortest-Path Mazes.')
    parser.add_argument('--scale', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--mode', choices=('ebr', 'direct'), default='ebr')
    parser.add_argument('--activation', choices=('softplus', 'identity'), default='softplus')
    parser.add_argument('--optimizer', choices=('muon', 'adamw'), default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = MAZE_CPU_CONFIG if args.scale == 'cpu' else MAZE_GPU_CONFIG
    if args.rounds is not None:
        config = config.replace(rounds=args.rounds)
    if args.optimizer is not None:
        config = config.replace(optimizer=args.optimizer)
    activation = nn.softplus if args.activation == 'softplus' else nn.identity

    dataset = make_maze_dataset(
        config.train_mazes + config.test_mazes,
        config.side,
        loop_prob=config.loop_prob,
        min_distance=config.min_distance,
        seed=args.seed,
    )
    train_split, test_split = split_maze_dataset(dataset, config.train_mazes)

    start_time = time.perf_counter()
    outcome = train_maze(config, train_split, mode=args.mode, activation=activation, seed=args.seed)
    jax.block_until_ready(outcome.param)
    elapsed = time.perf_counter() - start_time

    metrics = evaluate_maze(
        outcome.param,
        outcome.components,
        test_split,
        config,
        mode=args.mode,
        seed=args.seed + 101,
    )
    print(
        f'[maze/{args.scale}/{args.mode}/{args.activation}] {config.rounds} rounds in {elapsed:.1f}s | '
        f'loss {float(outcome.history.loss[0]):.3f} -> {float(outcome.history.loss[-1]):.3f}',
    )
    print(
        f'Held-out ({config.test_mazes} mazes, N={config.eval_samples}): '
        f'Pass@1={float(metrics.pass_at_1) * 100:.1f}%, '
        f'BoN(shortest)={float(metrics.best_of_n) * 100:.1f}%, '
        f'Exact={float(metrics.exact_match) * 100:.1f}%, '
        f'ValidPath={float(metrics.valid_path) * 100:.1f}%, '
        f'Oracle@N={float(metrics.oracle_at_n) * 100:.1f}%, '
        f'mean_detour={float(metrics.mean_detour):.2f}, '
        f'halted={float(metrics.halted_frac) * 100:.1f}%',
    )


if __name__ == '__main__':
    main()
