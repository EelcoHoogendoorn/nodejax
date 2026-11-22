# Energy-Based Reasoners

This directory implements Energy-Based Reasoners (EBRs) and state-based Explorative Modeling (SXM) from William Xingxu Chen, Alexi Gladstone, Zeyu Michael Li, Heng Ji, and Yilun Du, [Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model](https://arxiv.org/abs/2611.08977) (`arXiv:2611.08977`, November 2026), using `nodejax` composition and transforms.

An EBR replaces a directly learned recurrent transition with a scalar energy `E(z; x)` over a latent token state `z` conditioned on a problem embedding `x`. Each reasoning step applies projected Langevin descent along `-alpha * grad_z E(z; x)` onto the per-token unit-RMS sphere, so the same learned energy drives iterative refinement, provides a tangent-gradient convergence signal for adaptive halting, and ranks independent candidate trajectories in Best-of-N selection without a separate verifier.

## Architecture in `nodejax`

The implementation in [`core.py`](core.py) separates four orthogonal choices into composable `Node`s without mode flags or parameter remapping.

`energy_model` assembles the scalar potential as a `serial` pipe (`backbone >> nn.Linear(width) >> nn.silu >> nn.Linear(1, bias=False) >> activation >> sum`), where `backbone` is a `stack` of `ebr_block` Transformer layers with learned per-head log-space attention temperatures (`TemperedAttention`, with optional 2D RoPE) and SwiGLU blocks. Switching between unconstrained energies (`nn.identity`) and nonnegative per-token energies (`nn.softplus`, used on shortest-path mazes to penalize extra path cells) passes a different activation `Node`.

`EnergyStep(potential)` differentiates its wired energy member directly through `jax.value_and_grad(self.potential)` and advances a single carried `Struct`, while `DirectStep(transition)` applies the backbone directly for looped-recurrence ablations. Both share `Masked(step)`, a four-line `Wrapper` that freezes converged trajectories via `latent + active * (candidate.latent - latent)` with `safe_norm` keeping gradients strictly `0.0` on halted rows.

Truncated backpropagation through time (`descent_segment`) and lowest-energy candidate selection (`best_of_n`) compose directly from standard `nodejax` transforms:

```python
def descent_segment(step, *, warmup, tail):
    return tie(
        serial(
            warmup=detach(repeat(step, n=warmup)),
            cut=cut_latent,
            tail=repeated(remat(step), n=tail),
        ),
        'tail',
        'warmup',
    )

def best_of_n(solver, *, n_samples):
    return serial(
        candidates=batch(solver, n=n_samples),
        select=reduce(lowest_energy),
    )
```

In `descent_segment`, `tie` shares one parameter subtree (`param.segment.tail`) across both phases, `detach` prevents parameter and carry gradients during warmup, `cut_latent` stops gradients on `latent` while recording the post-warmup state for SXM replay and keeping `cond` live for encoder gradients, and `repeated(remat(step), n=tail)` checkpoints and stacks the final `tail` states for supervision.

For multi-solution training, `SXMSegment` shares weights with a `K`-trajectory explorer via `externalize(batch(batch(explore_model, n=K), n=B), field='weights')`, picks the best `(candidate, target)` pair with `select_sxm`, and replays only the differentiated tail from `cut_latent` using `tail_reasoner`. Both `TrainingSegment` and `SXMSegment` track exponential-moving-average weights inside `nn.EMA(tau=ema_decay, warm=True)`, and `carried` compiles the entire multi-round training run into one `jax.jit` call. Because `reasoner`, `tail_reasoner`, `eval_reasoner`, and `eval_direct_reasoner` expose identical `Struct(encode, segment.tail, decode)` parameter trees, evaluation binds the trained EMA parameters directly through `.bind(trained_param)`.

## Running the benchmarks

Install `nodejax` with the `optax` extra (Muon optimization uses `optax.contrib.muon`, available in recent `optax` releases; GPU runs also require a CUDA-enabled JAX install such as `jax[cuda12]`):

```sh
pip install -e ".[optax]"
```

Run the unit and integration suite on CPU from the repository root:

```sh
pytest examples/ebr/test_ebr.py -q
```

Run the small CPU presets (~22k parameters, 5-vertex 3-Coloring and 7x7 Maze):

```sh
python -m examples.ebr.coloring --scale cpu --mode sxm
python -m examples.ebr.maze --scale cpu --mode ebr --activation softplus
```

Reproduce the GPU Graph 3-Coloring ablations (~349k parameters, 8-vertex `--scale gpu` and 10-vertex `--scale gpu10`, with `--mode sxm`, `ebr`, or `direct`):

```sh
python -m examples.ebr.coloring --scale gpu --mode sxm
python -m examples.ebr.coloring --scale gpu --mode ebr
python -m examples.ebr.coloring --scale gpu --mode direct
python -m examples.ebr.coloring --scale gpu10 --mode sxm
python -m examples.ebr.coloring --scale gpu10 --mode direct
```

Reproduce the GPU 7x7 Multi-Route Shortest-Path Maze ablations (~347k parameters, comparing `softplus` energy, `identity` energy, and `direct` recurrence):

```sh
python -m examples.ebr.maze --scale gpu --mode ebr --activation softplus
python -m examples.ebr.maze --scale gpu --mode ebr --activation identity
python -m examples.ebr.maze --scale gpu --mode direct
```

Both CLI scripts also accept `--optimizer muon|adamw`, `--rounds N`, and `--seed S`.

## Compute and reference results

The GPU presets target single-GPU reproduction in a few minutes rather than the paper's 3 to 8 hour H200 runs (which train 1.8M to 7.1M parameter models for 30k to 70k steps). At GPU scale (`width = 128`, `depth = 2`, `heads = 4`, `float32`), each model has roughly 347k to 350k parameters and trains well within 8 GB of VRAM. All reference timings and metrics below were measured on a single NVIDIA RTX 4090 (24 GB VRAM) with `seed = 0` on 256 held-out test instances per task.

The table below reports 7x7 Multi-Route Shortest-Path Maze results (2048 training mazes with interior loops, 256 held-out mazes, `N = 5` samples, 1000 rounds, `alpha` in `[180, 240]`, noise `0.4`, `delta = 0.75`, 2D RoPE, Muon with `nn.EMA`):

| Variant | RTX 4090 time | Pass@1 | BoN@5 | Exact BFS | Valid path | Oracle@5 | Mean detour cells |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `direct` | 38.6 s | 91.8% | 91.8% | 89.8% | 94.9% | 95.3% | 0.07 |
| `ebr` (`nn.identity`) | 147.4 s | 91.8% | 95.7% | 93.0% | 100.0% | 98.4% | 0.10 |
| `ebr` (`nn.softplus`) | 148.1 s | 96.9% | 97.7% | 95.7% | 100.0% | 99.6% | 0.06 |

Constraining per-cell energies with `nn.softplus` reduces mean detour cells by 40% relative to `nn.identity` while reaching 100% valid paths and 97.7% BoN@5 shortest-path accuracy (+5.9 percentage points over `direct`).

The table below reports multi-solution Graph 3-Coloring results (4096 training graphs, 256 held-out graphs, `N = 20` samples, 1500 rounds for 8-vertex and 1800 rounds for 10-vertex, `alpha` in `[180, 240]`, noise `0.3`, `delta = 0.25`, Muon with `nn.EMA`):

| Task | Mode | RTX 4090 time | Pass@1 | BoN@20 | Oracle@20 | Cov@20 | `E(valid)` vs. `E(invalid)` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 8-vertex (avg 12.3 sols) | `direct` | 38.7 s | 98.0% | 98.0% | 98.8% | 9.2% | n/a |
| 8-vertex (avg 12.3 sols) | `ebr` (no XM) | 87.4 s | 83.2% | 90.2% | 97.3% | 11.7% | -11.82 vs. -11.65 |
| 8-vertex (avg 12.3 sols) | `sxm` (`K = 2`) | 130.9 s | 90.6% | 96.9% | 99.6% | 35.3% | -14.01 vs. -13.59 |
| 10-vertex (avg 19.0 sols) | `direct` | 47.8 s | 95.3% | 95.3% | 99.2% | 6.8% | n/a |
| 10-vertex (avg 19.0 sols) | `sxm` (`K = 2`) | 166.4 s | 94.9% | 98.8% | 99.6% | 31.9% | -18.51 vs. -17.66 |

Two regimes distinguish these runs from smaller toy settings. Large step sizes (`alpha` in `[180, 240]`) paired with `sigma` in `[0.3, 0.4]`, Muon on hidden segment matrices, and `nn.EMA` evaluation weights drive projected steepest descent to sharp fixed points (final residuals `0.005` to `0.015`, 98.6% to 100% evaluation halting). Defaults keep `pure_noise_prob = 1.0` because Eq. (8) noisy solution projection produced high initial logits at this model width and degraded Coloring accuracy; `noisy_solution_latent` remains implemented and tested in `core.py` for calibrated experiments.

## Files

The table below summarizes the five Python modules in this directory:

| File | Role |
| :--- | :--- |
| [`core.py`](core.py) | `TemperedAttention`, `ebr_block`, `energy_model`, `EnergyStep`, `DirectStep`, `Masked`, `descent_segment`, `reasoner`, `tail_reasoner`, `eval_reasoner`, `best_of_n`, `TrainingSegment`, `SXMSegment`, and `ebr_muon_optimizer`. |
| [`coloring.py`](coloring.py) | Multi-solution Graph k-Coloring dataset generator, `carried` training program (`sxm`, `ebr`, `direct`), evaluation harness, and `cpu` / `gpu` / `gpu10` CLI presets. |
| [`maze.py`](maze.py) | Multi-route Shortest-Path Maze generator with interior loops, 2D RoPE assembly, BFS shortest-path and detour evaluator, and `cpu` / `gpu` CLI presets. |
| [`test_ebr.py`](test_ebr.py) | Unit and integration tests covering activation polymorphism, bit-exact truncated BPTT isolation, zero-masked halting, SXM selection, `stablemax_xent` stability, Muon routing, and both benchmarks. |
| [`__init__.py`](__init__.py) | Package exports from `core.py`. |
