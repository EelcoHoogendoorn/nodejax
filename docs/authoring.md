# Authoring Nodes: Boundaries and Composition

## Single-`apply` Nodes and horizontal growth

A Node is defined by a single `apply` transition. Parameter shapes, initial state, RNG requirements, and member wiring all derive from tracing how `apply` transforms inputs. Because `batch`, `scan`, `remat`, `cyclic`, and `train_step` act on one transition function, giving every Node a single primary operation makes the entire Node transformable as a unit.

Conventional object-oriented code grows vertically by adding methods to shared mutable state. `nodejax` code grows horizontally by wrapping and connecting fixed `Node`s through `>>`, `Composite`, `Wrapper`, and transforms. Leaf inheritance via `derive` replaces selected hooks on an existing leaf without introducing subclass dispatch, and nothing downstream distinguishes a derived leaf from a fresh `Leaf`.

Named methods (`methods={'observe': ..., 'logprob': ..., 'sample': ...}`) expose read-only views over a Node's parameters and state for an enclosing `Composite` or `Wrapper` to call inside its own `apply`:

```python
@node
def CommandLikelihood(policy: Node) -> Node:
    def apply(self, observation, command):
        proposal = self.policy(observation)
        return Struct(
            logprob=self.policy.logprob(proposal, command),
            entropy=self.policy.entropy(proposal),
        )

    return Wrapper(policy=policy)(apply)
```

Transforms act on `apply` and ignore auxiliary methods. Whenever an operation such as policy log-probability evaluation needs to be mapped by `batch`, unrolled by `scan`, or differentiated by `train_step`, wrapping the method call in a `Wrapper` turns that operation into a Node's `apply`.

Across domains, programs differ in how state ownership and operations partition across sub-Nodes:

| Domain | State ownership | Composition pattern | Primary structural yield |
| :--- | :--- | :--- | :--- |
| Control and block-diagram systems (`nodejax.control`, `examples/actuator/`) | Small private carries per block | `>>`, multi-rate `Composite`s, `Delay`, and `closed_loop` | Hierarchical `state` composition without manual carry plumbing, plus unified `param` trees for sweeps, evolution, and `grad` |
| Physics simulation (`examples/pbd/`) | Shared multi-body record threaded across sub-passes | Acyclic `batch` / `repeat` passes closed via outer `cyclic(...)` | Gather-scatter solver passes closed into a `cyclic` `Node` that couples directly to actuators and controllers |
| Neural architectures (`nodejax.nn`) | Per-layer carries, caches, and moments | `>>`, `residual`, `stack`, `ensemble`, `tie`, `eval_mode` | High primitive reuse; closure across deep architectural and adaptation stacks |
| Reinforcement learning (`examples/rl/`) | Policy, critic, target, plant, replay `Buffer`, and optimizers | Per-phase `Wrapper` `Node`s plus `externalize` inside an iteration `Composite` | Single immutable experiment `PSNode`, zero-edit MLP/GRU policy polymorphism, and `scan` / `ensemble` over training runs |

## Control and physical simulation

Control systems decompose along physical and signal boundaries, and each cyclic Node owns the state that evolves across its steps. An `Integrator` owns its accumulator, a `Delay` owns its one-step register, and a plant owns its physical coordinates. Closing a unit-feedback loop combines an open-loop pipe with a `Delay` register inside a `Composite`:

```python
@node
def closed_loop(pipe: Node, output_spec: Any = 0.0) -> Node:
    def apply(self, input):
        error = jax.tree.map(jnp.subtract, input, self.last.state)
        out = self.pipe(error)
        self.last(out)
        return out

    last = Delay().with_input(output_spec)
    return Composite(pipe=pipe, last=last)(apply)
```

`closed_loop` assumes nothing about the internal structure of `pipe`. Connecting a learned recurrent committee to a DC motor and unrolling across time and setpoint batches uses stock transforms at the assembly site:

```python
loop = closed_loop(serial(controller=Controller(), motor=Motor(dt=0.05)))
rollout = scanned(loop)
trainer = train_step(batch(rollout), tracking_loss, optax.adam(1e-2), trainable='controller')
```

Physical plant constants that vary after compilation (such as winding resistance, inductance, rotor inertia, or friction swept across `vmap` batches or randomized domains in `examples/actuator/`) belong in `Motor`'s `param` rather than factory `statics`. Passing `trainable='controller'` restricts optimization updates to the controller subtree while keeping plant parameters fixed (or externally swept) inside the same compiled XLA graph.

When multiple substeps update a coupled collection of bodies within a single timestep, individual substeps stay acyclic functions of the body record while the outer solver step closes the state loop with `cyclic`. In position-based dynamics (`examples/pbd/pbd.py`), unconstrained prediction, iterative constraint projection, and velocity recovery all transform a `Particle` record passed as call data, and `cyclic` closes the composed step into a state-passing transition over the particle collection:

```python
def pbd_step(
    constraints: Node | PNode,
    *,
    n_solver_passes: int,
    dt: float,
    velocity_damping: float = 1.0,
) -> Node:
    predict = batch(FreeMotion(dt))
    solver = repeat(constraints, n=n_solver_passes)
    finalize = batch(VelocityUpdate(dt, velocity_damping))
    return cyclic(PBDStep(predict, solver, finalize))
```

Reset lifetimes attach to the Node whose state resets rather than threading reset flags through enclosing constructors. In `observed_loop`, the fed-back plant measurement declares `state_reinit(Delay().with_input(output_spec))` so it clears at episode boundaries, while the adaptive observer's `belief` `Delay` carries across episodes unchanged.

## Neural architectures

Architectural patterns that imperative libraries write as Python `for` loops and skip-connection boilerplate are algebraic expressions over Nodes:

```python
core = (
    nn.Linear(hidden)
    >> stack(residual(nn.GRU(hidden)), n=depth)
    >> nn.Projection()
)
committee = ensemble(core, n=members) >> reduce(jnp.mean)
```

`residual` wraps any shape-preserving Node, `stack` replicates a Node across depth with independent per-layer parameters and state, and `ensemble` replicates across a parallel population axis.

Shared weights are declared on the composed tree with `tie` rather than holding one Python object reference in two places. In `examples/test_char_lm.py`, `ambient` supplies shared dimensions across stock `nn` factories and `tie` binds the output projection to the input embedding matrix so the composite parameter tree stores the matrix once:

```python
def build(vocab: int) -> Node:
    with ambient(vocab=vocab, hidden=48, experts=4):
        pipe = serial(
            embed=nn.Embed(),
            core=stack(residual(nn.RNN()), n=3),
            moe=nn.MoE(),
            unembed=nn.Unembed(),
        )
    return tie(pipe, 'embed', 'unembed')
```

Auxiliary statistics emitted via `Aux` inside deep members (such as `nn.MoE()` load-balance metrics) route automatically under each member's path without altering intermediate call signatures, arriving at the loss function when it declares `aux`.

## Training and nested adaptation

`train_step(model, loss_fn, optimizer, trainable=...)` returns an ordinary cyclic Node whose `param` holds non-optimized parameters alongside initial values of `trainable` parameters (and any optimizer parameters), and whose `state` holds active `trainable` parameters, model state, and optimizer state. Gradient-based Optax updates and gradient-free population optimizers (`examples/test_evolution.py`) satisfy the same contract, and `trained` scans a trainer over a data sequence. Because trainers satisfy the same Node contract as layers and plants, multi-level adaptation stacks trainers inside standard transforms (`examples/third_order_learning.py`):

```python
def predictor() -> Node:
    block = residual(nn.MinGRU(width)) >> nn.BatchNorm(momentum=0.1) >> nn.tanh
    adaptive = next_step_ttt(train_step(block, token_loss, learned_sgd(ttt_rate)))
    recurrent = stack(adaptive, n=depth)
    observed = taps(recurrent >> nn.tanh)
    pipe = Encoder(features, width) >> observed >> Decoder(features, width)
    return tie(pipe, 'encoder', 'decoder')


def third_order_learning() -> Node:
    online = scanned(predictor())
    adapted = finetune(train_step(online, sequence_loss, learned_sgd(finetune_rate)))
    committee = ensemble(remat(adapted), n=members) >> reduce(jnp.mean)
    outer = train_step(batch(committee), sequence_loss, optax.adam(meta_rate))
    return trained(outer)
```

Inner test-time updates, per-episode support fine-tuning, rematerialized ensembles, BatchNorm running moments, and outer Adam meta-updates compose without custom loop plumbing, and exact third-order derivatives flow through the nested transitions.

## Reinforcement learning

Reinforcement learning libraries customarily center on a mutable `Agent` object with `.act()`, `.evaluate()`, and `.update()` methods surrounded by an imperative rollout loop. In `nodejax`, each transformed unit becomes a `Node`, and an assembly function wires complete collaborators into one iteration `Node`.

Policy-plant interaction is a two-member `Composite` (`examples/rl/control.py`). Its authored `init(self, input)` primes the plant from `initial_plant_state` and initializes any recurrent policy state from the plant's initial observation:

```python
@node
def ControlledStep(policy: Node, plant: PNode) -> Node:
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_plant_state):
        state = self.plant.state
        command = self.policy(self.plant.observe())
        output = self.plant(command=command, disturbance=disturbance)
        return Struct(
            state=state,
            command=command,
            action=output.action,
            cost=output.cost,
            next_state=self.plant.state,
        )

    def init(self, input):
        self.plant.reset(start=input.initial_plant_state)
        self.policy.reset(input=self.plant.observe())

    return members(apply, init=init)
```

Batching across parallel environments and unrolling across chunk horizons uses stock transforms:

```python
transition = state_reinit(ControlledStep(policy, plant), boundary='episode')
rollout = batch(scan(transition, n=n_steps_per_chunk), n=n_worlds)
```

Across SHAC, PPO, and SAC (`examples/rl/`), actor updates follow one pattern: `train_step` differentiates a loss over a Node that evaluates the policy on trajectory data, while algorithms differ in how that trajectory Node obtains its data and critic values:

| Algorithm | Differentiated actor Node | Critic / target wiring | Experience source |
| :--- | :--- | :--- | :--- |
| SHAC (`shac.py`) | `externalize(TerminalCritic(rollout, terminal_critic), 'critic')` with `trainable='policy'` | `nn.EMA(tau=..., warm=True)` smooths `critic_trainer.params()` and feeds externalized `critic` inputs | Differentiable live rollout through `plant` |
| PPO (`ppo.py`) | `ChunkReplay(scan(CommandLikelihood(policy)))` batched over minibatch rows and scanned over minibatches and epochs | `externalize(trajectory_value, 'value')` evaluates GAE targets from `critic_trainer.params()` | Recorded rollout chunks shuffled across worlds and chunks |
| SAC (`sac.py`) | `externalize(ValuedReplay(replay, online_critic), 'critic')` | Online `critic_trainer.params()` prices actor actions; `ema_critic` prices soft Bellman targets | Cyclic `Buffer(capacity)` Node storing chunks as ordinary state |

Three idioms keep RL iterations closed under `jit`, `scan`, and `vmap`.

When a sampler or actor rollout evaluates a policy or critic whose weights live inside a sibling trainer's state, `externalize(node, 'critic')` exposes that member's parameter subtree as a call input while keeping the member registered in the tree for state, RNG, and shape resolution.

In chunked recurrent PPO and SAC, collection records `policy.state` at the start of each chunk. During optimization, `ChunkReplay` (`Wrapper(run=run)`) calls `self.run.bind(state=initial)(...)` on each minibatch so every replay starts from its recorded memory under the current weights.

Off-policy storage uses `Buffer(capacity)` (`examples/rl/replay.py`), which holds its ring buffer, write cursor, and fill count as ordinary Node `state`. Calling `self.buffer(rows)` inserts fresh chunks via `apply`, and `self.buffer.sample(count, rng=rng.next())` gathers minibatches through a read-only method, so collection, storage, and gradient updates compile together inside one `scanned` iteration Node.

## Boundary design rules

| Principle | Structural smell | Preferred `nodejax` form |
| :--- | :--- | :--- |
| A Node boundary marks owned parameters, state, a replaceable collaborator, or a transform seam | A `@node` whose `apply` is a single reshape or arithmetic line | Local array math stays a plain Python function inside `apply` or the loss |
| `statics` fix trace-time structure; `param` holds rollout-invariant values varied after `jit` | Passing physical plant constants or filter gains as `@node` statics when sweeping or randomizing them | Declare post-compilation values in `param`, and pass `trainable=...` to `train_step` when optimizing only a controller subtree |
| Standard transforms stay visible at the assembly site | Factory arguments named `n_worlds`, `n_layers`, or `n_passes` that call `batch`, `stack`, or `repeat` internally | `batch`, `scan`, `stack`, `repeat`, and `iterated` wrap collaborators in the assembly function |
| Static configuration belongs on the Node that reads it | A static parameter accepted by an outer factory only to forward it unchanged into a child | Leaves declare their own statics, resolved directly or through `ambient(...)` and `specialize(...)` |
| Higher composites receive complete collaborators and contain only runtime dataflow glue | An iteration `@node` factory that instantiates sub-networks, optimizers, or rollouts from raw scalars | A `snake_case` assembly function (`shac_iteration`) builds collaborators and passes them into the `@node` (`SHACIteration`) |
| Pure functions become `Leaf` Nodes only when piped, transformed, swapped as members, or parameterized by statics | Either a `Leaf` called only from Python, or a closure hiding a static hyperparameter from `tree_view()` | `Leaf` wraps composable blocks (`nn.tanh`, `FloorConstraint(height)`), while local helpers (`td_lambda`) stay plain functions |
| Live weights from sibling trainers enter via `externalize`, and shared static weights use `tie` | Closing over an unregistered Node and calling `.bind(param)` inside `apply`, or hand-assembling deep parameter `Struct` trees | Collaborators register as members wrapped with `externalize(node, field)` or `tie(pipe, a, b)` |
| Single-member behavioral adapters use `Wrapper` | `Composite(x=x)` around a single member, adding an extra nesting key to `param`, `state`, and `aux` | `Wrapper(policy=policy)(apply)` wraps the single member without extra tree nesting (`CommandLikelihood`, `ChunkReplay`, `SampledCommand`) |
| Output post-processing applied once belongs outside via `>>` | Injecting a projection or clamp into a Node only to call it on the `return` line | `core >> Clamp()` pipes output constraints at the call site; projections stay inside only when applied on every inner solver step |
| `tree_view()` matches the block diagram on paper | Printed member paths hide the algorithm's hierarchy, or a `@node` factory dwarfs the assembly function | `tree_view()` mirrors the system diagram, the assembly function holds the architecture, and the largest `apply` stays around 10 to 15 statements |

## Inspecting boundaries with `tree_view()`

Running `tree_view()` on an assembled Node prints its structural hierarchy and `[P]` / `[C]` parameter and state markers before parameter initialization or compilation:

```python
import jax.numpy as jnp
from nodejax import batch, closed_loop, ensemble, nn, reduce, residual, scanned, serial, stack

core = serial(
    in_proj=nn.Linear(8),
    rnn=stack(residual(nn.GRU(8)), n=2),
    readout=nn.Projection(),
)
controller = serial(committee=ensemble(core, n=3), vote=reduce(jnp.mean))
rollout = batch(scanned(closed_loop(serial(controller=controller, plant=nn.Linear(1)))))
print(rollout.with_input(jnp.zeros((4, 16, 1))).tree_view())
```

```text
batch(scanned(closed_loop))
└── sample: scanned(closed_loop) [P]
    └── step: closed_loop [PC]
        ├── pipe: (controller >> plant) [PC]
        │   ├── controller: (committee >> vote) [PC]
        │   │   ├── committee: ensemble((in_proj >> rnn >> readout)) [PC]
        │   │   │   └── member: (in_proj >> rnn >> readout) [PC]
        │   │   │       ├── in_proj: linear [P]
        │   │   │       ├── rnn: stack(res(gru)) [PC]
        │   │   │       │   └── layer: res(gru) [PC]
        │   │   │       │       └── body: gru [PC]
        │   │   │       └── readout: projection [P]
        │   │   └── vote: reduce(mean)
        │   └── plant: linear [P]
        └── last: delay [C]
```

Every population axis, depth stack, residual skip, feedback delay, and time scan appears at its exact path in the hierarchy, matching the parameter and state PyTrees that JAX compiles.

## Further reading

- The [philosophy](philosophy.md) document explains why JAX primitives lead to single-`apply` FOOP and unified state contracts.
- The [cookbook](cookbook.md) walks step by step from authoring a minimal leaf to composing trainers and control loops.
- The [handbook](handbook.md) specifies every T1 and T2 primitive, transform, `nn` layer, and `control` block.
- The [design document](design.md) covers internal `Def` lowering and T3 transform authoring.
