"""Provides shared policy-plant interaction Nodes for reinforcement learning.

Models each plant as a cyclic Node taking `command` and `disturbance`, returning
`action` and `cost`, and exposing controller observations via `observe`.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode,
    Composite,
    Leaf,
    Node,
    PNode,
    Struct,
    batch,
    drop_aux,
    node,
    scanned,
    tree_broadcast_axis,
    tree_last,
    tree_len,
)


def initial_state(plant: PNode) -> Struct:
    """Evaluates `plant.init` at its default start using a fixed probe key when needed."""
    if plant.contract.init_takes_rng:
        return plant.init(rng=jax.random.PRNGKey(0))
    return plant.init()


def initial_observation(plant: PNode) -> Struct:
    """Returns the observation of `plant` at its default start state for contract resolution."""
    return plant.observe(state=initial_state(plant))


@node
def ControlledStep(policy: BaseNode, plant: BaseNode) -> Node | PNode:
    """Observes `plant`, evaluates `policy`, and advances one closed-loop step.

    Passes `initial_plant_state` as an `apply` input rather than a state-init input so
    boundary re-priming inside a scan can rebuild carried state from the stream's first
    element; subsequent transitions ignore `initial_plant_state`.
    """
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_plant_state):
        """Advances carried plant and policy state by one step."""
        state = self.plant.state
        command = self.policy(self.plant.observe())
        output = self.plant(
            command=command,
            disturbance=disturbance,
        )
        return Struct(
            state=state,
            command=command,
            action=output.action,
            cost=output.cost,
            next_state=self.plant.state,
        )

    def init(self, input):
        """Resets `plant` from `input.initial_plant_state` and primes `policy` on its observation."""
        self.plant.reset(start=input.initial_plant_state)
        self.policy.reset(input=self.plant.observe())

    return members(apply, init=init)


@node
def PlannedStep(policy: BaseNode, plant: BaseNode) -> Node | PNode:
    """Feeds full `plant` state rather than observations into `policy` for model-based planning.

    Preserves the same member names and output fields as `ControlledStep`.
    """
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_plant_state):
        state = self.plant.state
        command = self.policy(state)
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
        self.policy.reset(input=self.plant.state)

    return members(apply, init=init)


@node
def ProposalMean() -> PNode:
    """Extracts the deterministic `mean` command from a Gaussian proposal `Struct`."""
    return Leaf(lambda input: input.mean)


def mean_rollout_program(
    policy: PNode,
    plant: BaseNode,
    n_worlds: int,
) -> PNode:
    """Builds a batched deterministic rollout `PNode` for `n_worlds` parallel worlds."""
    mean_policy = policy >> ProposalMean()
    step = ControlledStep(mean_policy, plant)
    batched = batch(scanned(step), n=n_worlds)
    return batched if isinstance(batched, PNode) else batched.parameterize()


def mean_rollout(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    disturbance: jax.Array,
) -> Struct:
    """Runs a deterministic closed-loop rollout from `starts` under `(world, time)` `disturbance`."""
    n_worlds = tree_len(starts)
    steps = disturbance.shape[1]
    input = Struct(
        disturbance=disturbance,
        initial_plant_state=tree_broadcast_axis(starts, steps, axis=1),
    )
    rollout = mean_rollout_program(policy, plant, n_worlds)
    trajectory = drop_aux(rollout.apply(bundle=input))
    return Struct(
        cost=trajectory.cost,
        state=trajectory.next_state,
        action=trajectory.action,
    )


def policy_trajectory(
    policy: PNode,
    plant: BaseNode,
    initial_plant_state: Struct,
    steps: int,
    rng: jax.Array,
    step: Callable[..., BaseNode] = ControlledStep,
) -> Struct:
    """Evaluates `policy` on `plant` over `steps` while preserving recurrent policy state.

    Returns `(world, time)` trajectories of `command`, `action`, and `cost`, along with
    `state` including the terminal state as one extra step and `final_state`.
    """
    n_worlds = tree_len(initial_plant_state)
    input = Struct(
        disturbance=jnp.zeros((n_worlds, steps)),
        initial_plant_state=tree_broadcast_axis(initial_plant_state, steps, axis=1),
    )
    batched = batch(scanned(step(policy, plant)), n=n_worlds)
    rollout = batched if isinstance(batched, PNode) else batched.parameterize()
    if rollout.contract.apply_takes_rng:
        trajectory = rollout.apply(bundle=input, rng=rng)
    else:
        trajectory = rollout.apply(bundle=input)
    trajectory = drop_aux(trajectory)
    final_state = tree_last(trajectory.next_state, axis=1)
    state = jax.tree.map(
        lambda value, final: jnp.concatenate((value, final[:, None]), axis=1),
        trajectory.state,
        final_state,
    )
    return Struct(
        state=state,
        command=trajectory.command,
        action=trajectory.action,
        cost=trajectory.cost,
        final_state=final_state,
    )


@node
def OpenLoopStep(plant: BaseNode) -> Node | PNode:
    """Advances `plant` by one open-loop step under an explicit `command` and `disturbance`.

    Takes `initial_state` as a state-init input so an enclosing `scanned` transform
    consumes it once alongside the command sequence.
    """
    members = Composite(plant=plant)

    def apply(self, command, disturbance):
        state = self.plant.state
        output = self.plant(command=command, disturbance=disturbance)
        return Struct(state=state, cost=output.cost, next_state=self.plant.state)

    def init(self, initial_state):
        """Binds `plant` state to `initial_state`."""
        self.plant.bind(state=initial_state)

    return members(apply, init=init)


@node
def SamplingStep(policy: BaseNode, plant: BaseNode) -> Node | PNode:
    """Samples a stochastic command from `policy`, steps `plant`, and records replay fields.

    Initializes `plant` and `policy` once from `initial_plant_state` at the start of an
    internalized scan and records `policy_state` before each step so replay can resume
    recurrent policies from any step.
    """
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, rng):
        policy_state = self.policy.state
        observation = self.plant.observe()
        proposal = self.policy(observation)
        drawn = self.policy.sample(proposal, rng=rng.next())
        output = self.plant(command=drawn.command, disturbance=disturbance)
        return Struct(
            observation=observation,
            policy_state=policy_state,
            command=drawn.command,
            logprob=drawn.logprob,
            cost=output.cost,
            next_observation=self.plant.observe(),
        )

    def init(self, initial_plant_state):
        """Resets `plant` from `initial_plant_state` and primes `policy` on its observation."""
        self.plant.reset(start=initial_plant_state)
        self.policy.reset(input=self.plant.observe())

    return members(apply, init=init)
