"""Shared policy-plant interaction Nodes for reinforcement-learning examples.

A plant is a cyclic Node taking ``command`` and ``disturbance`` and returning
a record of ``action`` and ``cost``; its state is its own, read through the
member view before and after a step, and ``observe`` is a method on it.
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
    """The plant's state at its default start, for resolving an input
    contract; a plant whose init draws a key gets a fixed probe key, since
    only the shape is kept."""
    if plant.contract.init_takes_rng:
        return plant.init(rng=jax.random.PRNGKey(0))
    return plant.init()


def initial_observation(plant: PNode) -> Struct:
    """What a controller sees of the plant at its default start, for
    resolving a policy's or a value's input contract."""
    return plant.observe(state=initial_state(plant))


@node
def ControlledStep(policy: Node, plant: PNode) -> Node:
    """Observe, choose a command, and advance one controlled plant step.

    ``initial_plant_state`` rides the apply input rather than the state bundle
    because a boundary re-prime rebuilds the state from the stream's first
    element, so the value has to be in the stream; the step reads it at
    prime and ignores it afterwards. ``SamplingStep``, run fresh per call
    and never re-primed, takes its start as a state input instead.
    """
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_plant_state):
        """Advance carried state; ``initial_plant_state`` is consumed by prime."""
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
        """Start the plant from ``initial_plant_state`` and the policy from
        what it observes there.

        ``initial_plant_state`` shares the apply call because a scan primes from
        its first real element. It is ignored by later transitions. The plant's
        own init turns the start into its state; for a plant whose start is
        its state that is the identity.
        """
        self.plant.reset(start=input.initial_plant_state)
        self.policy.reset(input=self.plant.observe())

    return members(apply, init=init)


@node
def PlannedStep(policy: Node, plant: PNode) -> Node:
    """``ControlledStep`` for a planner: the policy is handed the plant's
    state, not its observation, since a planner rolls the plant's own
    model out from where the plant is. Everything else is the same, member
    names included, so paths into the two steps agree."""
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
def ProposalMean() -> Node:
    """Select the deterministic command from a Gaussian proposal."""
    return Leaf(lambda input: input.mean)


def mean_rollout_program(
    policy: PNode,
    plant: BaseNode,
    n_worlds: int,
) -> PNode:
    """Build a fresh deterministic rollout from an injected policy and plant."""
    mean_policy = policy >> ProposalMean()
    return batch(scanned(ControlledStep(mean_policy, plant)), n=n_worlds).parameterize()


def mean_rollout(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    disturbance: jax.Array,
) -> Struct:
    """Run a deterministic closed loop from the given starts under a
    ``disturbance`` shaped (world, time); the trajectory shares those axes."""
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
    step: Callable = ControlledStep,
) -> Struct:
    """Evaluate from fresh plant state while preserving recurrent policy carry.

    The trajectory is shaped (world, time), its ``state`` holding the final
    state as one extra step, its ``command`` what the policy asked for and
    its ``action`` what the plant did. ``step`` builds the transition from the policy
    and the plant: ``ControlledStep`` for a policy on observations,
    ``PlannedStep`` for a planner on states."""
    n_worlds = tree_len(initial_plant_state)
    input = Struct(
        disturbance=jnp.zeros((n_worlds, steps)),
        initial_plant_state=tree_broadcast_axis(initial_plant_state, steps, axis=1),
    )
    rollout = batch(scanned(step(policy, plant)), n=n_worlds).parameterize()
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
def OpenLoopStep(plant: PNode) -> Node:
    """Advance one plant step under a stored command; no controller.

    The plant state a rollout starts from is a state input, ``initial_state``,
    so a run internalized by ``scanned`` takes it once beside the commands.
    """
    members = Composite(plant=plant)

    def apply(self, command, disturbance):
        state = self.plant.state
        output = self.plant(command=command, disturbance=disturbance)
        return Struct(state=state, cost=output.cost, next_state=self.plant.state)

    def init(self, initial_state):
        """Start from the given plant state."""
        self.plant.bind(state=initial_state)

    return members(apply, init=init)


@node
def SamplingStep(policy: Node, plant: Node) -> Node:
    """One on-policy transition: sample, act, record what replay needs.

    ``initial_plant_state`` starts one sampled run. An enclosing internalized
    scan consumes it once to initialize the plant and a fresh policy lifecycle,
    then carries both states through the run. ``policy_state`` is the policy's
    state as this step saw it, so replay can resume a recurrent policy from any
    recorded step.
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
        """Start the plant from ``initial_plant_state`` and the policy from
        what it observes there."""
        self.plant.reset(start=initial_plant_state)
        self.policy.reset(input=self.plant.observe())

    return members(apply, init=init)
