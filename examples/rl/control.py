"""Shared policy-plant interaction Nodes for reinforcement-learning examples."""

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
    tile,
    tree_first,
    tree_last,
    tree_len,
)


@node
def ControlledStep(policy: Node, plant: PNode) -> Node:
    """Observe, choose a command, and advance one controlled plant step.

    ``initial_state`` rides the apply input rather than the state bundle
    because a boundary re-prime rebuilds the state from the stream's first
    element, so the value has to be in the stream; the step reads it at
    prime and ignores it afterwards. ``SamplingStep``, run fresh per call
    and never re-primed, takes its start as a state input instead.
    """
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_state):
        """Advance carried state; ``initial_state`` is consumed by prime."""
        state = self.state.plant
        command = self.policy(self.plant.observe())
        output = self.plant(
            command=command,
            disturbance=disturbance,
        )
        return Struct(
            state=state,
            action=output.action,
            cost=output.cost,
            next_state=output.state,
        )

    def init(param, input):
        """Adopt caller state and initialize policy state from observation.

        ``initial_state`` shares the apply call because a scan primes from its
        first real element. It is ignored by later transitions. A stateless
        policy contributes the empty slot: no fork on its lifecycle.
        """
        observation = plant.observe(state=input.initial_state)
        return Struct(
            policy=policy.bind(param.policy).init(input=observation),
            plant=input.initial_state,
        )

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
    return scanned(
        batch(ControlledStep(mean_policy, plant), n=n_worlds)
    ).parameterize()


def mean_rollout(
    policy: PNode,
    plant: BaseNode,
    starts: Struct,
    disturbance: jax.Array,
) -> Struct:
    """Run a deterministic closed loop from the given starts."""
    n_worlds = tree_len(starts)
    steps = tree_len(disturbance)
    input = Struct(
        disturbance=disturbance,
        initial_state=tile(starts, steps),
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
    initial_state: Struct,
    steps: int,
    rng: jax.Array,
) -> Struct:
    """Evaluate from fresh state while preserving recurrent carry."""
    n_worlds = tree_len(initial_state)
    input = Struct(
        disturbance=jnp.zeros((steps, n_worlds)),
        initial_state=tile(initial_state, steps),
    )
    world = batch(
        ControlledStep(policy, plant),
        n=n_worlds,
    ).parameterize()
    init_key, apply_key = jax.random.split(rng)
    rollout = (world.initialize(input=tree_first(input), rng=init_key)
               if world.contract.init_takes_rng else
               world.initialize(input=tree_first(input)))
    _, trajectory = (rollout.scan(bundle=input, rng=apply_key)
                     if rollout.contract.apply_takes_rng else
                     rollout.scan(bundle=input))
    trajectory = drop_aux(trajectory)
    final_state = tree_last(trajectory.next_state)
    state = jax.tree.map(
        lambda value, final: jnp.concatenate((value, final[None]), axis=0),
        trajectory.state,
        final_state,
    )
    return Struct(
        state=state,
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
        state = self.state.plant
        output = self.plant(command=command, disturbance=disturbance)
        return Struct(state=state, cost=output.cost, next_state=output.state)

    def init(param, initial_state):
        """Start from the given plant state."""
        return Struct(plant=initial_state)

    return members(apply, init=init)


@node
def SamplingStep(policy: Node, plant: Node) -> Node:
    """One on-policy transition: sample, act, record what replay needs.

    The plant state a rollout starts from is a state input, ``initial_state``,
    so a run internalized by ``scanned`` takes it once beside the sequence.
    ``policy_state`` is the policy's state as this step saw it, so a replay
    can resume a recurrent policy from any recorded step.
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

    def init(param, initial_state):
        """Start from the given plant state; a stateless policy contributes
        the empty slot."""
        observation = plant.observe(state=initial_state)
        return Struct(
            policy=policy.bind(param.policy).init(input=observation),
            plant=initial_state,
        )

    return members(apply, init=init)
