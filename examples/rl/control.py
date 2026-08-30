"""Shared policy-plant interaction Nodes for reinforcement-learning examples."""

import jax
import jax.numpy as jnp

from nodejax import (
    BaseNode,
    Composite,
    Node,
    PNode,
    PyTree,
    Struct,
    batch,
    node,
    split_aux,
    tile,
    tree_first,
)


@node
def ControlledStep(policy: Node, plant: PNode) -> Node:
    """Observe, choose a command, and advance one controlled plant step."""
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
def SamplingStep(policy: Node, plant: Node) -> Node:
    """One on-policy transition: sample, act, record what replay needs."""
    members = Composite(policy=policy, plant=plant)

    def apply(self, disturbance, initial_state, rng):
        observation = self.plant.observe()
        proposal = self.policy(observation)
        drawn = self.policy.sample(proposal, rng=rng.next())
        output = self.plant(command=drawn.command, disturbance=disturbance)
        return Struct(
            observation=observation,
            command=drawn.command,
            logprob=drawn.logprob,
            cost=output.cost,
            next_observation=self.plant.observe(),
        )

    def init(param, input):
        """Start a rollout from caller-supplied plant state; a stateless
        policy contributes the empty slot."""
        observation = plant.observe(state=input.initial_state)
        return Struct(
            policy=policy.bind(param.policy).init(input=observation),
            plant=input.initial_state,
        )

    return members(apply, init=init)


def collect(
    sampler: Node,
    policy_param: PyTree,
    initial_state: Struct,
    disturbance: jax.Array,
    key: jax.Array,
    *,
    n_chunks: int,
    n_steps_per_chunk: int,
) -> tuple[Struct, PyTree]:
    """Collect a native ``(chunk, time, world)`` rollout and chunk-end state."""
    starts = tile(tile(initial_state, n_steps_per_chunk), n_chunks)
    output = sampler.bind(Struct(policy=policy_param)).apply(
        disturbance=disturbance,
        initial_state=starts,
        rng=key,
    )
    record, aux = split_aux(output)
    # The sampler is scanned with record=True, so aux carries the sampler
    # state after every chunk; replay starts derive from that trace.
    return record, aux.state


def chunk_starts(
    policy: Node,
    policy_param: PyTree,
    observation: Struct,
    chunk_end_state: PyTree,
    n_worlds: int,
) -> PyTree:
    """Recover each chunk's policy state for deterministic policy init."""
    if not policy.cyclic:
        return ()
    first_observation = tree_first(tree_first(observation))
    batched_policy = batch(policy, n=n_worlds).bind(policy_param)
    first = batched_policy.init(input=first_observation)
    # Recorded state includes empty acyclic slots. Binding removes them so the
    # chunk endings have the same public tree as freshly initialized state.
    ends = batched_policy.bind(state=chunk_end_state.policy).state
    return jax.tree.map(
        lambda initial, ending: jnp.concatenate((initial[None], ending[:-1])),
        first,
        ends,
    )
