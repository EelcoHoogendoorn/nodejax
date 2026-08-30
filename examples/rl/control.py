"""Shared policy-plant interaction Nodes for reinforcement-learning examples."""

from nodejax import BaseNode, Composite, Node, PNode, Struct, node


@node
def ControlledStep(policy: BaseNode, plant: PNode) -> Node:
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

    def init(self, input):
        """Adopt caller state and initialize policy state from observation.

        ``initial_state`` shares the apply call because a scan primes from its
        first real element. It is ignored by later transitions. A stateless
        policy contributes the empty slot: no fork on its lifecycle.
        """
        observation = plant.observe(state=input.initial_state)
        weights = self.policy if policy.parametric else ()
        return Struct(
            policy=policy.bind(weights).init(input=observation),
            plant=input.initial_state,
        )

    return members(apply, init=init)
