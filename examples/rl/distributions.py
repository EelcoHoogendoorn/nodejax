"""Reusable probability-distribution Nodes for reinforcement-learning examples."""

import jax
import jax.numpy as jnp

from nodejax import Composite, Leaf, Node, Struct, node


@node
def StateIndependentLogStd(initial: float) -> Node:
    """A learned state-independent log standard deviation."""
    def param():
        return jnp.asarray(initial)

    def apply(param, input):
        return param

    return Leaf(apply, param=param)


@node
def LearnedGaussian(mean: Node, log_std: Node) -> Node:
    """A Gaussian proposal with independently replaceable parameter Nodes."""
    members = Composite(mean=mean, log_std=log_std)

    def apply(self, input):
        return Struct(
            mean=self.mean(input),
            log_std=self.log_std(input),
        )

    def sample(rng, proposal: Struct) -> jax.Array:
        return proposal.mean + jnp.exp(proposal.log_std) * jax.random.normal(
            rng.next(),
            proposal.mean.shape,
        )

    def logprob(proposal: Struct, command: jax.Array) -> jax.Array:
        spread = jnp.exp(proposal.log_std)
        element = (
            -0.5 * ((command - proposal.mean) / spread) ** 2
            - proposal.log_std
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        return jnp.sum(element)

    def entropy(proposal: Struct) -> jax.Array:
        element = jnp.broadcast_to(
            proposal.log_std,
            proposal.mean.shape,
        )
        return jnp.sum(
            element + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
        )

    return members(
        apply,
        methods={
            'sample': sample,
            'logprob': logprob,
            'entropy': entropy,
        },
    )
