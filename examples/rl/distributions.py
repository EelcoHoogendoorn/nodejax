"""Provides reusable probability-distribution Nodes for reinforcement learning."""

import jax
import jax.numpy as jnp

from nodejax import BaseNode, Composite, Leaf, Node, Struct, node


@node
def StateIndependentLogStd(initial: float) -> Node:
    """Returns a learned state-independent log standard deviation initialized to `initial`."""
    def param():
        return jnp.asarray(initial)

    def apply(param, input):
        return param

    return Leaf(apply, param=param)


@node
def LearnedGaussian(mean: BaseNode, log_std: BaseNode) -> Node:
    """Combines replaceable `mean` and `log_std` Nodes into a diagonal Gaussian proposal."""
    members = Composite(mean=mean, log_std=log_std)

    def apply(self, input):
        return Struct(
            mean=self.mean(input),
            log_std=self.log_std(input),
        )

    def sample(rng, proposal: Struct) -> Struct:
        command = proposal.mean + jnp.exp(proposal.log_std) * (
            jax.random.normal(rng.next(), proposal.mean.shape)
        )
        return Struct(
            command=command,
            logprob=logprob(proposal, command),
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


@node
def SquashedGaussian(mean: BaseNode, log_std: BaseNode, scale: float) -> Node:
    """Produces a tanh-squashed Gaussian proposal with commands bounded in `(-scale, scale)`.

    Computes both the squashed command and its change-of-variables log-probability directly
    from the pre-squash draw in `sample`, avoiding inverse hyperbolic tangents.
    """
    members = Composite(mean=mean, log_std=log_std)

    def apply(self, input):
        return Struct(
            mean=self.mean(input),
            log_std=self.log_std(input),
        )

    def sample(rng, proposal: Struct) -> Struct:
        spread = jnp.exp(proposal.log_std)
        draw = proposal.mean + spread * jax.random.normal(
            rng.next(),
            proposal.mean.shape,
        )
        gaussian = (
            -0.5 * ((draw - proposal.mean) / spread) ** 2
            - proposal.log_std
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        # log|d command / d draw|, with log(1 - tanh^2) in its stable form
        squash = jnp.log(scale) + 2.0 * (
            jnp.log(2.0) - draw - jax.nn.softplus(-2.0 * draw)
        )
        return Struct(
            command=scale * jnp.tanh(draw),
            logprob=jnp.sum(gaussian - squash),
        )

    return members(apply, methods={'sample': sample})
