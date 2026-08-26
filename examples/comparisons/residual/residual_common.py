"""Shared checks for the transparent-wrapper comparison.

A residual wrapper is the smallest transform there is. It owns no axis, no
state, no entropy and no parameters. It adds its input to its member's
output and forwards everything else untouched.

That is what makes it the controlled experiment. The lifted-stack comparison
next door confounds two things: axis machinery, which is genuinely hard
anywhere, and transparency, which is the point. Here there is no axis
machinery to blame, so whatever a column cannot do is a transparency failure
and nothing else.

Seven behaviours are the parity contract and are asserted. Whether a
parameter gradient reaches running state is reported separately because it
depends on the state representation chosen by each column.
"""

from __future__ import annotations

from dataclasses import dataclass


WIDTH = 8
DECAY = 0.75
DROP_RATE = 0.25
PARAM_SEED = 3
DROPOUT_SEED = 7
OTHER_DROPOUT_SEED = 11


@dataclass
class Evidence:
    """Observable behavior required from each transparent wrapper."""

    #: x + f(x) preserves the member's shape.
    output_shape: tuple[int, ...]
    #: The wrapper adds nothing to the member's parameters.
    parameter_shape: tuple[int, ...]
    #: The member's state still advances through the wrapper.
    state_advanced: bool
    #: Entropy still reaches the member: same key replays, another differs.
    replayed: bool
    different_draw: bool
    #: The member's aux still escapes, rather than being added to.
    aux_escaped: bool
    #: The wrapper accepts its own product around both a deterministic and a
    #: stochastic member.
    nests: bool
    #: Whether differentiating a loss, the framework's own way, reaches the
    #: member's running statistic as well as its weights. False is the
    #: wanted answer: a running statistic is not a parameter, and an
    #: optimizer handed this gradient would update it as if it were. The
    #: wrapper owns nothing here, so this is a fact about the framework's
    #: state model that the wrapper cannot repair. Reported, not asserted.
    state_gets_gradient: bool


def verify(label: str, evidence: Evidence) -> Evidence:
    """Assert the parity contract and print one compact summary."""
    assert evidence.output_shape == (WIDTH,)
    assert evidence.parameter_shape == (WIDTH,)
    assert evidence.state_advanced
    assert evidence.replayed
    assert evidence.different_draw
    assert evidence.aux_escaped
    assert evidence.nests
    print(
        f'{label:12s} out={evidence.output_shape} '
        f'params={evidence.parameter_shape} state=advanced '
        f'rng=replay/different aux=escaped '
        f'nests={"yes" if evidence.nests else "NO"} '
        + ('grad=REACHES-STATE' if evidence.state_gets_gradient
           else 'grad=weights-only'),
        flush=True,
    )
    return evidence
