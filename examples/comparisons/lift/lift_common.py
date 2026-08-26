"""Shared checks for the reusable lifted-stack comparison.

Twelve behaviours are the parity contract and are asserted, so a column
that fails one stops rather than printing a prettier number.

``nests`` is measured and printed instead. It is the one criterion the
columns disagree on, and asserting it would only make the failing column
crash; reporting it puts the difference in the output where it can be
compared. A transform whose own product is not an acceptable member is
not general over its input domain, whatever else it does.
"""

from __future__ import annotations

from dataclasses import dataclass


#: The member contexts every column offers its transform. A member always
#: owns parameters, since that is what a layer axis ranges over; what varies
#: is whether it carries state and whether it draws at apply. The last one is
#: the framework's OTHER mechanism for mutable state, which each of the three
#: has in a different form: a tag in nodejax, a Variable subclass in NNX, a
#: separate State object in Equinox.
CONTEXTS = ('plain', 'rng', 'state', 'state+rng', 'other-state')

WIDTH = 8
DEPTH = 4
DECAY = 0.75
DROP_RATE = 0.25
PARAM_SEED = 3
DROPOUT_SEED = 7
OTHER_DROPOUT_SEED = 11


@dataclass
class Evidence:
    """Observable behavior required from each reusable implementation."""

    parameter_shape: tuple[int, ...]
    state_shape: tuple[int, ...]
    aux_state_shape: tuple[int, ...]
    aux_energy_shape: tuple[int, ...]
    same_parameters: bool
    replayed: bool
    different_draw: bool
    missing_rng_rejected: bool
    surplus_rng_rejected: bool
    state_advanced: bool
    aux_matches_state: bool
    bare_output_supported: bool
    #: Whether the transform accepts its own product as a layer. Reported,
    #: not asserted; see the module docstring.
    nests: bool
    #: Which of CONTEXTS the transform carries. Reported, not asserted.
    contexts: tuple[str, ...]
    #: Whether differentiating a loss, the framework's own way, reaches the
    #: member's running statistic as well as its weights. False is the
    #: wanted answer: a running statistic is not a parameter, and an
    #: optimizer handed this gradient would update it as if it were.
    #: Reported, not asserted.
    state_gets_gradient: bool


def verify(label: str, evidence: Evidence) -> Evidence:
    """Assert the parity contract and print one compact summary."""
    assert evidence.parameter_shape == (DEPTH, WIDTH)
    assert evidence.state_shape == (DEPTH, WIDTH)
    assert evidence.aux_state_shape == (DEPTH, WIDTH)
    assert evidence.aux_energy_shape == (DEPTH,)
    assert evidence.same_parameters
    assert evidence.replayed
    assert evidence.different_draw
    assert evidence.missing_rng_rejected
    assert evidence.surplus_rng_rejected
    assert evidence.state_advanced
    assert evidence.aux_matches_state
    assert evidence.bare_output_supported
    missing = tuple(c for c in CONTEXTS if c not in evidence.contexts)
    print(
        f'{label:12s} params={evidence.parameter_shape} '
        f'state={evidence.state_shape} aux={evidence.aux_state_shape} '
        'rng=exact/replay/different bare-output=yes '
        f'nests={"yes" if evidence.nests else "NO"} '
        f'contexts={len(evidence.contexts)}/{len(CONTEXTS)}'
        + (f' rejected={",".join(missing)}' if missing else '')
        + (' grad=REACHES-STATE' if evidence.state_gets_gradient
           else ' grad=weights-only'),
        flush=True,
    )
    return evidence
