"""Re-exports `drop_aux` for stripping `Aux` payloads from concrete runtime outputs."""

from __future__ import annotations

from nodejax.core.binding import drop_aux, split_aux

__all__ = ['drop_aux']
