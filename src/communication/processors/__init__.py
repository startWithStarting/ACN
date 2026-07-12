"""Communication processors: per-agent transformation of delivered payloads.

Phase 1 ships the inbox-preserving :class:`DirectProcessor`
(:mod:`src.communication.processors.identity`); Phase 3 adds the first-seen
unchanged :class:`RelayProcessor` (:mod:`src.communication.processors.relay`)
over the same transport. Aggregation and learned processors arrive in later
delivery phases.
"""

from .identity import DirectProcessor
from .relay import RelayCarryoverTransport, RelayProcessor, RelayState

__all__ = ["DirectProcessor", "RelayCarryoverTransport", "RelayProcessor", "RelayState"]
