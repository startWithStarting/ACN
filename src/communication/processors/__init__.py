"""Communication processors: per-agent transformation of delivered payloads.

Phase 1 ships only the inbox-preserving :class:`DirectProcessor`
(:mod:`src.communication.processors.identity`). Relay, aggregation, and
learned processors arrive in later delivery phases over the same transport.
"""

from .identity import DirectProcessor

__all__ = ["DirectProcessor"]
