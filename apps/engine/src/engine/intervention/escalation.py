"""
Intervention — escalation.py (Day 4 implementation).

Human-in-the-loop escalation stub.
When the circuit breaker fires and forced summarization is not sufficient,
this module can escalate the run to a human reviewer.

Stub for Day 1 — full implementation in Day 4.
"""

import logging

logger = logging.getLogger(__name__)


def escalate_to_human(run_id: str, query: str, reason: str) -> None:
    """
    Stub: log the escalation request.
    Day 4+ will send this to a webhook, email, or Slack notification.
    """
    logger.warning(
        f"[ESCALATION] Run {run_id} requires human review | "
        f"Reason: {reason} | Query: {query[:80]}..."
    )
    # TODO (Day 4): Implement actual escalation (webhook, email, Slack, etc.)
