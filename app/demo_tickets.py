"""Curated synthetic demo tickets for the MVP UI."""

from __future__ import annotations

DEMO_TICKETS = [
    {
        "title": "Security incident on the account portal",
        "subject": "Critical security incident affecting the account portal",
        "body": (
            "Multiple users report that the account portal is unavailable after suspicious login "
            "activity. We need urgent investigation, containment guidance, and an update on "
            "service restoration steps."
        ),
    },
    {
        "title": "Invoice mismatch after subscription renewal",
        "subject": "Unexpected invoice amount after renewal",
        "body": (
            "Our latest invoice shows duplicate charges after renewing the enterprise plan. "
            "Please review the billing details, explain the discrepancy, and advise on the refund process."
        ),
    },
    {
        "title": "Product issue after firmware update",
        "subject": "Firmware update causes device instability",
        "body": (
            "Since the latest firmware update, the device starts but quickly loses connectivity "
            "and becomes unreliable. Please investigate the root cause and provide a short-term fix recommendation."
        ),
    },
    {
        "title": "Return request for defective accessories",
        "subject": "Request to return defective accessories",
        "body": (
            "We received several accessories with visible defects and need instructions for the return "
            "and replacement process, including shipping labels and processing time."
        ),
    },
    {
        "title": "Employee portal access request",
        "subject": "Need access to the employee onboarding portal",
        "body": (
            "A new team member cannot access the onboarding portal and related HR documents. Please "
            "help us restore access and confirm the required permissions."
        ),
    },
]


def get_demo_ticket(index: int = 0) -> tuple[int, dict[str, str]]:
    """Return one deterministic demo ticket by index."""
    if not DEMO_TICKETS:
        raise ValueError("No demo tickets are configured.")
    resolved_index = index % len(DEMO_TICKETS)
    return resolved_index, DEMO_TICKETS[resolved_index].copy()
