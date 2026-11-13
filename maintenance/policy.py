"""PREDICAI — maintenance recommendation policy (Part 8 / Phase B).

IMPORTANT: these thresholds are POLICY CHOICES for planning reviews —
they are NOT scientifically validated failure boundaries, and the UI/docs
must say so. A recommendation never claims a machine will fail at a
specific cycle; RUL itself is an estimate in cycles.

Inputs: predicted RUL (LSTM), cycles since last service, last service type.
Output: level + a human-readable reason that cites the actual numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

POLICY_VERSION = "1.0"

# Policy bands (cycles) — configurable, deliberately conservative.
RUL_REVIEW_MAX = 30        # RUL at or below this -> maintenance review
RUL_INSPECTION_MAX = 50    # RUL at or below this -> schedule inspection
SERVICE_REVIEW_CYCLES = 300  # long time since service escalates one level


@dataclass
class Recommendation:
    level: str   # MONITOR | INSPECTION | REVIEW
    reason: str


def recommend(
    predicted_rul: float | None,
    cycles_since_service: int | None,
    last_service_type: str | None = None,
) -> Recommendation:
    """Combine RUL + service history into a reasoned recommendation."""
    parts: list[str] = []
    level = "MONITOR"

    if predicted_rul is not None:
        parts.append(f"predicted RUL {predicted_rul:.0f} cycles")
        if predicted_rul <= RUL_REVIEW_MAX:
            level = "REVIEW"
        elif predicted_rul <= RUL_INSPECTION_MAX:
            level = "INSPECTION"
    else:
        parts.append("no RUL prediction available yet")

    if cycles_since_service is not None:
        parts.append(f"{cycles_since_service} cycles since last service")
        if cycles_since_service >= SERVICE_REVIEW_CYCLES:
            if level == "MONITOR":
                level = "INSPECTION"
                parts.append("service interval is long, escalating to inspection")
            elif level == "INSPECTION":
                level = "REVIEW"
                parts.append("long service interval combined with low RUL, escalating to review")
    else:
        parts.append("no service history recorded")

    if last_service_type:
        parts.append(f"last service type: {last_service_type}")

    action = {
        "MONITOR": "Continue monitoring",
        "INSPECTION": "Schedule an inspection",
        "REVIEW": "Maintenance review recommended",
    }[level]
    return Recommendation(level=level, reason=f"{action} — " + "; ".join(parts) + ".")
