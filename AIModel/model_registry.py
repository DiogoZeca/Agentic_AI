"""Model routing table — change priority here without touching api.py."""
from __future__ import annotations

MODEL_ROUTING: dict[str, list[str]] = {
    "carbonEmissions": ["formula", "timesfm", "prophet"],
    "consumption":     ["prophet"],
}
_DEFAULT_ROUTING: list[str] = ["timesfm", "prophet"]


def get_best_model_key(metric: str, available_keys: set[str]) -> str:
    """Return the first key from the routing list that is in available_keys."""
    routing = MODEL_ROUTING.get(metric, _DEFAULT_ROUTING)
    for key in routing:
        if key in available_keys:
            return key
    return "prophet"
