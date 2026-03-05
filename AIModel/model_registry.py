"""
Model Registry — routing table for best-model selection per metric.

Replaces the _TIMESFM_PREFERRED magic constant in api.py with a self-documenting
dict that can be extended without touching API logic.

Routing priority (left = highest):
  carbonEmissions → formula first (physically correct), then timesfm, then prophet
  consumption     → prophet (chained regressor excels here)
  all others      → timesfm, then prophet
"""
from __future__ import annotations

MODEL_ROUTING: dict[str, list[str]] = {
    "carbonEmissions": ["formula", "timesfm", "prophet"],
    "consumption":     ["prophet"],
}
_DEFAULT_ROUTING: list[str] = ["timesfm", "prophet"]


def get_best_model_key(metric: str, available_keys: set[str]) -> str:
    """Return the highest-priority model key available for a metric.

    Searches the routing list for `metric` (falling back to _DEFAULT_ROUTING)
    and returns the first key present in `available_keys`.
    Always falls back to 'prophet' unconditionally if nothing else matches.

    Args:
        metric:         Metric name (e.g. 'carbonEmissions', 'consumption')
        available_keys: Set of model keys that are actually loaded and ready

    Returns:
        A model key string: 'formula', 'timesfm', or 'prophet'
    """
    routing = MODEL_ROUTING.get(metric, _DEFAULT_ROUTING)
    for key in routing:
        if key in available_keys:
            return key
    return "prophet"
