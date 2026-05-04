"""PSI (Population Stability Index) utilities.

Why PSI?
--------
PSI is the standard metric for detecting distribution shift in production ML
systems because it produces a single scalar per feature that is directly
actionable: < 0.10 means stable, 0.10–0.25 means monitor, > 0.25 means retrain.
It is interpretable by operators without statistical background, unlike KL
divergence (unbounded, asymmetric) or Wasserstein distance (meaningful but harder
to threshold).  PSI is also symmetric in PSI(P,Q) ≈ PSI(Q,P) for moderate shifts,
which reduces false alarms from transient direction changes in the live window.

Threshold values (industry standard, published in credit-risk literature):
  < PSI_STABLE   (0.10) → no significant population shift; model stable
  PSI_STABLE–PSI_WARNING (0.10–0.25) → monitor; retrain may be warranted
  PSI_WARNING–PSI_EMERGENCY (0.25–0.50) → significant shift; schedule retraining
  > PSI_EMERGENCY (0.50) → severe shift; emergency rollback (model may be invalid)

These thresholds were originally calibrated for credit-scoring features that update
quarterly.  For 5-min telemetry features with faster drift, the thresholds are
conservative: a PSI of 0.30 in CPU data could reflect a planned capacity event
rather than permanent drift.  The system-wide alert in drift_monitor.py (> 10% of
features ≥ 0.20) catches broad shifts earlier without escalating single-feature noise.

Shared by spike/drift_monitor.py and evaluation/evaluate_cross_domain.py so that
the threshold constants and core computation have exactly one authoritative source.
"""
from __future__ import annotations

import numpy as np

# Threshold constants — see module docstring for derivation and interpretation.
PSI_STABLE    = 0.10   # < 0.10: no significant distribution shift
PSI_WARNING   = 0.25   # 0.10–0.25: worth monitoring; > 0.25 consider retraining
PSI_EMERGENCY = 0.50   # > 0.50: emergency rollback
N_PSI_BINS    = 10     # default quantile bin count (matches evaluate_cross_domain.py)


def compute_psi_single(
    expected: np.ndarray,
    actual:   np.ndarray,
    n_bins:   int = N_PSI_BINS,
) -> float:
    """Compute Population Stability Index for one feature.

    PSI = Σ (actual_pct_i - expected_pct_i) * ln(actual_pct_i / expected_pct_i)
    summed over all bins.  Positive contributions come from bins where the live
    distribution over-represents relative to training; negative from under-representation.
    The sum is always ≥ 0 by Jensen's inequality (it equals KL(actual || expected) +
    KL(expected || actual) for the discrete bin distributions).

    Why quantile edges from `expected` (not `actual`)?
    ---------------------------------------------------
    Deriving bin edges from the expected (training) distribution guarantees that
    every expected bin has at least one observation, which prevents division by zero
    and makes exp_pct a proper probability distribution.  If we used fixed-width bins
    or actual-derived edges, any empty training bin would produce NaN immediately.

    Why np.unique(edges)?
    ---------------------
    For features with low effective cardinality (e.g., n_tasks = 1 for all machines
    in a quiet window), quantiles collapse to a single repeated value.  np.unique()
    deduplicates so that numpy.histogram receives strictly increasing edges.  When
    fewer than 2 unique edges survive, the feature has zero variance in the reference
    and we return NaN — PSI is undefined for a degenerate reference distribution.

    Why a special case for binary features?
    ----------------------------------------
    Binary features (spike_now, spike_in_last_1, etc.) have only two values: {0, 1}.
    Quantile bins from these values always collapse to [-inf, 0.5, inf] or a
    single edge.  Instead we hard-code 2 exact bins [-0.5, 0.5] and [0.5, 1.5].
    This preserves the only meaningful distinction (was the feature 0 or 1?) and
    avoids the quantile-collapse NaN path.

    Why eps = 1e-4 (the "pseudocount")?
    -------------------------------------
    During quiet telemetry windows (e.g., midnight), binary spike features may be
    entirely 0 → the bin [0.5, 1.5] gets 0 actual observations → log(0/...) = -inf.
    Clipping counts to eps=1e-4 is the industry-standard pseudocount fix: it adds a
    tiny fictitious observation to each bin so the formula stays finite.  The resulting
    PSI (~2–3 for a fully shifted binary) is inflated compared to the true value, but
    the operator is warned via min_bins_warning in the report.

    Parameters
    ----------
    expected : 1-D array — training distribution values (NaN/inf are stripped).
    actual   : 1-D array — live distribution values (NaN/inf are stripped).
    n_bins   : number of quantile bins for continuous features (ignored for binary).

    Returns
    -------
    float — PSI value ≥ 0, or nan when the feature is degenerate.
    """
    # Strip non-finite values first; NaN from lag features in short live windows
    # is normal and must not propagate into the histogram.
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # Binary features: {0} or {1} or {0,1} → 2 exact bins.
    # Detected on the expected array; if expected is all-0 or all-1 we still use
    # this path (the eps clip handles the empty bin in actual).
    unique_vals = np.unique(expected)
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        edges = np.array([-0.5, 0.5, 1.5])
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges     = np.unique(np.percentile(expected, quantiles))
        if len(edges) < 2:
            # Zero variance in reference — PSI is undefined.
            return float("nan")

    eps = 1e-4   # pseudocount: prevents log(0) for empty bins
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual,   bins=edges)

    # Normalise to proportions then clip to eps so no bin is exactly zero.
    exp_pct = (exp_counts / len(expected)).clip(eps)
    act_pct = (act_counts / len(actual)).clip(eps)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
