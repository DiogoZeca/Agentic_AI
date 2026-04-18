# Baseline Comparison — Google Cluster 2011

All baselines evaluated on the **test split** (last 20% chronologically).
Scores computed from raw cluster_agg schema only (no engineered features).
Labels and splits identical to XGBoost evaluation.

## PR-AUC Comparison

| Baseline             | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m macro PR-AUC | OVR PR-AUC |
| ---                  | --- | --- | --- | --- | --- |
| random               | 0.052 | 0.078 | 0.099 | 0.333 | 0.018 |
| persistence          | 0.481 | 0.455 | 0.447 | 0.433 | 0.229 |
| static_thresh        | 0.320 | 0.301 | 0.295 | 0.414 | 0.163 |
| ewma_zscore          | 0.089 | 0.108 | 0.126 | 0.335 | 0.024 |
| rolling_zscore       | 0.096 | 0.114 | 0.132 | 0.335 | 0.025 |
| arima                | 0.466 | 0.442 | 0.435 | 0.430 | 0.221 |
| **XGBoost (ours)**   | **0.575** | **0.563** | **0.562** | **0.547** | **0.339** |

## Per-Horizon Detail

### 15m PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.052 | 0.501 | 4758488 | 0.0515 |
| persistence | 0.481 | 0.861 | 4758488 | 0.0515 |
| static_thresh | 0.320 | 0.720 | 4758488 | 0.0515 |
| ewma_zscore | 0.089 | 0.623 | 4758488 | 0.0515 |
| rolling_zscore | 0.096 | 0.633 | 4758488 | 0.0515 |
| arima | 0.466 | 0.845 | 4758488 | 0.0515 |
| **XGBoost** | **0.575** | — | — | — |

### 30m PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.078 | 0.501 | 4720928 | 0.0775 |
| persistence | 0.455 | 0.813 | 4720928 | 0.0775 |
| static_thresh | 0.301 | 0.675 | 4720928 | 0.0775 |
| ewma_zscore | 0.108 | 0.584 | 4720928 | 0.0775 |
| rolling_zscore | 0.114 | 0.594 | 4720928 | 0.0775 |
| arima | 0.442 | 0.799 | 4720928 | 0.0775 |
| **XGBoost** | **0.563** | — | — | — |

### 45m PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.099 | 0.501 | 4683365 | 0.0991 |
| persistence | 0.447 | 0.785 | 4683365 | 0.0991 |
| static_thresh | 0.295 | 0.651 | 4683365 | 0.0991 |
| ewma_zscore | 0.126 | 0.565 | 4683365 | 0.0991 |
| rolling_zscore | 0.132 | 0.574 | 4683365 | 0.0991 |
| arima | 0.435 | 0.772 | 4683365 | 0.0991 |
| **XGBoost** | **0.562** | — | — | — |

### 60m macro PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.333 | 0.500 | 4645803 | nan |
| persistence | 0.433 | 0.591 | 4645803 | nan |
| static_thresh | 0.414 | 0.565 | 4645803 | nan |
| ewma_zscore | 0.335 | 0.518 | 4645803 | nan |
| rolling_zscore | 0.335 | 0.521 | 4645803 | nan |
| arima | 0.430 | 0.588 | 4645803 | nan |
| **XGBoost** | **0.547** | — | — | — |

### OVR PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.018 | 0.501 | 4645803 | 0.0177 |
| persistence | 0.229 | 0.779 | 4645803 | 0.0177 |
| static_thresh | 0.163 | 0.715 | 4645803 | 0.0177 |
| ewma_zscore | 0.024 | 0.555 | 4645803 | 0.0177 |
| rolling_zscore | 0.025 | 0.565 | 4645803 | 0.0177 |
| arima | 0.221 | 0.768 | 4645803 | 0.0177 |
| **XGBoost** | **0.339** | — | — | — |
