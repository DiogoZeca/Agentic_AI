# Baseline Comparison — Zabbix

All baselines evaluated on the **test split** (last 20% chronologically).
Scores computed from raw cluster_agg schema only (no engineered features).
Labels and splits identical to XGBoost evaluation.

## PR-AUC Comparison

| Baseline             | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m macro PR-AUC | OVR PR-AUC |
| ---                  | --- | --- | --- | --- | --- |
| random               | 0.380 | 0.434 | 0.467 | 0.333 | 0.226 |
| persistence          | 0.950 | 0.945 | 0.944 | 0.454 | 0.631 |
| static_thresh        | 0.911 | 0.887 | 0.877 | 0.480 | 0.627 |
| ewma_zscore          | 0.394 | 0.453 | 0.485 | 0.335 | 0.237 |
| rolling_zscore       | 0.394 | 0.450 | 0.481 | 0.335 | 0.237 |
| arima                | 0.949 | 0.946 | 0.943 | 0.474 | 0.633 |
| **XGBoost (ours)**   | **0.957** | **0.957** | **0.957** | **0.848** | **0.926** |

## Per-Horizon Detail

### 15m PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.380 | 0.500 | 56361 | 0.3803 |
| persistence | 0.950 | 0.960 | 56361 | 0.3803 |
| static_thresh | 0.911 | 0.930 | 56361 | 0.3803 |
| ewma_zscore | 0.394 | 0.536 | 56361 | 0.3803 |
| rolling_zscore | 0.394 | 0.534 | 56361 | 0.3803 |
| arima | 0.949 | 0.957 | 56361 | 0.3803 |
| **XGBoost** | **0.957** | — | — | — |

### 30m PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.434 | 0.500 | 56328 | 0.4338 |
| persistence | 0.945 | 0.949 | 56328 | 0.4338 |
| static_thresh | 0.887 | 0.900 | 56328 | 0.4338 |
| ewma_zscore | 0.453 | 0.537 | 56328 | 0.4338 |
| rolling_zscore | 0.450 | 0.533 | 56328 | 0.4338 |
| arima | 0.946 | 0.947 | 56328 | 0.4338 |
| **XGBoost** | **0.957** | — | — | — |

### 45m PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.467 | 0.501 | 56295 | 0.4663 |
| persistence | 0.944 | 0.942 | 56295 | 0.4663 |
| static_thresh | 0.877 | 0.884 | 56295 | 0.4663 |
| ewma_zscore | 0.485 | 0.534 | 56295 | 0.4663 |
| rolling_zscore | 0.481 | 0.529 | 56295 | 0.4663 |
| arima | 0.943 | 0.939 | 56295 | 0.4663 |
| **XGBoost** | **0.957** | — | — | — |

### 60m macro PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.333 | 0.500 | 56262 | nan |
| persistence | 0.454 | 0.554 | 56262 | nan |
| static_thresh | 0.480 | 0.548 | 56262 | nan |
| ewma_zscore | 0.335 | 0.504 | 56262 | nan |
| rolling_zscore | 0.335 | 0.504 | 56262 | nan |
| arima | 0.474 | 0.554 | 56262 | nan |
| **XGBoost** | **0.848** | — | — | — |

### OVR PR-AUC

| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |
| --- | --- | --- | --- | --- |
| random | 0.226 | 0.502 | 56262 | 0.2248 |
| persistence | 0.631 | 0.873 | 56262 | 0.2248 |
| static_thresh | 0.627 | 0.886 | 56262 | 0.2248 |
| ewma_zscore | 0.237 | 0.541 | 56262 | 0.2248 |
| rolling_zscore | 0.237 | 0.540 | 56262 | 0.2248 |
| arima | 0.633 | 0.871 | 56262 | 0.2248 |
| **XGBoost** | **0.926** | — | — | — |
