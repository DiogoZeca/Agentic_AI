# Zabbix Evaluation Report
Generated: 2026-04-27T20:12:11.754585+00:00
Source model (Google Cluster 2011 K=2 baseline): 60m PR-AUC = 0.547

---

## Phase 0 — EDA

| node | n_buckets | idle_fraction | burstiness_B | pmr | spike_rate | biz_ratio | acf_lag1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atari | 24842 | 0.5308 | 0.0421 | 2.51 | 0.05 | 0.99 | 0.9867 |
| atnog-bkpConfigs | 24843 | 0.9458 | 0.5115 | 11.29 | 0.05 | 1.05 | 0.5211 |
| atnog-docker | 24838 | 0.0 | -0.6664 | 1.79 | 0.05 | 1.01 | 0.76 |
| atnog-nas | 24834 | 0.0 | 0.0186 | 3.26 | 0.0228 | 1.16 | 0.7753 |
| atnog-webphp | 24828 | 0.0 | -0.3605 | 2.07 | 0.05 | 0.98 | 0.5938 |
| jarvis-controller | 24828 | 0.0 | -0.7288 | 1.4 | 0.05 | 1.01 | 0.5607 |
| jarvis-node1 | 24856 | 0.0 | -0.4929 | 2.01 | 0.05 | 1.0 | 0.8561 |
| jarvis-node2 | 24838 | 0.0 | -0.3283 | 2.38 | 0.05 | 0.99 | 0.9649 |
| sega | 24847 | 0.5866 | 0.7886 | 13.55 | 0.05 | 0.97 | 0.9626 |
| skynet | 24789 | 0.7491 | 0.3634 | 5.22 | 0.05 | 1.1 | 0.7455 |
| xbox | 24855 | 0.7411 | 0.2716 | 4.32 | 0.05 | 1.11 | 0.9975 |

### ⚠️  Domain-incompatibility warnings
- 5/11 nodes have idle_fraction > 40% (Google 2011: ~15–20%). Per-machine p95 features may cluster at 0 during idle.
- All nodes have similar spike_rate (std=0.008). cluster_cpu_p90 and machine_rank_in_cluster may carry less signal than across Google's 12,555 diverse machines.

---

## Phase 1 — Domain Shift (PSI)

Feature shift summary: 25 MAJOR / 9 moderate / 6 minor

### MAJOR shift features (PSI > 0.25) — model degraded on these:

| Feature | PSI |
| --- | --- |
| task_dominance | 8.2831 |
| band_width | 4.364 |
| cluster_cpu_p90 | 3.1534 |
| n_tasks | 2.6904 |
| total_mem | 2.568 |
| disk_io | 1.6656 |
| peak_mem | 1.3562 |
| cpu_ewma_24 | 1.1284 |
| cpu_ewma_6 | 1.0331 |
| total_cpu | 1.021 |
| cpu_lag_1 | 1.0188 |
| cpu_lag_12 | 0.9875 |
| peak_cpu | 0.9318 |
| cpu_lag_24 | 0.901 |
| peak_cpu_vs_p99 | 0.8828 |
| cpu_rolling_std_6 | 0.8553 |
| cpu_per_task | 0.8503 |
| cpu_vs_p99 | 0.765 |
| cpu_vs_p95 | 0.6503 |
| cpu_vs_p95_slope_6 | 0.5478 |
| peak_cpu_vs_p95 | 0.4978 |
| cpu_delta_2 | 0.4883 |
| cpu_delta_1 | 0.4598 |
| cpu_vs_p95_delta | 0.3072 |
| cpu_vs_p95_slope_3 | 0.3053 |

### Moderate shift features (0.10 < PSI ≤ 0.25):

| Feature | PSI |
| --- | --- |
| time_to_p95_3 | 0.1568 |
| spike_severe_in_last_1 | 0.1385 |
| spike_severe_now | 0.1376 |
| time_since_last_spike | 0.1334 |
| spike_in_last_3 | 0.1233 |
| spike_in_last_1 | 0.12 |
| spike_now | 0.1197 |
| spike_in_last_6 | 0.1197 |
| spike_severe_in_last_3 | 0.1158 |

---

## Phase 2 — Model Evaluation

Test split: last 20% of Zabbix data (~18 days). Labels computed with K=1 (any exceedance).

| Model | Test PR-AUC | Source PR-AUC | Retention | Test ROC-AUC | N test |
| --- | --- | --- | --- | --- | --- |
| 60m severity  | 0.708 | 0.547     | 129%  | 0.871 | 56262 |
| 15m binary    | 0.956 | 0.575 | 166%  | 0.966 | 56361 |
| 30m binary    | 0.957 | 0.563            | 170%  | 0.960 | 56328 |
| 45m binary    | 0.956 | 0.562            | 170%  | 0.954 | 56295 |
| OVR severe    | 0.891 | 0.339 | 263% | 0.928 | 56394 |

### 60m per-class PR-AUC

| Class | Label | Test PR-AUC | Class rate |
| --- | --- | --- | --- |
| 0 | no_spike | 0.937 | 0.508 |
| 1 | moderate | 0.453 | 0.267 |
| 2 | severe | 0.734 | 0.225 |

---

## Decision

60m macro PR-AUC retention vs source: **129%**

→ **Recalibrate on Zabbix val split and deploy.** Model transfers well.

---

## Phase 3 — Zabbix Recalibration

### 60m Severity (Zabbix-recalibrated)

| Metric | Value |
| --- | --- |
| Macro PR-AUC | 0.800 |
| Macro ROC-AUC | 0.900 |
| Alarm threshold | 0.45 |
| Alarm P / R / F1 | 0.703 / 0.631 / 0.665 |

| Class | Label | PR-AUC | 95% CI | ROC-AUC |
| --- | --- | --- | --- | --- |
| 0 | no_spike | 0.926 | [0.923, 0.929] | 0.938 |
| 1 | moderate | 0.724 | [0.717, 0.730] | 0.852 |
| 2 | severe | 0.751 | [0.742, 0.759] | 0.911 |

### 15m Binary (Zabbix threshold)

| Metric | Value |
| --- | --- |
| PR-AUC | 0.956 |
| PR-AUC 95% CI | [0.954, 0.958] |
| ROC-AUC | 0.966 |
| Alarm threshold | 0.65 |
| Alarm P / R / F1 | 0.800 / 0.868 / 0.833 |

### 30m Binary (Zabbix threshold)

| Metric | Value |
| --- | --- |
| PR-AUC | 0.957 |
| PR-AUC 95% CI | [0.955, 0.958] |
| ROC-AUC | 0.960 |
| Alarm threshold | 0.45 |
| Alarm P / R / F1 | 0.814 / 0.912 / 0.860 |

### 45m Binary (Zabbix threshold)

| Metric | Value |
| --- | --- |
| PR-AUC | 0.956 |
| PR-AUC 95% CI | [0.954, 0.957] |
| ROC-AUC | 0.954 |
| Alarm threshold | 0.35 |
| Alarm P / R / F1 | 0.837 / 0.921 / 0.877 |

### OVR Severe (Zabbix threshold)

| Metric | Value |
| --- | --- |
| PR-AUC | 0.891 |
| PR-AUC 95% CI | [0.886, 0.896] |
| ROC-AUC | 0.928 |
| Alarm threshold | 0.55 |
| Alarm P / R / F1 | 0.825 / 0.596 / 0.692 |
