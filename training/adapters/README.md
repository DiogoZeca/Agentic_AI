# Data Adapters

An adapter converts your monitoring system's raw metrics into the `cluster_agg`
schema that the spike prediction pipeline expects.  Once your data is in this
format, the full pipeline — feature engineering, training, evaluation, and
inference — works unchanged.

## The `cluster_agg` schema

One row per `(machine_id, 5-minute bucket)`.

| Column | Type | Description |
|---|---|---|
| `machine_id` | int64 | Unique integer identifier for the machine |
| `bucket` | int64 | `unix_timestamp_sec // 300` — aligned to UTC midnight |
| `time_us` | int64 | `bucket * 300_000_000` (microseconds) |
| `total_cpu` | float32 | Duration-weighted CPU load, **fraction of 1 core** (not %) |
| `peak_cpu` | float32 | Peak CPU rate observed within the 5-min window |
| `total_mem` | float32 | Sum of canonical memory usage |
| `peak_mem` | float32 | Peak memory usage |
| `disk_io` | float32 | Max mean disk I/O time |
| `n_tasks` | int32 | Number of concurrent tasks in the bucket |

### Unit notes

- **CPU must be in fractions-of-core**, not percent. A machine using 1 full core
  = `1.0`; 50% of a 4-core machine = `2.0`. Divide `cpu_pct` by 100 and
  multiply by `n_cores`.
- If your monitoring system only exposes **CPU %**, set
  `total_cpu = cpu_pct / 100 * n_cores` and `peak_cpu = total_cpu`.
- If intra-bucket **peak** is unavailable, set `peak_cpu = total_cpu` and
  `peak_mem = total_mem`.
- If **disk I/O** is unavailable, set `disk_io = 0.0`.
- If **n_tasks** is unavailable, derive it from CPU load average:
  `n_tasks = max(1, round(load_avg_1min))`.

### Bucket alignment

```python
bucket = unix_timestamp_sec // 300   # integer division
time_us = bucket * 300_000_000
```

`bucket % 288 == 0` at UTC midnight, ensuring `hour_sin`/`hour_cos` features
are correctly phase-aligned.

## Provided adapters

| File | Source | Notes |
|---|---|---|
| `google_cluster.py` | Google Cluster Traces 2011 CSV | Step 1 of training pipeline |
| `zabbix.py` | Zabbix 7.x API | Batch fetch, 89-day history |

## Writing your own adapter

Your adapter must produce a parquet file with the schema above.  Refer to
`google_cluster.py` for a full example.  Minimal skeleton:

```python
import pandas as pd

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """Convert raw monitoring data to cluster_agg parquet.

    Parameters
    ----------
    input_path  : path to your raw data (CSV, API endpoint, TSDB query, ...)
    output_path : where to write the cluster_agg.parquet

    Returns
    -------
    DataFrame with the cluster_agg schema (also written to output_path).
    """
    # 1. Load your raw data
    # 2. Map machine identifiers to contiguous integers
    # 3. Compute bucket = unix_ts // 300
    # 4. Aggregate to one row per (machine_id, bucket)
    # 5. Convert CPU to fractions-of-core
    # 6. Write parquet
    df.to_parquet(output_path, index=False)
    return df
```

After writing your adapter, run the training pipeline from the repo root:

```bash
python training/train.py \
  --data-path data/your_cluster_agg.parquet \
  --artifacts-dir data/your_run \
  --tune-hyperparams --optuna-trials 30
```
