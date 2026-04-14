## Current State (2026-04-14)

---

## What is running right now

**GPU VM training run in progress — Phase 6 (fresh Optuna, 39 features).**

The previous run (Phase 6 intermediate) used the new streak features but with stale Optuna
hyperparameters from the 37-feature run, causing a regression in the severe class.
This run deletes all Optuna databases and re-tunes from scratch.

What was changed before starting this run:
- Dropped `current_severe_streak` (dead last in gain 0.0024 and SHAP 0.0013)
- Kept `current_spike_streak` (rank 26) and `max_spike_streak_24h` (rank 13, SHAP 0.035)
- Deleted `cluster_features.parquet` (feature set changed)
- Deleted all three `optuna.db` files (hyperparams stale after feature set change)

---

## When the training run finishes

### Step 1 — Fetch artifacts from VM

```bash
rsync -av <user>@<vm-ip>:~/spike/AIModel/data/full_run/ \
  /home/diogozeca/Documents/Agentic_AI/AIModel/data/full_run/
```

### Step 2 — Check results

```bash
python3 -c "
import json, pathlib

for name, path in [
    ('60m severity', 'AIModel/data/full_run/models/spike/spike_config.json'),
    ('15m binary',   'AIModel/data/full_run/models/spike_15m/spike_config.json'),
    ('OVR severe',   'AIModel/data/full_run/models/spike_severe_ovr/spike_config.json'),
]:
    cfg = json.loads(pathlib.Path(path).read_text())
    cv = cfg['cv_summary']
    fm = cfg.get('final_metrics', {})
    pr = fm.get('macro_pr_auc_calibrated') or fm.get('pr_auc', '?')
    print(f'{name}')
    print(f'  CV PR-AUC:   {cv[\"cv_macro_pr_auc_mean\"]:.4f} ± {cv[\"cv_macro_pr_auc_std\"]:.4f}')
    print(f'  Test PR-AUC: {pr}')
"
```

### Step 3 — Decide outcome

| Outcome | Action |
|---------|--------|
| 60m ≥ 0.574 cal **and** severe ≥ 0.330 | Streak features confirmed — proceed to K-of-N ablation |
| Improvement but not full recovery | Accept and move to next improvement (FFT or focal loss) |
| Regression vs Phase 5 | Drop streak features entirely, revert to 37-feature set |

### Step 4 — Run test suite (verify round-trip tests pass with new artifacts)

```bash
cd AIModel && .venv/bin/python3 -m pytest tests/ -v
# Expect: 262 pass, 0 fail
```

**Note:** Until new artifacts are synced, 4 `TestRoundTripRegression` tests in `test_predict_spike.py`
will fail with "Feature DataFrame is missing 1 required column(s): ['current_severe_streak']".
This is expected — the local model artifact is stale (trained on 40 features, code now has 39).
They go green automatically once the new artifacts are rsync'd back from the VM.

---

## Performance history (all completed runs)

| Run | Features | Hyperparams | 60m CV PR-AUC | 60m Test PR-AUC | Severe Test | Notes |
|-----|----------|-------------|---------------|-----------------|-------------|-------|
| Phase 5 | 37 | Tuned (150+75) | 0.559 ± 0.008 | 0.574 cal | 0.330 | Reference baseline |
| Phase 6 intermediate | 40 | **Stale from Phase 5** | 0.567 ± 0.009 | 0.554 cal | 0.277 | Included `current_severe_streak`; Optuna not re-run |
| **Phase 6 current** | **39** | **Fresh (150+75)** | TBD | TBD | TBD | `current_severe_streak` dropped |

Target: 60m ≥ 0.574 cal, severe ≥ 0.330.

---

## If results are good: K-of-N ablation

The training pipeline supports `--min-future-windows K` (K=1,2,3).
K=1 is the standard label: fires if any 1 of the next 12 windows is a spike.
K=2 requires 2 of the next 12; K=3 requires 3.
Higher K = stricter definition of "real spike" = fewer but more persistent positives.

**Reuse the Step 1 cache (cluster_agg.parquet) — only re-run from Step 2:**

```bash
# K=2 run
mkdir -p ~/spike/AIModel/data/full_run_k2
cp ~/spike/AIModel/data/full_run/cluster_agg.parquet ~/spike/AIModel/data/full_run_k2/

docker compose --profile train run --rm train \
  python3 -u train_spike_classifier.py \
    --data-path data/cluster_cpu_data.csv \
    --artifacts-dir data/full_run_k2 \
    --tune-hyperparams --optuna-trials 150 \
    --device cuda --from-step 2 --min-future-windows 2

# K=3 run
mkdir -p ~/spike/AIModel/data/full_run_k3
cp ~/spike/AIModel/data/full_run/cluster_agg.parquet ~/spike/AIModel/data/full_run_k3/

docker compose --profile train run --rm train \
  python3 -u train_spike_classifier.py \
    --data-path data/cluster_cpu_data.csv \
    --artifacts-dir data/full_run_k3 \
    --tune-hyperparams --optuna-trials 150 \
    --device cuda --from-step 2 --min-future-windows 3
```

**Compare results:**

```bash
python3 -c "
import json, pathlib
for k in (1, 2, 3):
    p = pathlib.Path(f'AIModel/data/full_run_k{k}/models/spike/spike_config.json')
    if p.exists():
        cfg = json.loads(p.read_text())
        cv = cfg['cv_summary']
        fm = cfg.get('final_metrics', {})
        print(f'K={k}  CV: {cv[\"cv_macro_pr_auc_mean\"]:.4f}±{cv[\"cv_macro_pr_auc_std\"]:.4f}  '
              f'Test: {fm.get(\"macro_pr_auc_calibrated\", \"?\"):.4f}  '
              f'Severe: {fm.get(\"pr_auc_class_2\", \"?\"):.4f}')
"
```

---

## Next improvements (priority order, not yet started)

### 1. FFT spectral features
Add dominant frequency and spectral energy bands from the 24-bucket CPU window.
Captures periodic load patterns (e.g., hourly batch jobs) that rolling stats miss.
- Input: last 24 `total_cpu` values per machine-bucket
- Features: dominant frequency index, top-3 energy bands (via `np.fft.rfft`)
- Implementation: in `_engineer_machine()`, alongside existing rolling features
- No leakage risk: uses only the current 24-bucket window, no future data

### 2. Focal loss for binary models
Two distinct severe-class metrics (do not confuse them):
- **60m model, severe class PR-AUC** — the per-class score inside the 3-class model.
  Phase 5 baseline: 0.330. Phase 6 intermediate (stale Optuna): dropped to 0.277.
  Phase 6 current: target ≥ 0.330 (should recover with fresh Optuna).
- **OVR severe model PR-AUC** — the standalone binary classifier (severe vs rest).
  Stable across runs: 0.362. Target with focal loss: push to ≥ 0.40.

Focal loss down-weights easy negatives and focuses training on hard positives — beneficial
for the rare severe class (3.3% of training data). XGBoost supports custom objectives.
- `spike_severe_ovr` is the primary target (rarest class, most to gain)
- `spike_in_15m` may also benefit (8.9% positives)
- Use `gamma=2.0` as starting point (standard focal loss parameter)
- Requires custom `obj` and `metric` functions in `BinarySpikeClassifier`

### 3. Zabbix cross-domain evaluation
A real production Slurm HPC cluster. Goal: measure how well the Google-trained model
generalises to real production data. This is a POC/evaluation — NOT for retraining.

**What is already done (2026-04-13):**
- `explore_zabbix.py` ran successfully
- Output saved in `AIModel/data/zabbix_explore_output.txt`
- Confirmed: 11 nodes, 89 days, 5-min interval, same cadence as Google data
- `n_tasks` proxy: `load_avg` (available for all nodes)

**Phase 3a — fetch_zabbix_data.py (NEXT STEP):**
- Write script to pull full metric history from Zabbix API
- Output format: `cluster_agg.parquet` compatible (machine_id, bucket, total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks)
- 89 days × 11 nodes × 288 buckets/day ≈ 275K rows (tiny vs Google's 24M)

**Phase 3b — domain shift assessment:**
- Compare Zabbix CPU distribution vs Google Cluster 2011
- Key check: are p95/p99 thresholds meaningful in the Zabbix context?
- Compute PSI (Population Stability Index) per feature

**Phase 3c — model evaluation:**
- Run inference on Zabbix windows using the existing trained model
- Primary metric: ROC-AUC (baseline always 0.5, regardless of spike rate — useful for cross-domain)
- PR-AUC will differ because Zabbix spike rate may not match Google's ~15%

### 4. Google 2019 dataset via GCP BigQuery
3–25× more training data than the 2011 dataset. More machines, longer duration.
You have a GCP account. The dataset is in BigQuery: `google.com:google-cluster-data`.
- First: confirm schema is compatible with the preprocessor (bucket aggregation approach)
- Cost: BigQuery free tier is 1TB/month of queries; the dataset is ~TB scale — check cost before querying
- Integration: add a `download_cluster_data_2019.py` next to the existing 2011 downloader
- If schema matches, Step 1 (preprocessor) runs unchanged; Steps 2+3 also unchanged

---

## Files changed in this session (Phase 6)

| File | Change | Why |
|------|--------|-----|
| `spike_feature_engineer.py` | Added `current_spike_streak`, `max_spike_streak_24h` to `_FEATURE_COLS` + `_engineer_machine()` | Persistence signal: consecutive spikes are more likely to continue |
| `spike_feature_engineer.py` | Added `_max_consecutive_run()` helper | Reusable O(n) consecutive-run counter used by `max_spike_streak_24h` |
| `spike_feature_engineer.py` | Added `min_future_windows: int = 1` to `_add_label()` + `engineer()` | K-of-N label ablation: K=2 requires 2 of 12 future windows to spike |
| `spike_feature_engineer.py` | **Dropped `current_severe_streak`** | Dead last in gain (0.0024) and SHAP (0.0013) — no contribution |
| `spike_classifier.py` | Added `current_spike_streak`, `max_spike_streak_24h` to `_X_COLS` + `_BINARY_MONOTONE_MAP` | Match new feature set |
| `spike_classifier.py` | **Removed `current_severe_streak`** from `_X_COLS` + `_BINARY_MONOTONE_MAP` | Dropped feature |
| `train_spike_classifier.py` | Added `--min-future-windows` CLI arg + wired through `run()` → `engineer()` | K-of-N ablation support |
| `Dockerfile.training` | Added `ENV MIN_FUTURE_WINDOWS=1`, wired to `--min-future-windows` in CMD | K-of-N ablation support |
| `tests/test_spike_feature_engineer.py` | Added `TestStreakFeatures` (8 tests) + `TestAddLabelMinFutureWindows` (8 tests) | Test coverage for new features |
| `tests/test_spike_feature_engineer.py` | Removed `test_severe_streak_zero_when_p99_not_exceeded` | Tests dropped feature |
| `tests/test_train_spike_classifier.py` | Added `1e-9` tolerance to AUC upper-bound assertion | Float64 epsilon: `average_precision_score` can return 1.0+ε on tiny data |

---

## Completed products (already shipped — do not rebuild unless asked)

| Product | File | Status | Notes |
|---------|------|--------|-------|
| Batch inference | `predict_spike.py` | ✅ Complete | Outputs batch_id, trained_at, data_quality, recommended_action, top_features (SHAP), alarm debounce |
| REST API | `spike_api.py` | ✅ Complete | FastAPI POST /predict, ALARM_MIN_CONSECUTIVE env var |
| Streamlit dashboard | `demo.py` | ✅ Complete | 4 tabs: Overview, Threshold Explorer, Calibration, Feature Attribution |
| Grafana ops dashboard | — | ❌ Not started | Deprioritised; focus is on model quality |

These read artifacts from `data/full_run/` — no code changes needed, just update the artifacts.

---

## VM sync procedure (after any code change)

```bash
# 1. Push code
rsync -av --progress \
  --exclude='.venv/' --exclude='data/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.pyc' \
  /home/diogozeca/Documents/Agentic_AI/AIModel/ \
  <user>@<vm-ip>:~/spike/AIModel/

# 2. If _FEATURE_COLS or _X_COLS changed — delete Step 2 cache
rm ~/spike/AIModel/data/full_run/cluster_features.parquet

# 3. If _X_COLS changed — delete all Optuna databases (hyperparams are stale)
rm ~/spike/AIModel/data/full_run/models/spike/optuna.db
rm ~/spike/AIModel/data/full_run/models/spike_15m/optuna.db
rm ~/spike/AIModel/data/full_run/models/spike_severe_ovr/optuna.db

# 4. Run training in tmux (required — training takes hours, tmux survives SSH disconnect)
tmux new -s train   # or: tmux attach -t train
cd ~/spike
docker compose --profile train run --rm --build train

# 5. Monitor (in a second tmux pane)
tail -f ~/spike/AIModel/data/run_log.txt

# 6. Fetch artifacts back
rsync -av <user>@<vm-ip>:~/spike/AIModel/data/full_run/ \
  /home/diogozeca/Documents/Agentic_AI/AIModel/data/full_run/
```
