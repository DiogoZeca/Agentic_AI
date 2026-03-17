"""CPU power model trainers — XGBoost and PyTorch MLP.

Both trainers consume the output of feature_engineering.build_features():
  X       : (N × 8) DataFrame — 7 numeric features + CPUTYPE string column
  y_reg   : smooth_power_w target in Watts
  weights : per-CPU-type NPTS-normalised sample weights (each type sums to 1.0)

Both trainers predict power in Watts only.
Spike classification is always derived downstream:
  is_spike = predicted_power_w >= metadata[cpu_type]["spike_threshold_w"]

Cross-validation helpers:
  cv_interpolation — 5-fold KFold: accuracy on known hardware at unseen cpu_pct values
  cv_loco          — leave-one-CPU-type-out: generalisation to unseen hardware

LOCO note on idle_w / dynamic_range_w
--------------------------------------
Test rows for the held-out CPU type keep their pre-computed idle_w and
dynamic_range_w.  This is the correct inference contract: at production time a
brief calibration pass (measure power at 0% and 100% load) provides these
constants for any new hardware before calling predict_power().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Module-level constants ─────────────────────────────────────────────────────

SPARSE_TYPES: frozenset[str] = frozenset({
    "intel-xeon-e5420-disk-raid0",
    "intel-xeon-e5420-ssd-noraid",
    "intel-xeon-e5420-ssd-raid0",
})

_NUMERIC_COLS: list[str] = [
    "cpu_pct",
    "cpu_pct_sq",
    "cpu_pct_cube",
    "sqrt_cpu_pct",
    "log_cpu_pct",
    "idle_w",
    "dynamic_range_w",
]


# ── Neural network architecture ────────────────────────────────────────────────

class _PowerNet(nn.Module):
    """Feed-forward network: n_input → [Linear→BN→ReLU→Drop] × k → 1.

    Defined at module level so that load_state_dict() can reconstruct it
    without pickling the architecture.
    """

    def __init__(
        self,
        n_input: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_input
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (N,)


# ── XGBoost trainer ────────────────────────────────────────────────────────────

class XGBoostPowerTrainer:
    """XGBoost regressor for CPU power prediction.

    CPUTYPE is encoded as a pandas Categorical and handled natively by XGBoost
    (enable_categorical=True, tree_method='hist').

    LOCO behaviour: when the held-out CPU type is unseen in training, XGBoost
    treats it as a missing category value and routes predictions through the
    remaining features (cpu_pct, idle_w, dynamic_range_w) — the correct
    physics-based fallback.
    """

    def __init__(self, *, n_estimators: int = 500) -> None:
        self._n_estimators = n_estimators
        self._model: xgb.XGBRegressor | None = None
        self._categories_: list[str] = []

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y_reg: pd.Series,
        weights: pd.Series,
    ) -> "XGBoostPowerTrainer":
        self._categories_ = sorted(X["CPUTYPE"].astype(str).unique())
        self._model = xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
        )
        self._model.fit(
            self._encode(X),
            y_reg.values,
            sample_weight=weights.values,
        )
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_power(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_power()")
        return self._model.predict(self._encode(X))

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_model(directory / "model.ubj")
        with open(directory / "categories.json", "w") as f:
            json.dump(self._categories_, f)

    @classmethod
    def load(cls, directory: Path | str) -> "XGBoostPowerTrainer":
        directory = Path(directory)
        inst = cls()
        inst._model = xgb.XGBRegressor()
        inst._model.load_model(directory / "model.ubj")
        with open(directory / "categories.json") as f:
            inst._categories_ = json.load(f)
        return inst

    # ── Private ────────────────────────────────────────────────────────────────

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert CPUTYPE to Categorical using the categories seen in fit().

        Values not present in the training categories are set to NaN before
        constructing the Categorical — this avoids the pandas 2.x deprecation
        warning and makes the intent explicit: XGBoost routes unknown CPU types
        through the numeric features (idle_w, dynamic_range_w, cpu_pct).
        """
        X = X.copy()
        cputype_str = X["CPUTYPE"].astype(str)
        X["CPUTYPE"] = pd.Categorical(
            cputype_str.where(cputype_str.isin(self._categories_)),
            categories=self._categories_,
        )
        return X


# ── MLP trainer ────────────────────────────────────────────────────────────────

class MLPPowerTrainer:
    """PyTorch MLP regressor for CPU power prediction.

    Preprocessing (fitted in fit(), applied in predict_power()):
      - StandardScaler  on the 7 numeric features
      - OneHotEncoder   on CPUTYPE — handle_unknown='ignore' produces an
        all-zero row for unseen hardware, so the model falls back to the
        numeric-feature path (correct inference behaviour for new CPU types)

    Training:
      - Full-batch gradient descent (all training rows per epoch).
        Avoids batch-composition randomness from extreme NPTS weight ratios
        (up to 2 587× within some CPU types); 1 002 rows is trivially small
        for one forward pass on CPU.
      - Loss: normalised weighted MSE → sum(w*(pred−y)²) / sum(w)
        Dividing by sum(w) bounds the gradient scale.  Adam's per-parameter
        adaptive LR handles the remaining weight skew naturally.
      - Validation: random 15% split, unweighted MSE only.
        Early stopping should reflect overall prediction quality, not be
        biased towards the highest-confidence measurement buckets.
    """

    # Fixed architecture — tuned for this dataset size
    _HIDDEN_SIZES: list[int] = [128, 64, 32]
    _DROPOUT: float = 0.1
    _LR: float = 1e-3
    _LR_PATIENCE: int = 20
    _LR_FACTOR: float = 0.5
    _LR_MIN: float = 1e-5
    _VAL_FRACTION: float = 0.15

    def __init__(
        self,
        *,
        max_epochs: int = 1000,
        patience: int = 50,
    ) -> None:
        self._max_epochs = max_epochs
        self._patience = patience
        self._model: _PowerNet | None = None
        self._scaler: StandardScaler | None = None
        self._encoder: OneHotEncoder | None = None
        self._n_input: int = 0

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y_reg: pd.Series,
        weights: pd.Series,
    ) -> "MLPPowerTrainer":
        torch.manual_seed(42)
        np.random.seed(42)

        # ── Preprocessing ─────────────────────────────────────────────────────
        self._scaler = StandardScaler()
        self._encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )
        X_num = self._scaler.fit_transform(
            X[_NUMERIC_COLS].values.astype(np.float32)
        ).astype(np.float32)
        X_cat = self._encoder.fit_transform(X[["CPUTYPE"]])
        X_proc: np.ndarray = np.hstack([X_num, X_cat]).astype(np.float32)
        self._n_input = X_proc.shape[1]

        # ── Train / validation split ──────────────────────────────────────────
        N = len(X_proc)
        perm = np.random.permutation(N)
        n_val = max(1, int(self._VAL_FRACTION * N))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_tr = torch.from_numpy(X_proc[train_idx])
        y_tr = torch.from_numpy(y_reg.values.astype(np.float32)[train_idx])
        w_tr = torch.from_numpy(weights.values.astype(np.float32)[train_idx])
        X_va = torch.from_numpy(X_proc[val_idx])
        y_va = torch.from_numpy(y_reg.values.astype(np.float32)[val_idx])

        # ── Model + optimiser ─────────────────────────────────────────────────
        self._model = _PowerNet(self._n_input, self._HIDDEN_SIZES, self._DROPOUT)
        # Initialise the output bias to the training target mean so the network
        # only needs to learn deviations (~±150 W) rather than the absolute
        # scale (~200 W) from scratch.  Without this, convergence requires
        # hundreds of extra epochs just to shift the baseline.
        nn.init.constant_(self._model.net[-1].bias, float(y_reg.mean()))
        optimizer = optim.Adam(self._model.parameters(), lr=self._LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self._LR_PATIENCE,
            factor=self._LR_FACTOR,
            min_lr=self._LR_MIN,
        )

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state: dict[str, Any] = {}
        stale = 0

        for _ in range(self._max_epochs):
            # Training step — normalised weighted MSE
            self._model.train()
            optimizer.zero_grad()
            pred_tr = self._model(X_tr)
            loss = (w_tr * (pred_tr - y_tr) ** 2).sum() / w_tr.sum()
            loss.backward()
            optimizer.step()

            # Validation — unweighted MSE for early stopping signal
            self._model.eval()
            with torch.no_grad():
                pred_va = self._model(X_va)
                val_loss = float(((pred_va - y_va) ** 2).mean())

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self._patience:
                    break

        self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_power(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_power()")
        X_num = self._scaler.transform(
            X[_NUMERIC_COLS].values.astype(np.float32)
        ).astype(np.float32)
        X_cat = self._encoder.transform(X[["CPUTYPE"]])
        X_proc = torch.from_numpy(
            np.hstack([X_num, X_cat]).astype(np.float32)
        )
        self._model.eval()
        with torch.no_grad():
            # Clip to >= 0: power cannot be negative (physical constraint).
            # Prevents near-zero outputs from an undertrained model going negative
            # due to linear output layer initialisation.
            return np.maximum(0.0, self._model(X_proc).numpy())

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), directory / "model.pt")
        joblib.dump(self._scaler, directory / "scaler.pkl")
        joblib.dump(self._encoder, directory / "encoder.pkl")
        config = {
            "hidden_sizes": self._HIDDEN_SIZES,
            "dropout": self._DROPOUT,
            "n_input": self._n_input,
        }
        with open(directory / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: Path | str) -> "MLPPowerTrainer":
        directory = Path(directory)
        with open(directory / "config.json") as f:
            config = json.load(f)
        inst = cls()
        inst._n_input = config["n_input"]
        inst._model = _PowerNet(
            config["n_input"],
            config["hidden_sizes"],
            config["dropout"],
        )
        inst._model.load_state_dict(
            torch.load(
                directory / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        inst._model.eval()
        inst._scaler = joblib.load(directory / "scaler.pkl")
        inst._encoder = joblib.load(directory / "encoder.pkl")
        return inst


# ── Private metric helper ──────────────────────────────────────────────────────

def _fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    metadata: dict,
) -> dict[str, float]:
    """RMSE (W), MAE (W), and spike F1 for one evaluation fold.

    Spike F1 applies the per-CPU-type threshold from metadata to both the
    true and predicted power values, then computes binary F1.
    zero_division=0.0 prevents errors on folds with no predicted spikes.
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    thresholds = (
        X_test["CPUTYPE"]
        .map({ct: meta["spike_threshold_w"] for ct, meta in metadata.items()})
        .values
    )
    spike_f1 = float(
        f1_score(
            y_true >= thresholds,
            y_pred >= thresholds,
            zero_division=0.0,
        )
    )
    return {"rmse": rmse, "mae": mae, "spike_f1": spike_f1}


# ── Cross-validation ───────────────────────────────────────────────────────────

def cv_interpolation(
    trainer_cls: type,
    X: pd.DataFrame,
    y_reg: pd.Series,
    weights: pd.Series,
    metadata: dict,
) -> dict[str, float]:
    """5-fold cross-validation on known CPU types.

    Each fold keeps all 11 CPU types in both train and test splits (random
    row shuffle).  Tests accuracy at cpu_pct values the model has not seen
    for hardware it has been trained on — the dominant production scenario.

    Returns RMSE (W), MAE (W), and spike F1 averaged across 5 folds.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results: list[dict[str, float]] = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y_reg.iloc[train_idx]
        w_tr = weights.iloc[train_idx]

        trainer = trainer_cls()
        trainer.fit(X_tr, y_tr, w_tr)

        y_pred = trainer.predict_power(X_te)
        fold_results.append(
            _fold_metrics(y_reg.iloc[test_idx].values, y_pred, X_te, metadata)
        )

    return {
        k: float(np.mean([r[k] for r in fold_results]))
        for k in fold_results[0]
    }


def cv_loco(
    trainer_cls: type,
    X: pd.DataFrame,
    y_reg: pd.Series,
    weights: pd.Series,
    metadata: dict,
) -> dict:
    """Leave-one-CPU-type-out cross-validation.

    For each of the 11 CPU types: train on the remaining 10, predict on the
    held-out one.  Tests whether the model can generalise to hardware it has
    never seen — it must rely on the physical features (idle_w,
    dynamic_range_w, cpu_pct) rather than the CPU type identity.

    Test rows keep their pre-computed idle_w and dynamic_range_w (see module
    docstring for why this is the correct inference contract, not a data leak).

    Results are split into:
      full_types   — 8 CPU types with dense 0–100% coverage (headline metric)
      sparse_types — 3 storage variants of intel-xeon-e5420, sparser coverage
                     (secondary metric — smaller test folds, higher variance)

    Each group includes a "mean" entry averaging across its members.
    """
    per_type: dict[str, dict[str, float]] = {}

    for cpu_type in sorted(X["CPUTYPE"].unique()):
        test_mask = X["CPUTYPE"] == cpu_type
        X_tr = X[~test_mask]
        y_tr = y_reg[~test_mask]
        w_tr = weights[~test_mask]
        X_te = X[test_mask]

        trainer = trainer_cls()
        trainer.fit(X_tr, y_tr, w_tr)

        y_pred = trainer.predict_power(X_te)
        per_type[cpu_type] = _fold_metrics(
            y_reg[test_mask].values, y_pred, X_te, metadata
        )

    full = {ct: m for ct, m in per_type.items() if ct not in SPARSE_TYPES}
    sparse = {ct: m for ct, m in per_type.items() if ct in SPARSE_TYPES}

    def _mean(d: dict) -> dict[str, float]:
        keys = next(iter(d.values())).keys()
        return {k: float(np.mean([v[k] for v in d.values()])) for k in keys}

    return {
        "full_types":   {**full,   "mean": _mean(full)},
        "sparse_types": {**sparse, "mean": _mean(sparse)},
    }
