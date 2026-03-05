"""
ModelBenchmark — unified evaluation runner for comparing forecasting models.

Used by carbon_analysis.py and ensemble_analysis.py for consistent comparison tables.
Models that fail evaluation are skipped with a warning — never crash the pipeline.
"""
from __future__ import annotations
import pandas as pd


class ModelBenchmark:
    """Run evaluate() on multiple models and summarise results."""

    def run(
        self,
        models: dict,
        df: pd.DataFrame,
        metric: str,
        train_ratio: float = 0.8,
    ) -> pd.DataFrame:
        """Evaluate each model on the held-out test set.

        Args:
            models:      Dict mapping model_name → object with .evaluate(df, metric)
            df:          Full dataset
            metric:      Target column name
            train_ratio: Train/test split ratio

        Returns:
            DataFrame with columns: model, MAE, MSE, RMSE, MAPE, sMAPE,
            train_size, test_size. Models that raise are skipped.
        """
        rows = []
        for name, model in models.items():
            try:
                result = model.evaluate(df, metric, train_ratio=train_ratio)
                rows.append({"model": name, **result})
            except Exception as exc:
                print(f"  [benchmark] '{name}' evaluation skipped: {exc}")
        return pd.DataFrame(rows)

    def print_table(self, results: pd.DataFrame) -> None:
        """Print a formatted comparison table."""
        if results.empty:
            print("  [benchmark] No results to display.")
            return
        print(f"\n  {'Model':<26} {'sMAPE':>8} {'MAPE':>7} {'RMSE':>12}")
        print(f"  {'-'*56}")
        for _, row in results.iterrows():
            print(
                f"  {row['model']:<26} {row['sMAPE']:>7.1f}%"
                f" {row['MAPE']:>6.1f}% {row['RMSE']:>12.6f}"
            )

    def winner(self, results: pd.DataFrame) -> str:
        """Return the model name with the lowest sMAPE.

        Returns empty string if results is empty.
        """
        if results.empty:
            return ""
        idx = results["sMAPE"].idxmin()
        return str(results.loc[idx, "model"])
