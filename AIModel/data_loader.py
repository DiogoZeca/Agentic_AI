"""Load and validate cpu_data.dat."""
from __future__ import annotations

import pandas as pd

_REQUIRED = {"CPUTYPE", "CPUPCT", "NPTS", "SUM", "SUM2", "AVGPOWER"}


def load_cpu_power_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"cpu_data.dat missing columns: {missing}")

    df["CPUPCT"] = df["CPUPCT"].astype(int)
    df["NPTS"]   = df["NPTS"].astype(int)
    for col in ("SUM", "SUM2", "AVGPOWER"):
        df[col] = df[col].astype(float)

    if df["CPUPCT"].lt(0).any() or df["CPUPCT"].gt(100).any():
        raise ValueError("CPUPCT values must be in [0, 100]")
    if df["AVGPOWER"].le(0).any():
        raise ValueError("AVGPOWER must be positive (Watts)")

    return df
