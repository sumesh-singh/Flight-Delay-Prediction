"""
01 - Data Exploration
=====================
Explores the BTS On-Time Performance dataset.
Addresses IEEE Reproducibility Limitation (#2) by providing transparent data analysis.

Usage:
    python notebooks/01_data_exploration.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.multiyear_loader import MultiYearDataLoader


def main():
    print("=" * 70)
    print("FLIGHT DELAY PREDICTION — DATA EXPLORATION")
    print("=" * 70)

    loader = MultiYearDataLoader()

    # ----------------------------------------------------------------
    # 1. Dataset Overview
    # ----------------------------------------------------------------
    print("\n--- 1. Dataset Inventory ---")
    total_files = 0
    for year in [2023, 2024, 2025]:
        files = loader.get_available_files(year)
        total_files += len(files)
        print(f"  {year}: {len(files)} monthly files found")
    print(f"  Total: {total_files} files")

    # ----------------------------------------------------------------
    # 2. Sample Data Inspection (January 2023)
    # ----------------------------------------------------------------
    print("\n--- 2. Sample Data (Jan 2023) ---")
    sample_files = loader.get_available_files(2023)
    if not sample_files:
        print("  ❌ No 2023 data found. Please download BTS data first.")
        return

    df = pd.read_csv(
        sample_files[0],
        usecols=loader.ESSENTIAL_COLS,
        nrows=100_000,
        low_memory=False,
    )

    print(f"  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}):")
    for col in sorted(df.columns):
        dtype = df[col].dtype
        null_pct = df[col].isnull().mean() * 100
        print(f"    {col:<30} {str(dtype):<10} {null_pct:>5.1f}% null")

    # ----------------------------------------------------------------
    # 3. Target Variable Distribution
    # ----------------------------------------------------------------
    print("\n--- 3. Target Variable (IS_DELAYED: ARR_DELAY > 15 min) ---")
    df_valid = df.dropna(subset=["ArrDelay"])
    df_valid = df_valid[
        (df_valid.get("Cancelled", 0) != 1) & (df_valid.get("Diverted", 0) != 1)
    ]

    # Handle both naming conventions
    delay_col = "ArrDelay" if "ArrDelay" in df_valid.columns else "ARR_DELAY"
    is_delayed = (df_valid[delay_col] > 15).astype(int)

    print(f"  Valid flights: {len(df_valid):,}")
    print(
        f"  On-Time (<=15 min): {(is_delayed == 0).sum():,} ({(is_delayed == 0).mean():.1%})"
    )
    print(
        f"  Delayed (>15 min):  {(is_delayed == 1).sum():,} ({(is_delayed == 1).mean():.1%})"
    )
    print(
        f"  Class ratio: {(is_delayed == 0).sum() / max((is_delayed == 1).sum(), 1):.1f}:1"
    )

    # ----------------------------------------------------------------
    # 4. Delay Distribution Statistics
    # ----------------------------------------------------------------
    print("\n--- 4. Delay Distribution (minutes) ---")
    delay_vals = df_valid[delay_col]
    print(f"  Mean:   {delay_vals.mean():>8.1f} min")
    print(f"  Median: {delay_vals.median():>8.1f} min")
    print(f"  Std:    {delay_vals.std():>8.1f} min")
    print(f"  Min:    {delay_vals.min():>8.1f} min")
    print(f"  Max:    {delay_vals.max():>8.1f} min")
    print(f"  P25:    {delay_vals.quantile(0.25):>8.1f} min")
    print(f"  P75:    {delay_vals.quantile(0.75):>8.1f} min")
    print(f"  P95:    {delay_vals.quantile(0.95):>8.1f} min")

    # ----------------------------------------------------------------
    # 5. Carrier Summary
    # ----------------------------------------------------------------
    print("\n--- 5. Top 10 Carriers by Flight Count ---")
    carrier_col = (
        "Reporting_Airline" if "Reporting_Airline" in df_valid.columns else "OP_CARRIER"
    )
    if carrier_col in df_valid.columns:
        carrier_stats = (
            df_valid.groupby(carrier_col)
            .agg(
                flights=(delay_col, "count"),
                avg_delay=(delay_col, "mean"),
                delay_rate=(delay_col, lambda x: (x > 15).mean()),
            )
            .sort_values("flights", ascending=False)
            .head(10)
        )
        print(f"  {'Carrier':<10} {'Flights':>10} {'Avg Delay':>12} {'Delay Rate':>12}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12}")
        for carrier, row in carrier_stats.iterrows():
            print(
                f"  {carrier:<10} {int(row['flights']):>10,} {row['avg_delay']:>10.1f}m {row['delay_rate']:>11.1%}"
            )

    # ----------------------------------------------------------------
    # 6. Missing Values Summary
    # ----------------------------------------------------------------
    print("\n--- 6. Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col:<30} {count:>8,} ({count / len(df):.1%})")
    else:
        print("  No missing values in sample!")

    print("\n" + "=" * 70)
    print("DATA EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
