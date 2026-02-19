"""
03 - Model Comparison & Ablation Study
=======================================
Trains Random Forest, Logistic Regression, and SGDClassifier on sampled data.
Compares classification AND regression metrics.
Performs ablation study: flight-only features vs full multi-source features.

Addresses:
  - Gap #1: Abstract claims RF achieves 83.2% accuracy; we verify with actual RF
  - Gap #4: RF vs LR vs SGD comparison
  - Gap #5: Ablation study (flight-only vs multi-source)

Usage:
    python notebooks/03_model_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import gc
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from src.data.multiyear_loader import MultiYearDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_SIZE = 500_000  # Rows for training (memory-safe for RF)
TEST_SIZE = 100_000  # Rows for testing
RANDOM_STATE = 42

# Feature groups for ablation study
FLIGHT_ONLY_FEATURES = [
    "CRS_DEP_MINUTES",
    "DISTANCE",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_peak_hour",
    "is_weekend",
    "OP_CARRIER_encoded",
    "ORIGIN_encoded",
    "DEST_encoded",
]

WEATHER_FEATURES = [
    "ORIGIN_TEMP",
    "ORIGIN_VISIBILITY",
    "ORIGIN_WIND_SPEED",
    "ORIGIN_PRECIP",
    "DEST_TEMP",
    "DEST_VISIBILITY",
    "DEST_WIND_SPEED",
    "DEST_PRECIP",
]

NETWORK_FEATURES = [
    "prev_flight_delay",
    "turnaround_stress",
]

CARRIER_AIRPORT_FEATURES = [
    "carrier_delay_rate",
    "carrier_avg_delay",
    "carrier_punctuality",
    "origin_delay_rate",
    "dest_delay_rate",
    "origin_flight_count",
    "dest_flight_count",
    "route_delay_rate",
    "route_avg_delay",
    "route_flight_count",
]


def normalize_schema(df):
    """Normalize BTS column names."""
    mapping = {
        "FlightDate": "FL_DATE",
        "Reporting_Airline": "OP_CARRIER",
        "Tail_Number": "TAIL_NUM",
        "Origin": "ORIGIN",
        "Dest": "DEST",
        "CRSDepTime": "CRS_DEP_TIME",
        "CRSArrTime": "CRS_ARR_TIME",
        "DepTime": "DEP_TIME",
        "ArrTime": "ARR_TIME",
        "DepDelay": "DEP_DELAY",
        "ArrDelay": "ARR_DELAY",
        "Cancelled": "CANCELLED",
        "Diverted": "DIVERTED",
        "Distance": "DISTANCE",
    }
    return df.rename(columns=mapping)


def load_sampled_data():
    """Load and sample data for tractable RF training."""
    print("--- Loading and sampling data ---")
    loader = MultiYearDataLoader()

    # Training: sample from 2023
    train_files = loader.get_available_files(2023)[:4]  # Jan-Apr
    test_files = loader.get_available_files(2025)[:1]  # Jan 2025

    if not train_files or not test_files:
        print("ERROR: Missing data files. Ensure 2023 and 2025 BTS data is downloaded.")
        sys.exit(1)

    # Load training data
    train_dfs = []
    for f in train_files:
        df = pd.read_csv(f, usecols=loader.ESSENTIAL_COLS, low_memory=False)
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    gc.collect()

    # Load test data
    test_df = pd.read_csv(
        test_files[0], usecols=loader.ESSENTIAL_COLS, low_memory=False
    )

    # Normalize
    train_df = normalize_schema(train_df)
    test_df = normalize_schema(test_df)

    # Clean
    for df in [train_df, test_df]:
        mask = (
            (df["CANCELLED"] != 1.0) & (df["DIVERTED"] != 1.0) & df["ARR_DELAY"].notna()
        )
        df.drop(df[~mask].index, inplace=True)

    # Sample
    if len(train_df) > SAMPLE_SIZE:
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    if len(test_df) > TEST_SIZE:
        test_df = test_df.sample(n=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"  Training: {len(train_df):,} samples")
    print(f"  Testing:  {len(test_df):,} samples")

    return train_df, test_df


def engineer_features(train_df, test_df):
    """Apply feature engineering."""
    print("--- Engineering features ---")
    engineer = FeatureEngineer(use_external_data=False)  # No external API calls
    target_gen = TargetGenerator()

    train_df = target_gen.create_target_variables(train_df)
    test_df = target_gen.create_target_variables(test_df)

    train_df = engineer.create_all_features(train_df, fit_encoders=True)
    test_df = engineer.create_all_features(test_df, fit_encoders=False)

    X_train, y_train = engineer.select_features_for_training(
        train_df, target_col="IS_DELAYED"
    )
    X_test, y_test = engineer.select_features_for_training(
        test_df, target_col="IS_DELAYED"
    )

    # Save ARR_DELAY for regression
    arr_delay_train = train_df.loc[y_train.index, "ARR_DELAY"].values
    arr_delay_test = test_df.loc[y_test.index, "ARR_DELAY"].values

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Align columns
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"  Features: {len(common_cols)}")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        arr_delay_train,
        arr_delay_test,
        common_cols,
    )


def evaluate_model(
    name, model, X_train, y_train, X_test, y_test, arr_delay_test, scaler=None
):
    """Train and evaluate a single model."""
    print(f"\n  Training {name}...")
    start = time.time()

    X_tr = scaler.transform(X_train) if scaler else X_train.values
    X_te = scaler.transform(X_test) if scaler else X_test.values

    model.fit(X_tr, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_te)

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Regression metrics from binary prediction
    median_delay = (
        np.median(arr_delay_test[arr_delay_test > 15])
        if (arr_delay_test > 15).any()
        else 30
    )
    y_pred_delay = np.where(y_pred == 1, median_delay, 0)
    rmse = np.sqrt(mean_squared_error(arr_delay_test, y_pred_delay))
    mae = mean_absolute_error(arr_delay_test, y_pred_delay)

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "rmse": rmse,
        "mae": mae,
        "train_time": train_time,
    }


def run_ablation(X_train, y_train, X_test, y_test, arr_delay_test, all_features):
    """Run ablation study: flight-only vs full features."""
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY: Feature Group Impact")
    print(f"{'=' * 70}")

    # Define feature sets
    flight_only = [f for f in FLIGHT_ONLY_FEATURES if f in all_features]
    flight_plus_carrier = flight_only + [
        f for f in CARRIER_AIRPORT_FEATURES if f in all_features
    ]
    flight_plus_network = flight_plus_carrier + [
        f for f in NETWORK_FEATURES if f in all_features
    ]
    full_features = all_features  # Everything available

    feature_sets = {
        "Flight-Only (Temporal + IDs)": flight_only,
        "+ Carrier/Airport Stats": flight_plus_carrier,
        "+ Network Propagation": flight_plus_network,
        "Full (All Features)": full_features,
    }

    print(
        f"\n  {'Feature Set':<35} {'# Feat':>6} {'Accuracy':>10} {'F1':>8} {'RMSE':>8} {'Delta':>8}"
    )
    print(f"  {'-' * 35} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}")

    baseline_acc = None

    for set_name, feat_list in feature_sets.items():
        if not feat_list:
            print(f"  {set_name:<35} {'0':>6}  (skipped -- no features available)")
            continue

        X_tr = X_train[feat_list].fillna(0)
        X_te = X_test[feat_list].fillna(0)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_tr_scaled, y_train)
        y_pred = rf.predict(X_te_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        median_delay = (
            np.median(arr_delay_test[arr_delay_test > 15])
            if (arr_delay_test > 15).any()
            else 30
        )
        y_pred_delay = np.where(y_pred == 1, median_delay, 0)
        rmse = np.sqrt(mean_squared_error(arr_delay_test, y_pred_delay))

        if baseline_acc is None:
            baseline_acc = acc
            delta = "--"
        else:
            delta = f"+{(acc - baseline_acc) * 100:.1f}%"

        print(
            f"  {set_name:<35} {len(feat_list):>6} {acc:>10.4f} {f1:>8.4f} {rmse:>7.1f}m {delta:>8}"
        )

        del rf, X_tr_scaled, X_te_scaled
        gc.collect()


def main():
    print("=" * 70)
    print("FLIGHT DELAY PREDICTION - MODEL COMPARISON & ABLATION STUDY")
    print("=" * 70)

    # Load data
    train_df, test_df = load_sampled_data()

    # Engineer features
    X_train, y_train, X_test, y_test, arr_delay_train, arr_delay_test, feature_cols = (
        engineer_features(train_df, test_df)
    )
    del train_df, test_df
    gc.collect()

    # ================================================================
    # PART 1: Model Comparison (RF vs LR vs SGD)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON (RF vs LR vs SGD)")
    print(f"{'=' * 70}")

    scaler = StandardScaler()
    scaler.fit(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SGD (Online Learning)": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        result = evaluate_model(
            name, model, X_train, y_train, X_test, y_test, arr_delay_test, scaler
        )
        results.append(result)

    # Print comparison table
    print(
        f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'RMSE':>8} {'Time':>8}"
    )
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        print(
            f"  {r['name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['rmse']:>7.1f}m {r['train_time']:>7.1f}s"
        )

    # ================================================================
    # PART 2: Regression Model (Delay Magnitude)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("REGRESSION: Delay Magnitude Prediction (Random Forest Regressor)")
    print(f"{'=' * 70}")

    X_tr_scaled = scaler.transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    rfr = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    rfr.fit(X_tr_scaled, arr_delay_train)
    y_pred_reg = rfr.predict(X_te_scaled)

    rmse = np.sqrt(mean_squared_error(arr_delay_test, y_pred_reg))
    mae = mean_absolute_error(arr_delay_test, y_pred_reg)
    r2 = r2_score(arr_delay_test, y_pred_reg)

    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  MAE:  {mae:.2f} minutes")
    print(f"  R²:   {r2:.4f}")

    del rfr, X_tr_scaled, X_te_scaled
    gc.collect()

    # ================================================================
    # PART 3: Ablation Study
    # ================================================================
    run_ablation(X_train, y_train, X_test, y_test, arr_delay_test, feature_cols)

    print(f"\n{'=' * 70}")
    print("COMPARISON & ABLATION STUDY COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
