"""
02 - Model Evaluation
=====================
Loads the trained SGD model and evaluates it on 2025 test data.
Reports both classification AND regression metrics (RMSE, MAE, R²).
Addresses IEEE Evaluation Limitation (#4) by using standardized metrics.

Usage:
    python notebooks/02_model_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import gc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.data.multiyear_loader import MultiYearDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator


def find_latest_model_dir():
    """Find the latest experiment results directory."""
    exp_dirs = sorted(Path("experiments").glob("multiyear_results_*"))
    if not exp_dirs:
        print("❌ No trained model found. Run the experiment first:")
        print("   python experiments/run_multiyear_experiments.py")
        sys.exit(1)
    return exp_dirs[-1]


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


def main():
    print("=" * 70)
    print("FLIGHT DELAY PREDICTION — MODEL EVALUATION")
    print("IEEE Standardized Metrics (Limitation #4)")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Load Trained Model
    # ----------------------------------------------------------------
    model_dir = find_latest_model_dir()
    print(f"\n--- Loading model from: {model_dir.name} ---")

    model = joblib.load(model_dir / "sgd_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    engineer = joblib.load(model_dir / "feature_engineer.joblib")

    print(f"  Model: {type(model).__name__}")
    print(f"  Features: {len(feature_names)}")

    # ----------------------------------------------------------------
    # 2. Load Test Data (2025 – first 3 months for speed)
    # ----------------------------------------------------------------
    print("\n--- Loading 2025 Test Data ---")
    loader = MultiYearDataLoader()
    target_gen = TargetGenerator()

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_arr_delay = []  # For regression metrics

    csv_files = loader.get_available_files(2025)[:3]  # Jan-Mar for speed
    print(f"  Using {len(csv_files)} months of 2025 data")

    for csv_file in csv_files:
        print(f"  Processing {csv_file.name}...")

        df = pd.read_csv(csv_file, usecols=loader.ESSENTIAL_COLS, low_memory=False)
        df = normalize_schema(df)

        # Clean
        df = df[(df["CANCELLED"] != 1.0) & (df["DIVERTED"] != 1.0)]
        df = df.dropna(subset=["ARR_DELAY"])

        # Save raw ARR_DELAY before feature engineering (for regression metrics)
        raw_arr_delay = df["ARR_DELAY"].copy()

        # Create target
        df = target_gen.create_target_variables(df)

        # Engineer features (don't refit)
        df = engineer.create_all_features(df, fit_encoders=False)

        # Select features
        X, y = engineer.select_features_for_training(df, target_col="IS_DELAYED")
        X = X.fillna(0)
        X = X.reindex(columns=feature_names, fill_value=0)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        y_pred = model.predict(X_scaled)

        # Probabilities (if available)
        try:
            y_prob = model.predict_proba(X_scaled)[:, 1]
            all_y_prob.extend(y_prob.tolist())
        except Exception:
            pass

        all_y_true.extend(y.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Align ARR_DELAY with valid indices
        aligned_delay = raw_arr_delay.loc[y.index]
        all_arr_delay.extend(aligned_delay.tolist())

        del df, X, X_scaled, y_pred
        gc.collect()

    # ----------------------------------------------------------------
    # 3. Classification Metrics (IEEE Table VII)
    # ----------------------------------------------------------------
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    arr_delay = np.array(all_arr_delay)

    print(f"\n{'=' * 70}")
    print("CLASSIFICATION METRICS (IEEE Table VII)")
    print(f"{'=' * 70}")
    print(f"  Test Samples:  {len(y_true):,}")
    print(f"  Accuracy:      {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision:     {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:        {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:      {f1_score(y_true, y_pred, zero_division=0):.4f}")

    if all_y_prob:
        y_prob_arr = np.array(all_y_prob)
        print(f"  ROC-AUC:       {roc_auc_score(y_true, y_prob_arr):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted On-Time  Predicted Delayed")
    print(f"    Actual On-Time:    {tn:>12,}      {fp:>12,}")
    print(f"    Actual Delayed:    {fn:>12,}      {tp:>12,}")

    # ----------------------------------------------------------------
    # 4. Regression Metrics (Delay Magnitude)
    # ----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("REGRESSION METRICS (Delay Magnitude)")
    print(f"{'=' * 70}")

    # Map binary prediction to estimated delay:
    # Predicted On-Time (0) → 0 min delay estimate
    # Predicted Delayed (1) → Median of actual delayed flights
    median_delay = (
        np.median(arr_delay[arr_delay > 15]) if (arr_delay > 15).any() else 30
    )
    y_pred_delay = np.where(y_pred == 1, median_delay, 0)

    rmse = np.sqrt(mean_squared_error(arr_delay, y_pred_delay))
    mae = mean_absolute_error(arr_delay, y_pred_delay)
    r2 = r2_score(arr_delay, y_pred_delay)

    print(f"  Note: Regression estimated from binary classifier output")
    print(f"        (Predicted Delayed → {median_delay:.0f} min, On-Time → 0 min)")
    print(f"  RMSE:  {rmse:.2f} minutes")
    print(f"  MAE:   {mae:.2f} minutes")
    print(f"  R²:    {r2:.4f}")

    # ----------------------------------------------------------------
    # 5. Feature Importance (SGD coefficients)
    # ----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("TOP 15 FEATURES (by coefficient magnitude)")
    print(f"{'=' * 70}")

    if hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        top_idx = np.argsort(coefs)[::-1][:15]
        for i, idx in enumerate(top_idx):
            print(f"  {i + 1:>2}. {feature_names[idx]:<35} {coefs[idx]:.4f}")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
