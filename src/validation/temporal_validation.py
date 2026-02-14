"""
Temporal Validation Module

Monitors model performance across time to detect degradation.
Implements sliding window validation for adaptive model assessment.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Dict, List, Optional
import json
from pathlib import Path


class TemporalValidator:
    """Validates model performance across temporal dimensions."""

    def __init__(self, model, feature_cols: List[str]):
        """
        Initialize temporal validator.

        Args:
            model: Trained sklearn model
            feature_cols: List of feature column names
        """
        self.model = model
        self.feature_cols = feature_cols

    def evaluate_by_month(
        self, test_df: pd.DataFrame, target_col: str = "IS_DELAYED"
    ) -> pd.DataFrame:
        """
        Evaluate model performance month-by-month.

        Args:
            test_df: Test DataFrame with 'month' column
            target_col: Name of target column

        Returns:
            DataFrame with monthly performance metrics
        """
        print("=" * 70)
        print("MONTHLY PERFORMANCE ANALYSIS")
        print("=" * 70)

        # Add month column if not present
        if "month" not in test_df.columns and "FL_DATE" in test_df.columns:
            test_df["month"] = pd.to_datetime(test_df["FL_DATE"]).dt.month

        monthly_results = []

        months = sorted(test_df["month"].unique())

        for month in months:
            month_df = test_df[test_df["month"] == month]

            X_month = month_df[self.feature_cols]
            y_month = month_df[target_col]

            # Predict
            y_pred = self.model.predict(X_month)

            # Calculate metrics
            results = {
                "month": f"2025-{month:02d}",
                "month_num": month,
                "n_flights": len(month_df),
                "n_delayed": int(y_month.sum()),
                "delay_rate": float(y_month.mean()),
                "accuracy": float(accuracy_score(y_month, y_pred)),
                "precision": float(precision_score(y_month, y_pred, zero_division=0)),
                "recall": float(recall_score(y_month, y_pred, zero_division=0)),
                "f1": float(f1_score(y_month, y_pred, zero_division=0)),
            }

            monthly_results.append(results)

            print(
                f"\nMonth {month:02d}: F1={results['f1']:.4f}, "
                f"Acc={results['accuracy']:.4f}, "
                f"Flights={results['n_flights']:,}"
            )

        # Convert to DataFrame
        results_df = pd.DataFrame(monthly_results)

        # Detect trend
        if len(results_df) > 1:
            f1_scores = results_df["f1"].values
            months_numeric = np.arange(len(f1_scores))

            # Linear regression to detect trend
            slope = np.polyfit(months_numeric, f1_scores, 1)[0]

            print(f"\n{'=' * 70}")
            print("PERFORMANCE TREND ANALYSIS")
            print(f"{'=' * 70}")
            print(f"Average F1: {f1_scores.mean():.4f}")
            print(f"Std Dev F1: {f1_scores.std():.4f}")
            print(f"Trend: {slope:.4f} F1 per month")

            if slope < -0.01:
                print("[WARNING] Performance degrading over time")
            elif slope > 0.01:
                print("[OK] Performance improving over time")
            else:
                print("[OK] Stable performance")

            print(f"{'=' * 70}\n")

        return results_df

    def evaluate_by_carrier(
        self,
        test_df: pd.DataFrame,
        carrier_col: str = "OP_CARRIER",
        target_col: str = "IS_DELAYED",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Evaluate performance for top carriers.

        Args:
            test_df: Test DataFrame
            carrier_col: Name of carrier column
            target_col: Name of target column
            top_n: Number of top carriers to analyze

        Returns:
            DataFrame with per-carrier metrics
        """
        print("\nCARRIER-SPECIFIC PERFORMANCE")
        print("-" * 70)

        # Get top carriers by flight count
        top_carriers = test_df[carrier_col].value_counts().head(top_n).index

        carrier_results = []

        for carrier in top_carriers:
            carrier_df = test_df[test_df[carrier_col] == carrier]

            X_carrier = carrier_df[self.feature_cols]
            y_carrier = carrier_df[target_col]

            y_pred = self.model.predict(X_carrier)

            results = {
                "carrier": carrier,
                "n_flights": len(carrier_df),
                "delay_rate": float(y_carrier.mean()),
                "f1": float(f1_score(y_carrier, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_carrier, y_pred)),
            }

            carrier_results.append(results)
            print(
                f"{carrier}: F1={results['f1']:.4f} ({results['n_flights']:,} flights)"
            )

        return pd.DataFrame(carrier_results)


class SlidingWindowValidator:
    """Implements sliding window cross-validation for temporal data."""

    def __init__(self, model_class, model_params: Dict, feature_cols: List[str]):
        """
        Initialize sliding window validator.

        Args:
            model_class: Model class (e.g., RandomForestClassifier)
            model_params: Model hyperparameters
            feature_cols: List of feature columns
        """
        self.model_class = model_class
        self.model_params = model_params
        self.feature_cols = feature_cols

    def sliding_window_validation(
        self,
        df: pd.DataFrame,
        target_col: str = "IS_DELAYED",
        window_size: int = 18,
        test_size: int = 3,
        step_size: int = 3,
    ) -> pd.DataFrame:
        """
        Perform sliding window cross-validation.

        Args:
            df: Full DataFrame with temporal ordering
            target_col: Name of target column
            window_size: Number of months for training window
            test_size: Number of months for test window
            step_size: Number of months to slide forward

        Returns:
            DataFrame with validation results for each window
        """
        print("=" * 70)
        print("SLIDING WINDOW VALIDATION")
        print("=" * 70)
        print(f"Window size: {window_size} months")
        print(f"Test size: {test_size} months")
        print(f"Step size: {step_size} months\n")

        # Ensure data is sorted
        df = df.sort_values("FL_DATE").reset_index(drop=True)

        # Create month_id for splitting
        df["year_month"] = pd.to_datetime(df["FL_DATE"]).dt.to_period("M")
        unique_months = sorted(df["year_month"].unique())
        month_to_id = {month: i for i, month in enumerate(unique_months)}
        df["month_id"] = df["year_month"].map(month_to_id)

        total_months = len(unique_months)
        results = []

        # Slide window
        for start in range(0, total_months - window_size - test_size + 1, step_size):
            train_end = start + window_size
            test_end = train_end + test_size

            # Create masks
            train_mask = (df["month_id"] >= start) & (df["month_id"] < train_end)
            test_mask = (df["month_id"] >= train_end) & (df["month_id"] < test_end)

            # Split data
            X_train = df[train_mask][self.feature_cols]
            y_train = df[train_mask][target_col]
            X_test = df[test_mask][self.feature_cols]
            y_test = df[test_mask][target_col]

            # Train model
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            result = {
                "window_id": len(results) + 1,
                "train_start": str(unique_months[start]),
                "train_end": str(unique_months[train_end - 1]),
                "test_start": str(unique_months[train_end]),
                "test_end": str(unique_months[test_end - 1]),
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "f1": float(f1),
                "accuracy": float(acc),
            }

            results.append(result)

            print(
                f"Window {result['window_id']}: "
                f"Train {result['train_start']} to {result['train_end']}, "
                f"Test {result['test_start']} to {result['test_end']}, "
                f"F1={f1:.4f}"
            )

        results_df = pd.DataFrame(results)

        print(f"\n{'=' * 70}")
        print(f"Average F1: {results_df['f1'].mean():.4f}")
        print(f"Std F1: {results_df['f1'].std():.4f}")
        print(f"Min F1: {results_df['f1'].min():.4f}")
        print(f"Max F1: {results_df['f1'].max():.4f}")
        print(f"{'=' * 70}\n")

        return results_df


if __name__ == "__main__":
    # Test temporal validator
    from sklearn.ensemble import RandomForestClassifier

    print("Testing Temporal Validator\n")

    # Create synthetic temporal data
    np.random.seed(42)
    n_samples = 10000

    df = pd.DataFrame(
        {
            "FL_DATE": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "IS_DELAYED": np.random.binomial(1, 0.3, n_samples),
        }
    )

    df["month"] = pd.to_datetime(df["FL_DATE"]).dt.month

    # Train simple model
    feature_cols = ["feature1", "feature2"]
    X = df[feature_cols]
    y = df["IS_DELAYED"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Test monthly evaluation
    validator = TemporalValidator(model, feature_cols)
    monthly_results = validator.evaluate_by_month(df)
    print("\nMonthly Results:")
    print(monthly_results)
