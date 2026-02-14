"""
Temporal Drift Detection Module

Detects distribution shifts between training (2023-2024) and test (2025) data.
Novel research contribution for temporal stability analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import json
from pathlib import Path


class DriftDetector:
    """Detects temporal drift in features between train and test sets."""

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize drift detector.

        Args:
            significance_level: P-value threshold for detecting drift (default: 0.05)
        """
        self.significance_level = significance_level
        self.drift_results = {}

    def kolmogorov_smirnov_test(
        self, train_data: pd.Series, test_data: pd.Series
    ) -> Dict:
        """
        Perform Kolmogorov-Smirnov test to detect distribution shift.

        Args:
            train_data: Training data for a feature
            test_data: Test data for the same feature

        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        train_clean = train_data.dropna()
        test_clean = test_data.dropna()

        # Check for empty data
        if len(train_clean) == 0 or len(test_clean) == 0:
            return {
                "statistic": 0.0,
                "pvalue": 1.0,  # No evidence of drift if no data
                "significant": False,
                "train_mean": 0.0,
                "test_mean": 0.0,
                "train_std": 0.0,
                "test_std": 0.0,
                "note": "Insufficient data",
            }

        # Check for empty data
        if len(train_clean) == 0 or len(test_clean) == 0:
            return {
                "statistic": 0.0,
                "pvalue": 1.0,  # No evidence of drift if no data
                "significant": False,
                "train_mean": 0.0,
                "test_mean": 0.0,
                "train_std": 0.0,
                "test_std": 0.0,
                "note": "Insufficient data",
            }

        # KS test
        statistic, pvalue = stats.ks_2samp(train_clean, test_clean)

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": bool(pvalue < self.significance_level),
            "train_mean": float(train_clean.mean()),
            "test_mean": float(test_clean.mean()),
            "train_std": float(train_clean.std()),
            "test_std": float(test_clean.std()),
        }

    def detect_distribution_shift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]
    ) -> Dict:
        """
        Detect distribution shift for multiple features.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            features: List of feature names to test

        Returns:
            Dictionary with drift results for each feature
        """
        print("=" * 70)
        print("TEMPORAL DRIFT DETECTION")
        print("=" * 70)
        print(f"Testing {len(features)} features for distribution shift...")
        print(f"Significance level: {self.significance_level}\n")

        drift_results = {}

        for feature in features:
            if feature not in train_df.columns or feature not in test_df.columns:
                print(f"[SKIP] {feature}: Not found in both datasets")
                continue

            # Only test numeric features
            if not pd.api.types.is_numeric_dtype(train_df[feature]):
                continue

            result = self.kolmogorov_smirnov_test(train_df[feature], test_df[feature])

            drift_results[feature] = result

            # Print significant drifts
            if result["significant"]:
                print(f"[DRIFT] {feature}:")
                print(f"  KS statistic: {result['statistic']:.4f}")
                print(f"  p-value: {result['pvalue']:.4e}")
                print(
                    f"  Train mean: {result['train_mean']:.2f}, Test mean: {result['test_mean']:.2f}"
                )

        # Summary
        n_drifted = sum(1 for r in drift_results.values() if r["significant"])
        n_tested = len(drift_results)

        print(f"\n{'=' * 70}")
        print(f"DRIFT SUMMARY")
        print(f"{'=' * 70}")
        print(f"Features tested: {n_tested}")
        print(
            f"Features with significant drift: {n_drifted} ({n_drifted / n_tested * 100:.1f}%)"
        )
        print(
            f"Stable features: {n_tested - n_drifted} ({(n_tested - n_drifted) / n_tested * 100:.1f}%)"
        )
        print(f"{'=' * 70}\n")

        self.drift_results = drift_results
        return drift_results

    def save_drift_report(self, output_path: Path):
        """Save drift detection results to JSON file."""
        report = {
            "significance_level": self.significance_level,
            "total_features": len(self.drift_results),
            "drifted_features": sum(
                1 for r in self.drift_results.values() if r["significant"]
            ),
            "features": self.drift_results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[OK] Drift report saved to: {output_path}")

    def get_top_drifted_features(self, top_n: int = 10) -> List[tuple]:
        """
        Get features with highest drift.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, ks_statistic) tuples
        """
        # Sort by KS statistic (higher = more drift)
        sorted_features = sorted(
            self.drift_results.items(), key=lambda x: x[1]["statistic"], reverse=True
        )

        return [(name, result["statistic"]) for name, result in sorted_features[:top_n]]


def analyze_temporal_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path = None,
) -> Dict:
    """
    Convenience function to run complete drift analysis.

    Args:
        train_df: Training data (2023-2024)
        test_df: Test data (2025)
        feature_cols: List of features to test
        output_dir: Directory to save report

    Returns:
        Drift detection results
    """
    detector = DriftDetector(significance_level=0.05)

    # Detect drift
    drift_results = detector.detect_distribution_shift(train_df, test_df, feature_cols)

    # Save report if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        detector.save_drift_report(output_dir / "drift_detection_report.json")

    # Print top drifted features
    print("\nTOP 10 DRIFTED FEATURES:")
    print("-" * 70)
    top_drifted = detector.get_top_drifted_features(top_n=10)
    for i, (feature, statistic) in enumerate(top_drifted, 1):
        result = drift_results[feature]
        print(f"{i}. {feature}")
        print(f"   KS statistic: {statistic:.4f}, p-value: {result['pvalue']:.4e}")
        print(f"   Train: {result['train_mean']:.2f} ± {result['train_std']:.2f}")
        print(f"   Test:  {result['test_mean']:.2f} ± {result['test_std']:.2f}")

    return drift_results


if __name__ == "__main__":
    # Test drift detector
    print("Testing Drift Detector\n")

    # Create synthetic data with drift
    np.random.seed(42)

    train_df = pd.DataFrame(
        {
            "feature1": np.random.normal(10, 2, 1000),  # No drift
            "feature2": np.random.normal(5, 1, 1000),  # Drift in mean
            "feature3": np.random.normal(0, 1, 1000),  # Drift in variance
        }
    )

    test_df = pd.DataFrame(
        {
            "feature1": np.random.normal(10, 2, 500),  # Same distribution
            "feature2": np.random.normal(7, 1, 500),  # Mean shifted
            "feature3": np.random.normal(0, 3, 500),  # Variance increased
        }
    )

    # Detect drift
    drift_results = analyze_temporal_drift(
        train_df, test_df, ["feature1", "feature2", "feature3"]
    )
