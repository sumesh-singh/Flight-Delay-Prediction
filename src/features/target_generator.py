"""
Target Generation Module
Creates target variables for flight delay prediction models

Supports three target types:
1. Binary classification: IS_DELAYED (Yes/No based on threshold)
2. Multiclass classification: DELAY_CATEGORY (Early/OnTime, Minor, Moderate, Severe)
3. Regression: Continuous ARR_DELAY value

Features:
- Configurable delay thresholds
- No data leakage (uses ARR_DELAY only)
- Clean integration with scikit-learn models
- Consistent with config/data_config.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

# Import configuration
try:
    from config.data_config import (
        DELAY_THRESHOLD,
        DELAY_CATEGORIES,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.data_config import (
        DELAY_THRESHOLD,
        DELAY_CATEGORIES,
    )


class TargetGenerator:
    """
    Generate target variables for flight delay prediction

    Creates three types of targets:
    1. Binary: Delayed (1) vs On-Time (0)
    2. Multiclass: 4 categories of delay severity
    3. Regression: Continuous delay in minutes

    All targets derived from ARR_DELAY only (no data leakage)
    """

    def __init__(
        self,
        binary_threshold: int = DELAY_THRESHOLD,
        multiclass_bins: Optional[list] = None,
        multiclass_labels: Optional[list] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize target generator

        Args:
            binary_threshold: Minutes of delay for binary classification (default: 15)
            multiclass_bins: Bin edges for multiclass categories
            multiclass_labels: Labels for multiclass categories
            log_level: Logging level
        """
        self.binary_threshold = binary_threshold

        # Use config defaults if not provided
        if multiclass_bins is None:
            self.multiclass_bins = DELAY_CATEGORIES["bins"]
        else:
            self.multiclass_bins = multiclass_bins

        if multiclass_labels is None:
            self.multiclass_labels = DELAY_CATEGORIES["labels"]
        else:
            self.multiclass_labels = multiclass_labels

        # Validate configuration
        if len(self.multiclass_labels) != len(self.multiclass_bins) - 1:
            raise ValueError(
                f"Number of labels ({len(self.multiclass_labels)}) must be "
                f"one less than bins ({len(self.multiclass_bins)})"
            )

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(
            f"TargetGenerator initialized (binary_threshold={binary_threshold})"
        )

    def create_binary_target(
        self, df: pd.DataFrame, target_col: str = "ARR_DELAY"
    ) -> pd.Series:
        """
        Create binary classification target

        Target: IS_DELAYED
        - 1 = Delayed (ARR_DELAY > threshold)
        - 0 = On-Time (ARR_DELAY <= threshold)

        Args:
            df: DataFrame with arrival delay column
            target_col: Name of delay column (default: 'ARR_DELAY')

        Returns:
            Binary target series (0 or 1)
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        # Check for NaN values (defensive programming)
        n_nan = df[target_col].isna().sum()
        if n_nan > 0:
            self.logger.warning(
                f"Found {n_nan:,} NaN values in {target_col}. "
                "These will be treated as on-time (0). "
                "Consider running data_cleanser first."
            )

        # Create binary target
        is_delayed = (df[target_col] > self.binary_threshold).astype(int)

        # Statistics
        n_delayed = is_delayed.sum()
        n_total = len(is_delayed)
        pct_delayed = (n_delayed / n_total * 100) if n_total > 0 else 0

        self.logger.info(f"Binary target created:")
        self.logger.info(f"  Threshold: {self.binary_threshold} minutes")
        self.logger.info(f"  Delayed: {n_delayed:,} ({pct_delayed:.1f}%)")
        self.logger.info(
            f"  On-Time: {n_total - n_delayed:,} ({100 - pct_delayed:.1f}%)"
        )

        is_delayed.name = "IS_DELAYED"
        return is_delayed

    def create_multiclass_target(
        self, df: pd.DataFrame, target_col: str = "ARR_DELAY"
    ) -> pd.Series:
        """
        Create multiclass classification target

        Default categories (configurable):
        - 0: Early/OnTime (delay <= 0)
        - 1: Minor (0 < delay <= 15)
        - 2: Moderate (15 < delay <= 45)
        - 3: Severe (delay > 45)

        Args:
            df: DataFrame with arrival delay column
            target_col: Name of delay column (default: 'ARR_DELAY')

        Returns:
            Categorical target series (0, 1, 2, 3)
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        # Check for NaN values (defensive programming)
        n_nan = df[target_col].isna().sum()
        if n_nan > 0:
            self.logger.warning(
                f"Found {n_nan:,} NaN values in {target_col}. "
                "These will be filled as category 0 (Early/OnTime). "
                "Consider running data_cleanser first."
            )

        # Create multiclass categories
        delay_category = pd.cut(
            df[target_col],
            bins=self.multiclass_bins,
            labels=range(len(self.multiclass_labels)),
            include_lowest=True,
        )

        # Convert to integer (handle NaN as "Early/OnTime")
        delay_category = delay_category.fillna(0).astype(int)

        # Statistics
        category_counts = delay_category.value_counts().sort_index()

        self.logger.info(f"Multiclass target created:")
        self.logger.info(f"  Bins: {self.multiclass_bins}")
        for idx, label in enumerate(self.multiclass_labels):
            count = category_counts.get(idx, 0)
            pct = (count / len(delay_category) * 100) if len(delay_category) > 0 else 0
            self.logger.info(f"  {idx}: {label:15s} - {count:,} ({pct:.1f}%)")

        delay_category.name = "DELAY_CATEGORY"
        return delay_category

    def create_regression_target(
        self, df: pd.DataFrame, target_col: str = "ARR_DELAY"
    ) -> pd.Series:
        """
        Create regression target (continuous delay)

        Target: ARR_DELAY (continuous minutes)
        - Negative values = Early arrival
        - Zero = Exactly on time
        - Positive values = Delayed arrival

        Args:
            df: DataFrame with arrival delay column
            target_col: Name of delay column (default: 'ARR_DELAY')

        Returns:
            Continuous target series
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        # Check for NaN values (defensive programming)
        n_nan = df[target_col].isna().sum()
        if n_nan > 0:
            self.logger.warning(
                f"Found {n_nan:,} NaN values in {target_col}. "
                "These will cause errors during model training! "
                "Run data_cleanser first to handle missing values."
            )

        arr_delay = df[target_col].copy()

        # Statistics (handle empty DataFrame)
        if len(arr_delay) > 0:
            self.logger.info(f"Regression target created:")
            self.logger.info(f"  Mean delay: {arr_delay.mean():.2f} minutes")
            self.logger.info(f"  Median delay: {arr_delay.median():.2f} minutes")
            self.logger.info(f"  Std deviation: {arr_delay.std():.2f} minutes")
            self.logger.info(
                f"  Min: {arr_delay.min():.2f}, Max: {arr_delay.max():.2f}"
            )
        else:
            self.logger.warning("Empty dataset - no statistics available")

        arr_delay.name = "ARR_DELAY"
        return arr_delay

    def create_all_targets(
        self, df: pd.DataFrame, target_col: str = "ARR_DELAY"
    ) -> pd.DataFrame:
        """
        Create all three target types at once

        Args:
            df: DataFrame with arrival delay column
            target_col: Name of delay column (default: 'ARR_DELAY')

        Returns:
            DataFrame with three target columns:
            - IS_DELAYED (binary)
            - DELAY_CATEGORY (multiclass)
            - ARR_DELAY (regression)
        """
        self.logger.info("=" * 60)
        self.logger.info("CREATING ALL TARGET VARIABLES")
        self.logger.info("=" * 60)

        targets = pd.DataFrame(index=df.index)

        # Binary target
        self.logger.info("\n[1/3] Binary classification target...")
        targets["IS_DELAYED"] = self.create_binary_target(df, target_col)

        # Multiclass target
        self.logger.info("\n[2/3] Multiclass classification target...")
        targets["DELAY_CATEGORY"] = self.create_multiclass_target(df, target_col)

        # Regression target
        self.logger.info("\n[3/3] Regression target...")
        targets["ARR_DELAY"] = self.create_regression_target(df, target_col)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TARGET GENERATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Generated {len(targets.columns)} target variables")
        self.logger.info(f"Sample count: {len(targets):,}")

        return targets

        return {
            "binary_threshold": self.binary_threshold,
            "multiclass_bins": self.multiclass_bins,
            "multiclass_labels": self.multiclass_labels,
            "target_types": ["IS_DELAYED", "DELAY_CATEGORY", "ARR_DELAY"],
        }

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable (IS_DELAYED) and add to DataFrame.
        Used by memory-optimized runner.

        Args:
            df: Input DataFrame with ARR_DELAY

        Returns:
            DataFrame with IS_DELAYED column added
        """
        # Create binary target
        target = self.create_binary_target(df, "ARR_DELAY")

        # Add to DataFrame (modify in-place if possible, but assign new column)
        df["IS_DELAYED"] = target

        return df


def prepare_training_data(
    df: pd.DataFrame,
    target_type: str = "binary",
    binary_threshold: int = DELAY_THRESHOLD,
    exclude_arr_delay: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to prepare X and y for model training

    Args:
        df: DataFrame with features and ARR_DELAY column
        target_type: 'binary', 'multiclass', or 'regression'
        binary_threshold: Threshold for binary classification
        exclude_arr_delay: If True, exclude ARR_DELAY from features (prevents leakage)
                          Note: DEP_DELAY, IS_DELAYED, and DELAY_CATEGORY are always excluded

    Returns:
        Tuple of (X features, y target)
    """
    generator = TargetGenerator(binary_threshold=binary_threshold)

    # Create appropriate target
    if target_type == "binary":
        y = generator.create_binary_target(df)
    elif target_type == "multiclass":
        y = generator.create_multiclass_target(df)
    elif target_type == "regression":
        y = generator.create_regression_target(df)
    else:
        raise ValueError(
            f"Invalid target_type: {target_type}. Must be 'binary', 'multiclass', or 'regression'"
        )

    # Prepare features (exclude target column to prevent leakage)
    exclude_cols = ["ARR_DELAY"] if exclude_arr_delay else []

    # Also exclude other target columns if they exist
    exclude_cols.extend(["IS_DELAYED", "DELAY_CATEGORY", "DEP_DELAY"])

    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    return X, y


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("=" * 60)
    print("TARGET GENERATOR MODULE TEST")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Simulate realistic delay distribution
    # Most flights on time or slightly delayed, some severe delays
    delays = np.concatenate(
        [
            np.random.normal(-5, 10, int(n_samples * 0.5)),  # 50% early/on-time
            np.random.normal(10, 5, int(n_samples * 0.3)),  # 30% minor delays
            np.random.normal(30, 10, int(n_samples * 0.15)),  # 15% moderate delays
            np.random.normal(90, 30, int(n_samples * 0.05)),  # 5% severe delays
        ]
    )

    sample_data = pd.DataFrame(
        {
            "ARR_DELAY": delays,
            "DISTANCE": np.random.randint(200, 2500, n_samples),
            "hour": np.random.randint(0, 24, n_samples),
        }
    )

    print(f"\nSample data created: {sample_data.shape}")
    print(f"ARR_DELAY distribution:")
    print(sample_data["ARR_DELAY"].describe())

    # Test target generation
    generator = TargetGenerator(binary_threshold=15)

    # Generate all targets
    targets = generator.create_all_targets(sample_data)

    print(f"\nTargets generated: {targets.shape}")
    print("\nTarget columns:")
    print(targets.head(10))

    # Test convenience function
    print("\n" + "=" * 60)
    print("Testing prepare_training_data()...")
    print("=" * 60)

    X, y = prepare_training_data(sample_data, target_type="binary")
    print(f"\nBinary classification:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    X, y = prepare_training_data(sample_data, target_type="multiclass")
    print(f"\nMulticlass classification:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Class distribution: {y.value_counts().sort_index().to_dict()}")

    X, y = prepare_training_data(sample_data, target_type="regression")
    print(f"\nRegression:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Target stats: mean={y.mean():.2f}, std={y.std():.2f}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
