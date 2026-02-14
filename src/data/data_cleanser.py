"""
Data Cleansing Module
Implements comprehensive data cleaning pipeline for BTS flight data

Features:
- Removal of cancelled and diverted flights
- Domain-aware missing value handling
- Duplicate detection and removal
- Multi-method outlier detection (Isolation Forest, LOF, 3-sigma)
- Detailed logging of data quality metrics
- Optional Parquet persistence

Output: Clean, stable schema ready for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging
from datetime import datetime

# Import configuration
try:
    from config.data_config import (
        PROCESSED_DATA_DIR,
        OUTLIER_CONTAMINATION,
        OUTLIER_DETECTION_COLUMNS,
        MISSING_VALUE_STRATEGY,
        VALIDATION_RULES,
        PARQUET_COMPRESSION,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.data_config import (
        PROCESSED_DATA_DIR,
        OUTLIER_CONTAMINATION,
        OUTLIER_DETECTION_COLUMNS,
        MISSING_VALUE_STRATEGY,
        VALIDATION_RULES,
        PARQUET_COMPRESSION,
    )


class DataCleanser:
    """
    Comprehensive data cleaning pipeline for flight delay prediction

    Implements IEEE paper best practices for data quality:
    - Systematic removal of invalid records
    - Domain-aware imputation strategies
    - Multiple outlier detection methods
    - Audit trail of all transformations

    Output Schema (21 columns):
    - FL_DATE: Flight date (datetime64)
    - OP_CARRIER: Operating carrier (object)
    - OP_CARRIER_FL_NUM: Flight number (int64)
    - ORIGIN, DEST: Airport codes (object)
    - CRS_DEP_TIME, DEP_TIME: Times in minutes since midnight (float64)
    - CRS_ARR_TIME, ARR_TIME: Times in minutes since midnight (float64)
    - CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME: Duration (float64)
    - DEP_DELAY, ARR_DELAY: Delay in minutes (float64) - TARGET
    - CANCELLED, DIVERTED: Status flags (int64) - REMOVED after cleaning
    - DISTANCE: Miles (float64)
    - CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY,
      LATE_AIRCRAFT_DELAY: Delay attribution (float64)
    """

    def __init__(
        self,
        outlier_method: str = "isolation_forest",
        contamination: float = OUTLIER_CONTAMINATION,
        log_level: str = "INFO",
    ):
        """
        Initialize data cleanser

        Args:
            outlier_method: 'isolation_forest', 'lof', or '3sigma'
            contamination: Expected outlier fraction (default: 0.05 = 5%)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.outlier_method = outlier_method
        self.contamination = contamination

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

        # Track cleaning statistics
        self.stats = {
            "original_records": 0,
            "cancelled_removed": 0,
            "diverted_removed": 0,
            "missing_values_removed": 0,
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "final_records": 0,
            "quality_score": 0.0,
        }

    def remove_cancelled_diverted(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cancelled and diverted flights

        Rationale: Cannot predict delays for flights that didn't operate normally

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cancelled/diverted flights removed
        """
        initial_count = len(df)

        # Remove cancelled flights
        cancelled_mask = df["CANCELLED"] == 1
        self.stats["cancelled_removed"] = cancelled_mask.sum()

        # Remove diverted flights
        diverted_mask = df["DIVERTED"] == 1
        self.stats["diverted_removed"] = diverted_mask.sum()

        # Keep only normal flights
        df_clean = df[(~cancelled_mask) & (~diverted_mask)].copy()

        self.logger.info(
            f"Removed {self.stats['cancelled_removed']:,} cancelled flights"
        )
        self.logger.info(f"Removed {self.stats['diverted_removed']:,} diverted flights")
        self.logger.info(f"Remaining: {len(df_clean):,} records")

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using domain-specific logic

        Strategy:
        1. Critical columns (DEP_DELAY, ARR_DELAY): Drop rows
        2. Temporal columns (times): Drop rows (can't impute schedules)
        3. Delay causes: Fill with 0 (missing = no delay from that cause)
        4. Other numeric: Forward fill or median

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        initial_count = len(df)

        # 1. Drop if critical delay metrics are missing
        critical_cols = MISSING_VALUE_STRATEGY["critical_columns"]
        df_clean = df.dropna(subset=critical_cols).copy()

        critical_missing = initial_count - len(df_clean)
        self.logger.info(
            f"Dropped {critical_missing:,} rows with missing delay metrics"
        )

        # 2. Drop if temporal columns are missing
        temporal_cols = MISSING_VALUE_STRATEGY["temporal_columns"]
        existing_temporal = [col for col in temporal_cols if col in df_clean.columns]
        before_temporal = len(df_clean)
        df_clean = df_clean.dropna(subset=existing_temporal)
        after_temporal = len(df_clean)

        temporal_missing = before_temporal - after_temporal
        if temporal_missing > 0:
            self.logger.info(f"Dropped {temporal_missing:,} rows with missing times")

        # 3. Fill delay causes with 0 (missing = no delay from that source)
        delay_causes = MISSING_VALUE_STRATEGY["delay_causes"]
        for col in delay_causes:
            if col in df_clean.columns:
                missing_count = df_clean[col].isna().sum()
                if missing_count > 0:
                    df_clean[col] = df_clean[col].fillna(0)
                    self.logger.debug(
                        f"Filled {missing_count:,} missing values in {col} with 0"
                    )

        # 4. Handle remaining missing values (should be minimal)
        remaining_missing = df_clean.isna().sum()
        if remaining_missing.any():
            self.logger.warning(
                f"Remaining missing values:\n{remaining_missing[remaining_missing > 0]}"
            )

            # Forward fill for numeric columns, drop rest
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[
                numeric_cols
            ].ffill()  # pandas 2.1+ compatible
            df_clean = df_clean.dropna()  # Drop any remaining

        self.stats["missing_values_removed"] = initial_count - len(df_clean)
        self.logger.info(
            f"Total removed for missing values: {self.stats['missing_values_removed']:,}"
        )

        return df_clean

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate flight records

        Definition: Same flight date, carrier, flight number, origin, destination

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)

        # Define unique flight identifier
        identifier_cols = [
            "FL_DATE",
            "OP_CARRIER",
            "OP_CARRIER_FL_NUM",
            "ORIGIN",
            "DEST",
        ]

        # Keep first occurrence, drop duplicates
        df_clean = df.drop_duplicates(subset=identifier_cols, keep="first")

        self.stats["duplicates_removed"] = initial_count - len(df_clean)

        if self.stats["duplicates_removed"] > 0:
            self.logger.info(
                f"Removed {self.stats['duplicates_removed']:,} duplicate records"
            )
        else:
            self.logger.info("No duplicates found")

        return df_clean

    def detect_outliers_isolation_forest(
        self, df: pd.DataFrame, columns: List[str]
    ) -> np.ndarray:
        """
        Detect outliers using Isolation Forest (primary method)

        Isolation Forest advantages:
        - Works well with high-dimensional data
        - No assumptions about data distribution
        - Efficient for large datasets
        - Handles non-linear relationships

        Args:
            df: Input DataFrame
            columns: Columns to use for outlier detection

        Returns:
            Boolean mask (True = outlier, False = inlier)
        """
        self.logger.info("Running Isolation Forest outlier detection...")

        # Prepare data (only numeric columns that exist)
        existing_cols = [col for col in columns if col in df.columns]
        X = df[existing_cols].values

        # Initialize Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples="auto",
            n_jobs=-1,
        )

        # Fit and predict (-1 = outlier, 1 = inlier)
        predictions = iso_forest.fit_predict(X)

        # Convert to boolean mask
        outlier_mask = predictions == -1

        n_outliers = outlier_mask.sum()
        self.logger.info(
            f"Isolation Forest detected {n_outliers:,} outliers ({n_outliers / len(df) * 100:.2f}%)"
        )

        return outlier_mask

    def detect_outliers_lof(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """
        Detect outliers using Local Outlier Factor (optional method)

        LOF advantages:
        - Detects local density deviations
        - Good for clustered data
        - Sensitive to neighborhood structure

        Args:
            df: Input DataFrame
            columns: Columns to use for outlier detection

        Returns:
            Boolean mask (True = outlier, False = inlier)
        """
        self.logger.info("Running LOF outlier detection...")

        existing_cols = [col for col in columns if col in df.columns]
        X = df[existing_cols].values

        # Initialize LOF
        lof = LocalOutlierFactor(
            contamination=self.contamination, n_neighbors=20, n_jobs=-1
        )

        # Fit and predict (-1 = outlier, 1 = inlier)
        predictions = lof.fit_predict(X)

        outlier_mask = predictions == -1

        n_outliers = outlier_mask.sum()
        self.logger.info(
            f"LOF detected {n_outliers:,} outliers ({n_outliers / len(df) * 100:.2f}%)"
        )

        return outlier_mask

    def detect_outliers_3sigma(
        self, df: pd.DataFrame, columns: List[str]
    ) -> np.ndarray:
        """
        Detect outliers using 3-sigma rule (fallback method)

        3-sigma rule:
        - Simple statistical method
        - Assumes normal distribution
        - Points beyond μ ± 3σ are outliers

        Args:
            df: Input DataFrame
            columns: Columns to use for outlier detection

        Returns:
            Boolean mask (True = outlier, False = inlier)
        """
        self.logger.info("Running 3-sigma outlier detection...")

        outlier_mask = np.zeros(len(df), dtype=bool)

        for col in columns:
            if col not in df.columns:
                continue

            # Calculate mean and std
            mean = df[col].mean()
            std = df[col].std()

            # Points beyond μ ± 3σ
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask |= col_outliers

            if col_outliers.sum() > 0:
                self.logger.debug(
                    f"{col}: {col_outliers.sum():,} outliers "
                    f"(range: [{lower_bound:.1f}, {upper_bound:.1f}])"
                )

        n_outliers = outlier_mask.sum()
        self.logger.info(
            f"3-sigma detected {n_outliers:,} outliers ({n_outliers / len(df) * 100:.2f}%)"
        )

        return outlier_mask

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using configured detection method

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers removed
        """
        initial_count = len(df)

        # Select detection method
        if self.outlier_method == "isolation_forest":
            outlier_mask = self.detect_outliers_isolation_forest(
                df, OUTLIER_DETECTION_COLUMNS
            )
        elif self.outlier_method == "lof":
            outlier_mask = self.detect_outliers_lof(df, OUTLIER_DETECTION_COLUMNS)
        elif self.outlier_method == "3sigma":
            outlier_mask = self.detect_outliers_3sigma(df, OUTLIER_DETECTION_COLUMNS)
        else:
            self.logger.warning(
                f"Unknown method '{self.outlier_method}', defaulting to 3-sigma"
            )
            outlier_mask = self.detect_outliers_3sigma(df, OUTLIER_DETECTION_COLUMNS)

        # Remove outliers
        df_clean = df[~outlier_mask].copy()

        self.stats["outliers_removed"] = initial_count - len(df_clean)
        self.logger.info(
            f"Removed {self.stats['outliers_removed']:,} outliers "
            f"({self.stats['outliers_removed'] / initial_count * 100:.2f}%)"
        )

        return df_clean

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate cleaned data against quality rules

        Checks:
        - All delay values within acceptable ranges
        - All distances within acceptable ranges
        - No remaining null values
        - No duplicate records

        Args:
            df: Cleaned DataFrame

        Returns:
            Dictionary of validation results
        """
        validation_results = {}

        # Check for remaining nulls
        null_counts = df.isna().sum()
        validation_results["no_nulls"] = null_counts.sum() == 0

        if not validation_results["no_nulls"]:
            self.logger.warning(
                f"Remaining null values:\n{null_counts[null_counts > 0]}"
            )

        # Check value ranges
        for col, rules in VALIDATION_RULES.items():
            if col not in df.columns:
                continue

            min_val, max_val = rules["min"], rules["max"]
            in_range = df[col].between(min_val, max_val).all()
            validation_results[f"{col}_valid_range"] = in_range

            if not in_range:
                out_of_range = df[~df[col].between(min_val, max_val)][col]
                self.logger.warning(
                    f"{col} has {len(out_of_range)} values outside "
                    f"[{min_val}, {max_val}]: {out_of_range.describe()}"
                )

        # Check for duplicates
        identifier_cols = [
            "FL_DATE",
            "OP_CARRIER",
            "OP_CARRIER_FL_NUM",
            "ORIGIN",
            "DEST",
        ]
        has_duplicates = df.duplicated(subset=identifier_cols).any()
        validation_results["no_duplicates"] = not has_duplicates

        if has_duplicates:
            self.logger.warning(
                f"Found {df.duplicated(subset=identifier_cols).sum()} duplicates"
            )

        # Overall quality score (defensive check for empty results)
        if len(validation_results) > 0:
            quality_score = sum(validation_results.values()) / len(validation_results)
        else:
            quality_score = 0.0
        self.stats["quality_score"] = quality_score

        self.logger.info(f"Data quality score: {quality_score * 100:.1f}%")

        return validation_results

    def full_pipeline(
        self,
        df: pd.DataFrame,
        save_parquet: bool = False,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Execute complete data cleaning pipeline

        Pipeline steps:
        1. Remove cancelled/diverted flights
        2. Handle missing values
        3. Remove duplicates
        4. Remove outliers
        5. Validate data quality
        6. (Optional) Save to Parquet

        Args:
            df: Input DataFrame with BTS data
            save_parquet: If True, save cleaned data to Parquet
            output_path: Custom output path (default: PROCESSED_DATA_DIR)

        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        self.stats["original_records"] = len(df)
        self.logger.info("=" * 60)
        self.logger.info("STARTING DATA CLEANING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Input records: {len(df):,}")
        self.logger.info(f"Input columns: {len(df.columns)}")

        # Step 1: Remove cancelled/diverted
        self.logger.info("\n[1/5] Removing cancelled and diverted flights...")
        df = self.remove_cancelled_diverted(df)

        # Step 2: Handle missing values
        self.logger.info("\n[2/5] Handling missing values...")
        df = self.handle_missing_values(df)

        # Step 3: Remove duplicates
        self.logger.info("\n[3/5] Removing duplicates...")
        df = self.remove_duplicates(df)

        # Step 4: Remove outliers
        self.logger.info("\n[4/5] Detecting and removing outliers...")
        df = self.remove_outliers(df)

        # Step 5: Validate quality
        self.logger.info("\n[5/5] Validating data quality...")
        validation_results = self.validate_data_quality(df)

        self.stats["final_records"] = len(df)

        # Summary statistics
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CLEANING PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Original records:      {self.stats['original_records']:,}")
        self.logger.info(f"Cancelled removed:     {self.stats['cancelled_removed']:,}")
        self.logger.info(f"Diverted removed:      {self.stats['diverted_removed']:,}")
        self.logger.info(
            f"Missing data removed:  {self.stats['missing_values_removed']:,}"
        )
        self.logger.info(f"Duplicates removed:    {self.stats['duplicates_removed']:,}")
        self.logger.info(f"Outliers removed:      {self.stats['outliers_removed']:,}")
        self.logger.info(f"Final records:         {self.stats['final_records']:,}")

        # Calculate retention rate (defensive check for division by zero)
        if self.stats["original_records"] > 0:
            retention_rate = (
                self.stats["final_records"] / self.stats["original_records"]
            ) * 100
        else:
            retention_rate = 0.0
        self.logger.info(f"Data retention rate:   {retention_rate:.1f}%")
        self.logger.info(
            f"Quality score:         {self.stats['quality_score'] * 100:.1f}%"
        )

        # Save to Parquet if requested
        if save_parquet:
            if output_path is None:
                output_path = (
                    PROCESSED_DATA_DIR
                    / f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                )
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, compression=PARQUET_COMPRESSION, index=False)
            self.logger.info(f"\n✓ Saved cleaned data to {output_path}")

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cleaning statistics from last pipeline run

        Returns:
            Dictionary with cleaning metrics
        """
        return self.stats.copy()


# Convenience function
def clean_bts_data(
    filepath: Path, outlier_method: str = "isolation_forest", save_parquet: bool = True
) -> pd.DataFrame:
    """
    Convenience function to clean BTS data from file

    Args:
        filepath: Path to raw BTS CSV file
        outlier_method: 'isolation_forest', 'lof', or '3sigma'
        save_parquet: If True, save cleaned data to Parquet

    Returns:
        Cleaned DataFrame
    """
    # Load data
    print(f"Loading data from {filepath.name}...")
    df = pd.read_csv(filepath, low_memory=False)

    # Clean data
    cleanser = DataCleanser(outlier_method=outlier_method)
    df_clean = cleanser.full_pipeline(df, save_parquet=save_parquet)

    return df_clean


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("=" * 60)
    print("DATA CLEANSER MODULE TEST")
    print("=" * 60)

    # Test with sample data
    sample_data = {
        "FL_DATE": ["2024-01-01"] * 10,
        "OP_CARRIER": ["AA"] * 10,
        "OP_CARRIER_FL_NUM": list(range(100, 110)),
        "ORIGIN": ["ATL"] * 10,
        "DEST": ["DFW"] * 10,
        "DEP_DELAY": [
            5,
            10,
            -5,
            200,
            15,
            np.nan,
            8,
            12,
            3,
            1000,
        ],  # One null, one extreme outlier
        "ARR_DELAY": [3, 8, -3, 195, 12, 9, 7, 10, 2, 995],  # One extreme outlier
        "CANCELLED": [0] * 9 + [1],  # One cancelled
        "DIVERTED": [0] * 10,
        "DISTANCE": [750] * 10,
        "CARRIER_DELAY": [0] * 10,
        "WEATHER_DELAY": [0] * 10,
        "NAS_DELAY": [0] * 10,
        "SECURITY_DELAY": [0] * 10,
        "LATE_AIRCRAFT_DELAY": [0] * 10,
        # Times already in minutes since midnight (480 = 8:00 AM)
        "CRS_DEP_TIME": [480] * 10,
        "DEP_TIME": [485] * 10,
        "CRS_ARR_TIME": [600] * 10,
        "ARR_TIME": [605] * 10,
        "CRS_ELAPSED_TIME": [120] * 10,
        "ACTUAL_ELAPSED_TIME": [120] * 10,
    }

    df_sample = pd.DataFrame(sample_data)
    print(f"\nSample data created: {len(df_sample)} records")

    # Test cleaning pipeline
    cleanser = DataCleanser(outlier_method="isolation_forest", log_level="INFO")
    df_clean = cleanser.full_pipeline(df_sample, save_parquet=False)

    print(f"\nCleaned data: {len(df_clean)} records")
    print("\nStatistics:")
    for key, value in cleanser.get_statistics().items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
