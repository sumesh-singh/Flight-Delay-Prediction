"""
Memory-Optimized Feature Engineering Module

Key optimizations:
1. Removed unnecessary .copy() operations
2. Convert string columns to categorical dtype
3. Use int16/int8 for small integer columns
4. Garbage collection between steps
5. Process in chunks where possible
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
import gc

# Import configuration
try:
    from config.data_config import (
        PROCESSED_DATA_DIR,
        PEAK_HOURS,
        CATEGORICAL_COLUMNS,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.data_config import (
        PROCESSED_DATA_DIR,
        PEAK_HOURS,
        CATEGORICAL_COLUMNS,
    )

try:
    from src.data.external.external_manager import ExternalManager
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.data.external.external_manager import ExternalManager


class FeatureEngineer:
    """
    MEMORY-OPTIMIZED Feature Engineer

    Changes from original:
    - Removed .copy() calls (modify in-place)
    - Convert to categorical dtype
    - Aggressive dtype downcasting
    - Garbage collection between steps
    """

    def __init__(
        self,
        use_external_data: bool = False,
        external_data_path: Optional[Path] = None,
    ):
        """
        Initialize feature engineer with memory optimizations.

        Args:
            use_external_data: Enable external data enrichment
            external_data_path: Path to external data sources
        """
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.raw_features = []
        self.engineered_features = []

        # External data integration
        self.use_external_data = use_external_data
        self.external_manager = None

        if use_external_data:
            self.external_manager = ExternalManager(
                data_dir=external_data_path or Path("data/external")
            )
            self.logger.info("External Data Manager integration enabled")

        self.logger.info("FeatureEngineer initialized")

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MEMORY OPTIMIZATION: Downcast dtypes to save memory.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized dtypes
        """
        self.logger.info("Optimizing dtypes...")

        # Convert categorical columns
        categorical_cols = ["OP_CARRIER", "ORIGIN", "DEST", "Reporting_Airline"]
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype("category")

        # Downcast integers
        for col in df.select_dtypes(include=["int64"]).columns:
            if col.endswith("ID"):  # IDs can stay int32
                df[col] = df[col].astype("int32")
            elif df[col].max() < 127 and df[col].min() > -128:
                df[col] = df[col].astype("int8")
            elif df[col].max() < 32767 and df[col].min() > -32768:
                df[col] = df[col].astype("int16")
            else:
                df[col] = df[col].astype("int32")

        # Downcast floats
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")

        self.logger.info("Dtypes optimized")
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features with cyclical encoding.
        NO COPY - modifies in place.
        """
        self.logger.info("Creating temporal features...")

        # Parse date if needed
        if "FL_DATE" not in df.columns and "FlightDate" in df.columns:
            df.rename(columns={"FlightDate": "FL_DATE"}, inplace=True)

        if df["FL_DATE"].dtype == "object":
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

        # Extract components
        df["hour"] = pd.to_datetime(
            df["CRS_DEP_TIME"], format="%H%M", errors="coerce"
        ).dt.hour
        df["day_of_week"] = df["FL_DATE"].dt.dayofweek
        df["month"] = df["FL_DATE"].dt.month
        df["day"] = df["FL_DATE"].dt.day
        df["quarter"] = df["FL_DATE"].dt.quarter

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Peak hour indicator
        df["is_peak_hour"] = (
            (df["hour"] >= 7) & (df["hour"] <= 9)
            | (df["hour"] >= 17) & (df["hour"] <= 19)
        ).astype("int8")

        # Weekend flag
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

        # Holiday proximity (simplified)
        df["is_near_holiday"] = df["month"].isin([11, 12, 1]).astype("int8")

        temporal_features = [
            "hour",
            "day_of_week",
            "month",
            "day",
            "quarter",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_peak_hour",
            "is_weekend",
            "is_near_holiday",
        ]

        self.logger.info(f"Created {len(temporal_features)} temporal features")
        self.engineered_features.extend(temporal_features)

        return df

    def create_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MEMORY OPTIMIZED: No .copy(), merge directly.
        """
        self.logger.info("Creating carrier features...")

        # Calculate stats (small DataFrame)
        carrier_stats = (
            df.groupby("OP_CARRIER", observed=True)  # observed=True for categorical
            .agg(
                {
                    "ARR_DELAY": ["mean", "std", lambda x: (x > 15).mean()],
                    "DEP_DELAY": "mean",
                }
            )
            .reset_index()
        )

        carrier_stats.columns = [
            "OP_CARRIER",
            "carrier_avg_arr_delay",
            "carrier_arr_delay_std",
            "carrier_delay_rate",
            "carrier_avg_dep_delay",
        ]

        # Fill NaN
        carrier_stats["carrier_arr_delay_std"] = carrier_stats[
            "carrier_arr_delay_std"
        ].fillna(0)

        # Downcast to float32
        for col in carrier_stats.select_dtypes(include=["float64"]).columns:
            carrier_stats[col] = carrier_stats[col].astype("float32")

        # Merge IN PLACE (no copy)
        df = df.merge(carrier_stats, on="OP_CARRIER", how="left")

        carrier_features = [
            "carrier_avg_arr_delay",
            "carrier_arr_delay_std",
            "carrier_delay_rate",
            "carrier_avg_dep_delay",
        ]

        self.logger.info(f"Created {len(carrier_features)} carrier features")
        self.engineered_features.extend(carrier_features)

        # Clean up
        del carrier_stats
        gc.collect()

        return df

    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MEMORY OPTIMIZED: Route-level statistics.
        """
        self.logger.info("Creating route features...")

        # Create route key
        df["route"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
        df["route"] = df["route"].astype("category")

        # Calculate route stats
        route_stats = (
            df.groupby("route", observed=True)
            .agg(
                {
                    "ARR_DELAY": ["mean", lambda x: (x > 15).mean()],
                    "DISTANCE": "mean",
                }
            )
            .reset_index()
        )

        route_stats.columns = [
            "route",
            "route_avg_delay",
            "route_delay_rate",
            "route_avg_distance",
        ]

        # Downcast
        for col in route_stats.select_dtypes(include=["float64"]).columns:
            route_stats[col] = route_stats[col].astype("float32")

        # Merge
        df = df.merge(route_stats, on="route", how="left")

        route_features = ["route_avg_delay", "route_delay_rate", "route_avg_distance"]

        self.logger.info(f"Created {len(route_features)} route features")
        self.engineered_features.extend(route_features)

        # Clean up
        del route_stats
        gc.collect()

        return df

    def create_airport_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MEMORY OPTIMIZED: Airport-level statistics.
        """
        self.logger.info("Creating airport features...")

        # Origin airport stats
        origin_stats = (
            df.groupby("ORIGIN", observed=True)
            .agg(
                {
                    "ARR_DELAY": lambda x: (x > 15).mean(),
                    "OP_CARRIER": "count",  # Airport size
                }
            )
            .reset_index()
        )

        origin_stats.columns = ["ORIGIN", "origin_delay_rate", "origin_flight_count"]
        origin_stats["origin_delay_rate"] = origin_stats["origin_delay_rate"].astype(
            "float32"
        )
        origin_stats["origin_flight_count"] = origin_stats[
            "origin_flight_count"
        ].astype("int32")

        # Dest airport stats
        dest_stats = (
            df.groupby("DEST", observed=True)
            .agg(
                {
                    "ARR_DELAY": lambda x: (x > 15).mean(),
                    "OP_CARRIER": "count",
                }
            )
            .reset_index()
        )

        dest_stats.columns = ["DEST", "dest_delay_rate", "dest_flight_count"]
        dest_stats["dest_delay_rate"] = dest_stats["dest_delay_rate"].astype("float32")
        dest_stats["dest_flight_count"] = dest_stats["dest_flight_count"].astype(
            "int32"
        )

        # Merge both
        df = df.merge(origin_stats, on="ORIGIN", how="left")
        df = df.merge(dest_stats, on="DEST", how="left")

        airport_features = [
            "origin_delay_rate",
            "origin_flight_count",
            "dest_delay_rate",
            "dest_flight_count",
        ]

        self.logger.info(f"Created {len(airport_features)} airport features")
        self.engineered_features.extend(airport_features)

        # Clean up
        del origin_stats, dest_stats
        gc.collect()

        return df

    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MEMORY OPTIMIZED: Simplified network features.
        """
        self.logger.info("Creating network features...")

        # Sort by carrier and time
        df = df.sort_values(["OP_CARRIER", "FL_DATE", "CRS_DEP_TIME"])

        # Previous flight delay (same carrier, same day)
        # Use categorical groupby for memory efficiency
        df["prev_flight_delay"] = (
            df.groupby(["OP_CARRIER", "FL_DATE"], observed=True)["ARR_DELAY"]
            .shift(1)
            .fillna(0)
            .astype("float32")
        )

        # Turnaround stress
        df["turnaround_stress"] = (df["prev_flight_delay"] > 15).astype("int8")

        network_features = ["prev_flight_delay", "turnaround_stress"]

        self.logger.info(f"Created {len(network_features)} network features")
        self.engineered_features.extend(network_features)

        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        MEMORY OPTIMIZED: Label encode categorical features.
        Uses already-categorical columns, no copy needed.
        """
        self.logger.info("Encoding categorical features...")

        cat_cols_to_encode = ["OP_CARRIER", "ORIGIN", "DEST"]

        for col in cat_cols_to_encode:
            if col not in df.columns:
                continue

            if fit:
                # Training: fit new encoder
                self.label_encoders[col] = LabelEncoder()

                # Fit on current values + UNKNOWN to handle unseen categories later
                current_vals = df[col].astype(str).unique()
                all_vals = np.concatenate([current_vals, ["UNKNOWN"]])
                self.label_encoders[col].fit(all_vals)

                df[f"{col}_encoded"] = (
                    self.label_encoders[col]
                    .transform(df[col].astype(str))
                    .astype("int16")
                )  # Use int16 for encoded values
            else:
                # Inference: use existing encoder
                if col not in self.label_encoders:
                    raise ValueError(f"No encoder found for {col}")

                # Handle unseen categories
                known_cats = set(self.label_encoders[col].classes_)
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: x if x in known_cats else "UNKNOWN")
                )

                df[f"{col}_encoded"] = (
                    self.label_encoders[col].transform(df[col]).astype("int16")
                )

        encoded_features = [
            f"{col}_encoded" for col in cat_cols_to_encode if col in df.columns
        ]

        self.logger.info(f"Encoded {len(encoded_features)} categorical features")
        self.engineered_features.extend(encoded_features)

        return df

    def create_all_features(
        self, df: pd.DataFrame, fit_encoders: bool = True
    ) -> pd.DataFrame:
        """
        MEMORY-OPTIMIZED Full pipeline.

        Changes:
        - Optimize dtypes first
        - Garbage collect after each step
        - No intermediate copies
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FEATURE ENGINEERING PIPELINE (MEMORY OPTIMIZED)")
        self.logger.info("=" * 60)
        self.logger.info(f"Input shape: {df.shape}")
        self.logger.info(
            f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )

        # Reset tracking
        self.raw_features = list(df.columns)
        self.engineered_features = []

        # OPTIMIZATION: Convert dtypes first
        self.logger.info("\n[0/6] Optimizing dtypes...")
        df = self.optimize_dtypes(df)
        self.logger.info(
            f"Memory after dtype optimization: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        gc.collect()

        # Step 1: Temporal
        self.logger.info("\n[1/6] Creating temporal features...")
        df = self.create_temporal_features(df)
        gc.collect()

        # Step 2: Carrier
        self.logger.info("\n[2/6] Creating carrier features...")
        df = self.create_carrier_features(df)
        gc.collect()

        # Step 3: Route
        self.logger.info("\n[3/6] Creating route features...")
        df = self.create_route_features(df)
        gc.collect()

        # Step 4: Airport
        self.logger.info("\n[4/6] Creating airport features...")
        df = self.create_airport_features(df)
        gc.collect()

        # Step 5: Network
        self.logger.info("\n[5/6] Creating network features...")
        df = self.create_network_features(df)
        gc.collect()

        # Step 6: Encoding
        self.logger.info("\n[6/6] Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=fit_encoders)
        gc.collect()

        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FEATURE ENGINEERING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Output shape: {df.shape}")
        self.logger.info(
            f"Final memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        self.logger.info(f"Engineered features: {len(self.engineered_features)}")

        return df

    def select_features_for_training(
        self, df: pd.DataFrame, target_col: str = "IS_DELAYED"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select final features for training.
        """
        # Exclude non-feature columns
        exclude_cols = [
            "FL_DATE",
            "FlightDate",
            "Tail_Number",
            "OP_CARRIER",
            "ORIGIN",
            "DEST",
            "Reporting_Airline",
            "route",
            "ARR_DELAY",
            "DEP_DELAY",
            "IS_DELAYED",
            "Cancelled",
            "Diverted",
        ]

        feature_cols = [
            col for col in df.columns if col not in exclude_cols and col != target_col
        ]

        # Only numeric features
        numeric_features = (
            df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )

        X = df[numeric_features]
        y = df[target_col] if target_col in df.columns else None

        self.logger.info(f"Selected {len(numeric_features)} features for training")

        return X, y
