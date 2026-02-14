"""
Feature Engineering Module
Creates rich features from cleaned flight data for delay prediction

Feature Categories:
1. Temporal Features - Time-based patterns (cyclical encoding, peak hours, weekend)
2. Flight & Airline Features - Historical performance (carrier delay rates, route stats)
3. Airport Features - Infrastructure and congestion (size, traffic)
4. Network Effects - Cascading delays (previous flight, turnaround stress)
5. Encoding - Categorical to numeric transformation

Output: Feature matrix ready for ML model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime

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
    Comprehensive feature engineering for flight delay prediction

    Transforms cleaned flight data into rich feature set including:
    - Temporal patterns (cyclical time encoding)
    - Historical performance metrics
    - Airport congestion indicators
    - Network delay propagation effects

    Output Schema: ~40-50 features ready for ML
    """

    def __init__(self, log_level: str = "INFO", use_external_data: bool = True):
        """
        Initialize feature engineer

        Args:
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            use_external_data: Whether to enrich with external APIs (Weather/Traffic)
        """
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

        # Store label encoders for categorical columns
        self.label_encoders = {}

        self.raw_features = []
        self.engineered_features = []

        # Initialize External Manager (if requested)
        self.use_external_data = use_external_data
        self.external_manager = None
        if self.use_external_data:
            try:
                self.external_manager = ExternalManager()
                self.logger.info("External Data Manager integration enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize External Manager: {e}")

        self.logger.info("FeatureEngineer initialized")

    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features with cyclical encoding

        Features created:
        - hour, day_of_week, month (raw)
        - hour_sin, hour_cos (cyclical encoding)
        - day_of_week_sin, day_of_week_cos
        - month_sin, month_cos
        - is_weekend (binary)
        - is_peak_hour (binary)

        Args:
            df: DataFrame with FL_DATE and CRS_DEP_TIME columns

        Returns:
            DataFrame with temporal features added
        """
        self.logger.info("Creating temporal features...")
        df = df.copy()

        # Parse date if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["FL_DATE"]):
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

        # Extract basic temporal components
        df["month"] = df["FL_DATE"].dt.month
        df["day_of_week"] = df["FL_DATE"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["day_of_month"] = df["FL_DATE"].dt.day

        # Extract hour from CRS_DEP_TIME (minutes since midnight -> hour)
        df["hour"] = (df["CRS_DEP_TIME"] // 60).astype(int)

        # Cyclical encoding for hour (0-23)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day of week (0-6)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Cyclical encoding for month (1-12)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        # Weekend indicator (Saturday=5, Sunday=6)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Peak hour indicator
        df["is_peak_hour"] = 0
        for start_min, end_min in PEAK_HOURS:
            in_peak = (df["CRS_DEP_TIME"] >= start_min) & (
                df["CRS_DEP_TIME"] <= end_min
            )
            df["is_peak_hour"] = df["is_peak_hour"] | in_peak.astype(int)

        # Season indicator (Winter=0, Spring=1, Summer=2, Fall=3)
        df["season"] = ((df["month"] % 12) // 3).astype(int)

        temporal_features = [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "season",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "is_peak_hour",
        ]

        self.logger.info(f"Created {len(temporal_features)} temporal features")
        self.engineered_features.extend(temporal_features)

        return df

    # ========================================================================
    # FLIGHT & AIRLINE FEATURES
    # ========================================================================

    def create_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create carrier-based historical performance features

        Features created:
        - carrier_delay_rate: Historical % of delayed flights
        - carrier_avg_delay: Average delay for this carrier
        - carrier_delay_std: Delay variability
        - carrier_cancellation_rate: Historical cancellation rate

        Args:
            df: DataFrame with OP_CARRIER and delay columns

        Returns:
            DataFrame with carrier features added
        """
        self.logger.info("Creating carrier features...")
        df = df.copy()

        # Calculate carrier-level statistics
        carrier_stats = (
            df.groupby("OP_CARRIER")
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

        # Fill NaN std with 0 (carriers with single flight)
        carrier_stats["carrier_arr_delay_std"] = carrier_stats[
            "carrier_arr_delay_std"
        ].fillna(0)

        # Merge back to main dataframe
        df = df.merge(carrier_stats, on="OP_CARRIER", how="left")

        carrier_features = [
            "carrier_avg_arr_delay",
            "carrier_arr_delay_std",
            "carrier_delay_rate",
            "carrier_avg_dep_delay",
        ]

        self.logger.info(f"Created {len(carrier_features)} carrier features")
        self.engineered_features.extend(carrier_features)

        return df

    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create route-based delay statistics

        Features created:
        - route_delay_rate: Historical % of delays on this route
        - route_avg_delay: Average delay on this route
        - route_distance_normalized: Distance / median route distance

        Args:
            df: DataFrame with ORIGIN, DEST, DISTANCE, and delay columns

        Returns:
            DataFrame with route features added
        """
        self.logger.info("Creating route features...")
        df = df.copy()

        # Create route identifier
        df["route"] = df["ORIGIN"] + "_" + df["DEST"]

        # Calculate route-level statistics
        route_stats = (
            df.groupby("route")
            .agg(
                {
                    "ARR_DELAY": ["mean", lambda x: (x > 15).mean()],
                    "DISTANCE": "first",  # Distance is constant per route
                }
            )
            .reset_index()
        )

        route_stats.columns = [
            "route",
            "route_avg_delay",
            "route_delay_rate",
            "route_distance",
        ]

        # Merge back
        df = df.merge(route_stats, on="route", how="left")

        # Normalize distance by median (defensive check for zero)
        median_distance = df["DISTANCE"].median()
        if median_distance > 0:
            df["route_distance_normalized"] = df["DISTANCE"] / median_distance
        else:
            self.logger.warning(
                "Median distance is 0 or NaN, using fallback normalization"
            )
            df["route_distance_normalized"] = 1.0

        # Drop temporary route column
        df = df.drop(columns=["route"])

        route_features = [
            "route_avg_delay",
            "route_delay_rate",
            "route_distance_normalized",
        ]

        self.logger.info(f"Created {len(route_features)} route features")
        self.engineered_features.extend(route_features)

        return df

    # ========================================================================
    # AIRPORT FEATURES
    # ========================================================================

    def create_airport_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create airport congestion and size features

        Features created:
        - origin_flight_count: Daily departures (congestion proxy)
        - dest_flight_count: Daily arrivals (congestion proxy)
        - origin_size_category: Airport size (0=small, 1=medium, 2=large)
        - dest_size_category: Airport size

        Args:
            df: DataFrame with ORIGIN, DEST, and FL_DATE columns

        Returns:
            DataFrame with airport features added
        """
        self.logger.info("Creating airport features...")
        df = df.copy()

        # Count daily flights per airport (congestion proxy)
        df["date"] = df["FL_DATE"].dt.date

        # Origin airport congestion
        origin_daily = (
            df.groupby(["ORIGIN", "date"])
            .size()
            .reset_index(name="origin_daily_flights")
        )
        origin_stats = (
            origin_daily.groupby("ORIGIN")["origin_daily_flights"].mean().reset_index()
        )
        origin_stats.columns = ["ORIGIN", "origin_flight_count"]

        # Destination airport congestion
        dest_daily = (
            df.groupby(["DEST", "date"]).size().reset_index(name="dest_daily_flights")
        )
        dest_stats = (
            dest_daily.groupby("DEST")["dest_daily_flights"].mean().reset_index()
        )
        dest_stats.columns = ["DEST", "dest_flight_count"]

        # Merge back
        df = df.merge(origin_stats, on="ORIGIN", how="left")
        df = df.merge(dest_stats, on="DEST", how="left")

        # Categorize airport size based on traffic (handle NaN)
        # Small: <50 flights/day, Medium: 50-200, Large: >200
        df["origin_size_category"] = pd.cut(
            df["origin_flight_count"], bins=[0, 50, 200, float("inf")], labels=[0, 1, 2]
        )
        df["origin_size_category"] = df["origin_size_category"].fillna(0).astype(int)

        df["dest_size_category"] = pd.cut(
            df["dest_flight_count"], bins=[0, 50, 200, float("inf")], labels=[0, 1, 2]
        )
        df["dest_size_category"] = df["dest_size_category"].fillna(0).astype(int)

        # Drop temporary date column
        df = df.drop(columns=["date"])

        airport_features = [
            "origin_flight_count",
            "dest_flight_count",
            "origin_size_category",
            "dest_size_category",
        ]

        self.logger.info(f"Created {len(airport_features)} airport features")
        self.engineered_features.extend(airport_features)

        return df

    # ========================================================================
    # NETWORK EFFECT FEATURES
    # ========================================================================

    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create network delay propagation features

        Features created:
        - prev_flight_delay: Delay of previous flight by same aircraft
        - turnaround_time: Time between arrival and next departure
        - turnaround_stress: Binary flag for tight turnaround (<30 min)
        - same_day_carrier_delays: Number of delayed flights by carrier today

        Args:
            df: DataFrame with carrier, times, and delay columns

        Returns:
            DataFrame with network features added
        """
        self.logger.info("Creating network features...")
        df = df.copy()

        # Sort by date and time for sequence analysis
        df = df.sort_values(["FL_DATE", "OP_CARRIER", "CRS_DEP_TIME"])

        # Previous flight delay (simplified - by carrier and date)
        # In production, would use tail number for actual aircraft tracking
        df["prev_flight_delay"] = df.groupby(["OP_CARRIER", "FL_DATE"])[
            "ARR_DELAY"
        ].shift(1)
        df["prev_flight_delay"] = df["prev_flight_delay"].fillna(0)

        # Turnaround time estimation (difference between flights)
        df["prev_arr_time"] = df.groupby(["OP_CARRIER", "FL_DATE"])["ARR_TIME"].shift(1)
        df["turnaround_time"] = df["CRS_DEP_TIME"] - df["prev_arr_time"]

        # Handle negative turnarounds (different aircraft) and missing
        df["turnaround_time"] = df["turnaround_time"].clip(lower=0)

        # Fill missing with median (defensive check for all-NaN)
        median_turnaround = df["turnaround_time"].median()
        if pd.isna(median_turnaround):
            self.logger.warning(
                "All turnaround times are NaN, using default 60 minutes"
            )
            median_turnaround = 60  # Default: 60 minutes
        df["turnaround_time"] = df["turnaround_time"].fillna(median_turnaround)

        # Turnaround stress indicator (tight connection)
        df["turnaround_stress"] = (df["turnaround_time"] < 30).astype(int)

        # Same-day carrier delay count (use PREVIOUS flights only - no data leakage)
        # CRITICAL: Don't use current flight's DEP_DELAY as that's what we're trying to predict!
        df["prev_delayed"] = (
            (df.groupby(["OP_CARRIER", "FL_DATE"])["ARR_DELAY"].shift(1) > 15)
            .astype(int)
            .fillna(0)
        )
        df["same_day_carrier_delays"] = df.groupby(["OP_CARRIER", "FL_DATE"])[
            "prev_delayed"
        ].cumsum()

        # Drop temporary columns
        df = df.drop(columns=["prev_arr_time", "prev_delayed"])

        network_features = [
            "prev_flight_delay",
            "turnaround_time",
            "turnaround_stress",
            "same_day_carrier_delays",
        ]

        self.logger.info(f"Created {len(network_features)} network features")
        self.engineered_features.extend(network_features)

        return df

    # ========================================================================
    # ENCODING & TRANSFORMATION
    # ========================================================================

    def encode_categorical_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using label encoding

        Encodes: OP_CARRIER, ORIGIN, DEST

        Args:
            df: DataFrame with categorical columns
            fit: If True, fit new encoders. If False, use existing encoders

        Returns:
            DataFrame with encoded categorical features
        """
        self.logger.info("Encoding categorical features...")
        df = df.copy()

        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found, skipping encoding")
                continue

            if fit:
                # Fit new encoder
                encoder = LabelEncoder()
                df[col + "_encoded"] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
                self.logger.debug(
                    f"Fitted encoder for {col}: {len(encoder.classes_)} classes"
                )
            else:
                # Use existing encoder
                if col not in self.label_encoders:
                    raise ValueError(
                        f"No fitted encoder found for {col}. Run with fit=True first."
                    )

                encoder = self.label_encoders[col]
                # Handle unseen categories
                df[col + "_encoded"] = (
                    df[col]
                    .astype(str)
                    .apply(
                        lambda x: (
                            encoder.transform([x])[0] if x in encoder.classes_ else -1
                        )
                    )
                )

        encoded_features = [
            col + "_encoded" for col in CATEGORICAL_COLUMNS if col in df.columns
        ]
        self.logger.info(f"Encoded {len(encoded_features)} categorical features")
        self.engineered_features.extend(encoded_features)

        return df

    # ========================================================================
    # FULL PIPELINE
    # ========================================================================

    def create_all_features(
        self, df: pd.DataFrame, fit_encoders: bool = True
    ) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline

        Pipeline steps:
        1. Temporal features
        2. Carrier features
        3. Route features
        4. Airport features
        5. Network features
        6. Categorical encoding

        Args:
            df: Cleaned DataFrame from data_cleanser
            fit_encoders: If True, fit new encoders (training). If False, use existing (inference)

        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Input shape: {df.shape}")

        # Reset feature tracking
        self.raw_features = list(df.columns)
        self.engineered_features = []

        # Step 1: Temporal features
        self.logger.info("\n[1/6] Creating temporal features...")
        df = self.create_temporal_features(df)

        # Step 2: Carrier features
        self.logger.info("\n[2/6] Creating carrier features...")
        df = self.create_carrier_features(df)

        # Step 3: Route features
        self.logger.info("\n[3/6] Creating route features...")
        df = self.create_route_features(df)

        # Step 4: Airport features
        self.logger.info("\n[4/6] Creating airport features...")
        df = self.create_airport_features(df)

        # Step 5: Network features
        self.logger.info("\n[5/6] Creating network features...")
        df = self.create_network_features(df)

        # Step 6: Categorical encoding
        self.logger.info("\n[6/6] Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=fit_encoders)

        # Step 7: External Enrichment
        if self.use_external_data and self.external_manager:
            self.logger.info(
                "\n[7/7] Enriching with external data (Weather/Traffic)..."
            )
            try:
                # Get current features to track what's added
                features_before = set(df.columns)
                df = self.external_manager.enrich_dataframe(df)
                new_features = list(set(df.columns) - features_before)
                if new_features:
                    self.logger.info(
                        f"Added {len(new_features)} external features: {new_features}"
                    )
                    self.engineered_features.extend(new_features)
            except Exception as e:
                self.logger.error(f"External enrichment failed (skipping): {e}")

        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FEATURE ENGINEERING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Output shape: {df.shape}")
        self.logger.info(f"Raw features: {len(self.raw_features)}")
        self.logger.info(f"Engineered features: {len(self.engineered_features)}")
        self.logger.info(f"Total features: {len(df.columns)}")

        return df

    def get_feature_names(self, include_raw: bool = False) -> List[str]:
        """
        Get list of feature names

        Args:
            include_raw: If True, include original raw features

        Returns:
            List of feature names
        """
        if include_raw:
            return self.raw_features + self.engineered_features
        else:
            return self.engineered_features

    def select_features_for_training(
        self, df: pd.DataFrame, target_col: str = "ARR_DELAY"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select final feature set for model training

        Excludes:
        - Target variable
        - Identifiers (FL_DATE, OP_CARRIER, ORIGIN, DEST)
        - Intermediate/redundant features

        Args:
            df: DataFrame with all features
            target_col: Name of target column

        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        self.logger.info("Selecting features for training...")

        # Define columns to exclude
        exclude_cols = [
            # Target
            target_col,
            # Also exclude DEP_DELAY if predicting ARR_DELAY (data leakage)
            "DEP_DELAY" if target_col == "ARR_DELAY" else None,
            # Identifiers (keep encoded versions)
            "FL_DATE",
            "OP_CARRIER",
            "ORIGIN",
            "DEST",
            # Intermediate features used for calculation (none currently)
            # Actual times (use scheduled times only to avoid leakage)
            "DEP_TIME",
            "ARR_TIME",
            "ACTUAL_ELAPSED_TIME",
            # Delay attribution (only available post-flight)
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",
            "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY",
        ]

        # Remove None values
        exclude_cols = [col for col in exclude_cols if col is not None]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        self.logger.info(f"Selected {len(feature_cols)} features for training")
        self.logger.info(f"Target variable: {target_col}")
        self.logger.info(f"Sample count: {len(X):,}")

        return X, y


# Convenience function
def engineer_features(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    save_parquet: bool = False,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convenience function to engineer features from cleaned data

    Args:
        df: Cleaned DataFrame from data_cleanser
        fit_encoders: If True, fit new encoders (training mode)
        save_parquet: If True, save engineered data to Parquet
        output_path: Custom output path (default: PROCESSED_DATA_DIR)

    Returns:
        DataFrame with all engineered features
    """
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df, fit_encoders=fit_encoders)

    if save_parquet:
        if output_path is None:
            output_path = (
                PROCESSED_DATA_DIR
                / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(output_path, compression="snappy", index=False)
        print(f"✓ Saved features to {output_path}")

    return df_features


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("=" * 60)
    print("FEATURE ENGINEER MODULE TEST")
    print("=" * 60)

    # Create sample cleaned data
    np.random.seed(42)
    n_samples = 100

    sample_data = {
        "FL_DATE": pd.date_range("2024-01-01", periods=n_samples, freq="1h"),
        "OP_CARRIER": np.random.choice(["AA", "DL", "UA"], n_samples),
        "OP_CARRIER_FL_NUM": np.random.randint(100, 999, n_samples),
        "ORIGIN": np.random.choice(["ATL", "DFW", "ORD", "LAX"], n_samples),
        "DEST": np.random.choice(["ATL", "DFW", "ORD", "LAX"], n_samples),
        "CRS_DEP_TIME": np.random.randint(0, 1440, n_samples),  # Minutes since midnight
        "CRS_ARR_TIME": np.random.randint(0, 1440, n_samples),
        "DEP_TIME": np.random.randint(0, 1440, n_samples),
        "ARR_TIME": np.random.randint(0, 1440, n_samples),
        "CRS_ELAPSED_TIME": np.random.randint(60, 360, n_samples),
        "ACTUAL_ELAPSED_TIME": np.random.randint(60, 360, n_samples),
        "DEP_DELAY": np.random.normal(10, 20, n_samples),
        "ARR_DELAY": np.random.normal(5, 25, n_samples),
        "DISTANCE": np.random.randint(200, 2500, n_samples),
        "CARRIER_DELAY": np.zeros(n_samples),
        "WEATHER_DELAY": np.zeros(n_samples),
        "NAS_DELAY": np.zeros(n_samples),
        "SECURITY_DELAY": np.zeros(n_samples),
        "LATE_AIRCRAFT_DELAY": np.zeros(n_samples),
    }

    df_sample = pd.DataFrame(sample_data)
    print(f"\nSample data created: {df_sample.shape}")

    # Test feature engineering
    engineer = FeatureEngineer(log_level="INFO")
    df_features = engineer.create_all_features(df_sample)

    print(f"\nEngineered data shape: {df_features.shape}")
    print(f"Feature names ({len(engineer.get_feature_names())} total):")
    for i, feat in enumerate(engineer.get_feature_names()[:10], 1):
        print(f"  {i}. {feat}")
    print(f"  ... and {len(engineer.get_feature_names()) - 10} more")

    # Test feature selection
    X, y = engineer.select_features_for_training(df_features)
    print(f"\nTraining features: {X.shape}")
    print(f"Target values: {y.shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
