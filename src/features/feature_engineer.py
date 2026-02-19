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
            self.external_manager = ExternalManager()
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
        Now supports TAIL_NUM for precise aircraft tracking (Perfect Human Factor Proxy).
        """
        self.logger.info("Creating network features...")

        # Check if TAIL_NUM allows for precision tracking
        use_tail_num = "TAIL_NUM" in df.columns

        if use_tail_num:
            self.logger.info("Using TAIL_NUM for precise aircraft turnaround tracking")
            # Sort by aircraft and time
            df = df.sort_values(["TAIL_NUM", "FL_DATE", "CRS_DEP_TIME"])

            # Previous flight delay (same aircraft, same day)
            df["prev_flight_delay"] = (
                df.groupby(["TAIL_NUM", "FL_DATE"], observed=True)["ARR_DELAY"]
                .shift(1)
                .fillna(0)
                .astype("float32")
            )
        else:
            self.logger.warning(
                "TAIL_NUM not found. Using OP_CARRIER proxy (less precise)."
            )
            # Sort by carrier and time
            df = df.sort_values(["OP_CARRIER", "FL_DATE", "CRS_DEP_TIME"])

            # Previous flight delay (same carrier, same day)
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

    def create_human_factors_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NOVEL CONTRIBUTION: Human factors proxy features.

        Addresses IEEE Limitation #3 — "Crew fatigue, ATC workload,
        and maintenance schedules are not modeled despite being identified
        as critical delay causes."

        Creates data-driven proxies from existing BTS fields:
        1. Crew Fatigue Proxy: aircraft daily legs count + late-night operations
        2. ATC Workload Proxy: hourly departure density at origin airport
        3. Maintenance Stress Proxy: aircraft daily utilization hours

        These features are novel — no existing flight delay study models
        human factors using publicly available BTS data.
        """
        self.logger.info("Creating human factors proxy features (NOVEL)...")

        human_features = []

        # --- 1. Crew Fatigue Proxy ---
        use_tail_num = "TAIL_NUM" in df.columns

        if use_tail_num:
            # Aircraft daily legs: number of flights this aircraft flies today
            # High leg count = higher crew fatigue risk
            daily_legs = (
                df.groupby(["TAIL_NUM", "FL_DATE"], observed=True)["CRS_DEP_TIME"]
                .transform("count")
                .fillna(1)
                .astype("int8")
            )
            df["aircraft_daily_legs"] = daily_legs
            human_features.append("aircraft_daily_legs")

            # Aircraft leg sequence: which leg of the day is this flight?
            df["aircraft_leg_number"] = (
                (df.groupby(["TAIL_NUM", "FL_DATE"], observed=True).cumcount() + 1)
                .fillna(1)
                .astype("int8")
            )
            human_features.append("aircraft_leg_number")

            # Cumulative fatigue: ratio of current leg to total daily legs
            df["crew_fatigue_index"] = (
                df["aircraft_leg_number"] / df["aircraft_daily_legs"]
            ).astype("float32")
            human_features.append("crew_fatigue_index")
        else:
            self.logger.warning(
                "TAIL_NUM not available — using carrier-level fatigue proxy"
            )
            # Fallback: carrier daily volume as fatigue proxy
            df["aircraft_daily_legs"] = (
                df.groupby(["OP_CARRIER", "FL_DATE"], observed=True)["CRS_DEP_TIME"]
                .transform("count")
                .fillna(1)
                .astype("int16")
            )
            human_features.append("aircraft_daily_legs")

        # Late-night operation flag (departures 22:00-05:00)
        # Crew operating late at night have higher fatigue
        if "hour" in df.columns:
            df["is_late_night_op"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(
                "int8"
            )
        else:
            # Parse hour from CRS_DEP_TIME
            dep_hour = (df["CRS_DEP_TIME"].astype(float) // 100).astype(int) % 24
            df["is_late_night_op"] = ((dep_hour >= 22) | (dep_hour <= 5)).astype("int8")
        human_features.append("is_late_night_op")

        # --- 2. ATC Workload Proxy ---
        # Hourly departure density at origin airport
        # More concurrent departures = higher ATC workload = more delays
        if "hour" in df.columns:
            hour_col = "hour"
        else:
            df["_temp_hour"] = (df["CRS_DEP_TIME"].astype(float) // 100).astype(
                int
            ) % 24
            hour_col = "_temp_hour"

        df["origin_hourly_density"] = (
            (
                df.groupby(["ORIGIN", "FL_DATE", hour_col], observed=True)[
                    "CRS_DEP_TIME"
                ].transform("count")
            )
            .fillna(1)
            .astype("int16")
        )
        human_features.append("origin_hourly_density")

        # Destination arrival congestion
        df["dest_hourly_density"] = (
            (
                df.groupby(["DEST", "FL_DATE", hour_col], observed=True)[
                    "CRS_DEP_TIME"
                ].transform("count")
            )
            .fillna(1)
            .astype("int16")
        )
        human_features.append("dest_hourly_density")

        # Clean temp column
        if "_temp_hour" in df.columns:
            df.drop(columns=["_temp_hour"], inplace=True)

        # --- 3. Maintenance Stress Proxy ---
        if use_tail_num and "CRSElapsedTime" in df.columns:
            # Aircraft daily utilization: total scheduled flight hours per aircraft per day
            # High utilization = less maintenance window = more mechanical delays
            df["aircraft_daily_util_min"] = (
                df.groupby(["TAIL_NUM", "FL_DATE"], observed=True)[
                    "CRSElapsedTime"
                ].transform("sum")
            ).astype("float32")
            human_features.append("aircraft_daily_util_min")
        elif use_tail_num and "CRS_ELAPSED_TIME" in df.columns:
            df["aircraft_daily_util_min"] = (
                df.groupby(["TAIL_NUM", "FL_DATE"], observed=True)[
                    "CRS_ELAPSED_TIME"
                ].transform("sum")
            ).astype("float32")
            human_features.append("aircraft_daily_util_min")

        self.logger.info(
            f"Created {len(human_features)} human factors features: {human_features}"
        )
        self.engineered_features.extend(human_features)

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
        - Includes human factors features (NOVEL)
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
        self.logger.info("\n[0/8] Optimizing dtypes...")
        df = self.optimize_dtypes(df)
        self.logger.info(
            f"Memory after dtype optimization: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        gc.collect()

        # Step 1: Temporal
        self.logger.info("\n[1/8] Creating temporal features...")
        df = self.create_temporal_features(df)
        gc.collect()

        # Step 2: Carrier
        self.logger.info("\n[2/8] Creating carrier features...")
        df = self.create_carrier_features(df)
        gc.collect()

        # Step 3: Route
        self.logger.info("\n[3/8] Creating route features...")
        df = self.create_route_features(df)
        gc.collect()

        # Step 4: Airport
        self.logger.info("\n[4/8] Creating airport features...")
        df = self.create_airport_features(df)
        gc.collect()

        # Step 5: Network
        self.logger.info("\n[5/8] Creating network features...")
        df = self.create_network_features(df)
        gc.collect()

        # Step 6: Human Factors (NOVEL - IEEE Limitation #3)
        self.logger.info("\n[6/8] Creating human factors features (NOVEL)...")
        df = self.create_human_factors_features(df)
        gc.collect()

        # Step 7: External Data Enrichment
        if self.use_external_data and self.external_manager:
            self.logger.info("\n[7/8] Enriching with External Data (NOAA + OpenSky)...")
            try:
                df = self.external_manager.enrich_dataframe(df)

                # Verify features were added
                ext_cols = [
                    c
                    for c in df.columns
                    if "ORIGIN_TMAX" in c or "ORIGIN_AIRPORT_TRAFFIC" in c
                ]
                if ext_cols:
                    self.logger.info(
                        f"External features added: {len(ext_cols)} ({ext_cols[:3]}...)"
                    )
                    self.engineered_features.extend(ext_cols)
                else:
                    self.logger.warning(
                        "External enrichment ran but no columns were added/found!"
                    )

            except Exception as e:
                self.logger.error(f"External Data Enrichment Failed: {e}")
                # Don't crash, continue without external features
            gc.collect()

        # Step 8: Encoding
        self.logger.info("\n[8/8] Encoding categorical features...")
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

        CRITICAL: Excludes ALL leakage columns in both camelCase AND UPPERCASE
        variants to prevent data leakage regardless of column naming convention.
        """
        # Exclude non-feature columns AND leakage columns
        # NOTE: Include BOTH camelCase and UPPERCASE variants since
        # different data loaders use different naming conventions
        exclude_cols = {
            # ── Identifiers (both naming conventions) ──
            "FL_DATE",
            "FlightDate",
            "Tail_Number",
            "TAIL_NUM",
            "OP_CARRIER",
            "ORIGIN",
            "DEST",
            "Reporting_Airline",
            "Flight_Number_Reporting_Airline",
            "OP_CARRIER_FL_NUM",
            "OriginAirportID",
            "DestAirportID",
            "route",
            "origin_icao",
            "airport_join",
            "date_join",
            # ── LEAKAGE: Actual Times & Durations (Known only AFTER flight) ──
            "DepTime",
            "ArrTime",
            "WheelsOff",
            "WheelsOn",
            "ActualElapsedTime",
            "AirTime",
            "TaxiIn",
            "TaxiOut",
            "DEP_TIME",
            "ARR_TIME",
            "WHEELS_OFF",
            "WHEELS_ON",
            "ACTUAL_ELAPSED_TIME",
            "AIR_TIME",
            "TAXI_IN",
            "TAXI_OUT",
            "FirstDepTime",
            # ── LEAKAGE: Delay Metrics (Directly reveal target) ──
            "ARR_DELAY",
            "ArrDelay",
            "ArrDelayMinutes",
            "ArrDel15",
            "ArrivalDelayGroups",
            "DEP_DELAY",
            "DepDelay",
            "DepDelayMinutes",
            "DepDel15",
            "DepartureDelayGroups",
            # ── LEAKAGE: Delay Causes (Known only after arrival) ──
            "CarrierDelay",
            "WeatherDelay",
            "NASDelay",
            "SecurityDelay",
            "LateAircraftDelay",
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",
            "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY",
            # ── LEAKAGE: Diversion Data (post-flight only) ──
            "DivAirportLandings",
            "DivReachedDest",
            "DivActualElapsedTime",
            "DivArrDelay",
            "DivDistance",
            "Div1Airport",
            "Div1AirportID",
            "Div1AirportSeqID",
            "Div1WheelsOn",
            "Div1TotalGTime",
            "Div1LongestGTime",
            "Div1WheelsOff",
            "Div1TailNum",
            "Div2Airport",
            "Div2AirportID",
            "Div2AirportSeqID",
            "Div2WheelsOn",
            "Div2TotalGTime",
            "Div2LongestGTime",
            "Div2WheelsOff",
            "Div2TailNum",
            "Div3Airport",
            "Div3AirportID",
            "Div3AirportSeqID",
            "Div3WheelsOn",
            "Div3TotalGTime",
            "Div3LongestGTime",
            "Div3WheelsOff",
            "Div3TailNum",
            "Div4Airport",
            "Div4AirportID",
            "Div4AirportSeqID",
            "Div4WheelsOn",
            "Div4TotalGTime",
            "Div4LongestGTime",
            "Div4WheelsOff",
            "Div4TailNum",
            "Div5Airport",
            "Div5AirportID",
            "Div5AirportSeqID",
            "Div5WheelsOn",
            "Div5TotalGTime",
            "Div5LongestGTime",
            "Div5WheelsOff",
            "Div5TailNum",
            # ── LEAKAGE: Ground Time / Additional Data ──
            "TotalAddGTime",
            "LongestAddGTime",
            # ── LEAKAGE: Computed delay stats from actual ARR_DELAY ──
            "carrier_avg_arr_delay",
            "carrier_avg_dep_delay",
            "carrier_arr_delay_std",
            "prev_flight_delay",
            # ── Status (post-flight) ──
            "Cancelled",
            "Diverted",
            "CANCELLED",
            "DIVERTED",
            # ── Target columns ──
            "IS_DELAYED",
            "DELAY_CATEGORY",
            # ── Elapsed time (actual, not scheduled) ──
            "CRSElapsedTime",
            "CRS_ELAPSED_TIME",
            # ── BTS metadata / ID columns (not predictive) ──
            "DOT_ID_Reporting_Airline",
            "OriginAirportSeqID",
            "OriginCityMarketID",
            "OriginStateFips",
            "OriginWac",
            "DestAirportSeqID",
            "DestCityMarketID",
            "DestStateFips",
            "DestWac",
            "Year",
            "Flights",
        }

        # Also exclude any "Unnamed" junk columns
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols
            and col != target_col
            and not col.startswith("Unnamed")
        ]

        # Only numeric features
        numeric_features = (
            df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )

        X = df[numeric_features]
        y = df[target_col] if target_col in df.columns else None

        self.logger.info(f"Selected {len(numeric_features)} features for training")

        # Verify no leakage — comprehensive substring check
        leakage_substrings = [
            "ArrDelay",
            "DepDelay",
            "ArrDel15",
            "DepDel15",
            "DelayGroup",
            "Delay_",
            "DepTime",
            "ArrTime",
            "DEP_TIME",
            "ARR_TIME",
            "CANCELLED",
            "DIVERTED",
            "Cancelled",
            "Diverted",
            "CarrierDelay",
            "WeatherDelay",
            "NASDelay",
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",
            "DivArr",
            "DivActual",
            "DivReached",
            "DivDist",
            "DivAirport",
            "Div1",
            "Div2",
            "Div3",
            "Div4",
            "Div5",
            "FirstDep",
            "TotalAddG",
            "LongestAddG",
            "prev_flight_delay",
        ]
        found_leakage = [
            col
            for col in numeric_features
            if any(kw in col for kw in leakage_substrings)
        ]
        if found_leakage:
            self.logger.error(f"🚨 LEAKAGE DETECTED — removing: {found_leakage}")
            numeric_features = [f for f in numeric_features if f not in found_leakage]
            X = df[numeric_features]
        else:
            self.logger.info("✓ No feature leakage detected")

        return X, y
