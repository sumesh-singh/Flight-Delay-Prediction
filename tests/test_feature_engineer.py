"""
Unit Tests for Feature Engineer Module

Tests:
1. Temporal feature creation (cyclical encoding range checks)
2. Carrier/airport/route/network feature creation
3. Human factors feature creation (NOVEL)
4. CRITICAL: Leakage prevention — verify no post-flight features in output
5. Feature selection consistency
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator


class TestTemporalFeatures:
    def test_cyclical_encoding_range(self, sample_flight_data):
        """Cyclical sin/cos features must be in [-1, 1]."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)

        for col in [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]:
            assert col in df.columns, f"Missing cyclical feature: {col}"
            assert df[col].min() >= -1.0, f"{col} min below -1"
            assert df[col].max() <= 1.0, f"{col} max above 1"

    def test_peak_hour_binary(self, sample_flight_data):
        """is_peak_hour should be 0 or 1 only."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)

        assert set(df["is_peak_hour"].unique()).issubset({0, 1})

    def test_weekend_flag(self, sample_flight_data):
        """is_weekend should be 0 or 1."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)

        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_temporal_features_count(self, sample_flight_data):
        """Should create 14 temporal features."""
        eng = FeatureEngineer()
        eng.engineered_features = []
        df = eng.create_temporal_features(sample_flight_data)

        assert len(eng.engineered_features) == 14


class TestCarrierFeatures:
    def test_carrier_features_created(self, sample_flight_data):
        """Carrier features should be created without errors."""
        eng = FeatureEngineer()
        eng.engineered_features = []
        df = eng.create_carrier_features(sample_flight_data)

        expected = [
            "carrier_avg_arr_delay",
            "carrier_arr_delay_std",
            "carrier_delay_rate",
            "carrier_avg_dep_delay",
        ]
        for feat in expected:
            assert feat in df.columns, f"Missing carrier feature: {feat}"

    def test_delay_rate_range(self, sample_flight_data):
        """carrier_delay_rate should be between 0 and 1."""
        eng = FeatureEngineer()
        df = eng.create_carrier_features(sample_flight_data)

        assert df["carrier_delay_rate"].min() >= 0.0
        assert df["carrier_delay_rate"].max() <= 1.0


class TestNetworkFeatures:
    def test_network_features_created(self, sample_flight_data):
        """Network features should be created with TAIL_NUM."""
        eng = FeatureEngineer()
        eng.engineered_features = []
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_network_features(df)

        assert "prev_flight_delay" in df.columns
        assert "turnaround_stress" in df.columns

    def test_turnaround_stress_binary(self, sample_flight_data):
        """turnaround_stress should be 0 or 1."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_network_features(df)

        assert set(df["turnaround_stress"].unique()).issubset({0, 1})


class TestHumanFactorsFeatures:
    """Tests for NOVEL human factors features — IEEE Limitation #3."""

    def test_crew_fatigue_features_created(self, sample_flight_data):
        """With TAIL_NUM, should create per-aircraft fatigue features."""
        eng = FeatureEngineer()
        eng.engineered_features = []
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_human_factors_features(df)

        assert "aircraft_daily_legs" in df.columns
        assert "aircraft_leg_number" in df.columns
        assert "crew_fatigue_index" in df.columns

    def test_crew_fatigue_index_range(self, sample_flight_data):
        """crew_fatigue_index should be between 0 and 1."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_human_factors_features(df)

        assert df["crew_fatigue_index"].min() > 0
        assert df["crew_fatigue_index"].max() <= 1.0

    def test_late_night_flag(self, sample_flight_data):
        """is_late_night_op should be 0 or 1."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_human_factors_features(df)

        assert "is_late_night_op" in df.columns
        assert set(df["is_late_night_op"].unique()).issubset({0, 1})

    def test_atc_workload_features(self, sample_flight_data):
        """ATC workload proxies should be created."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_human_factors_features(df)

        assert "origin_hourly_density" in df.columns
        assert "dest_hourly_density" in df.columns
        # Density must be >= 1 (at least 1 flight)
        assert df["origin_hourly_density"].min() >= 1

    def test_maintenance_stress_feature(self, sample_flight_data):
        """With TAIL_NUM + CRSElapsedTime, should create utilization feature."""
        eng = FeatureEngineer()
        df = eng.create_temporal_features(sample_flight_data)
        df = eng.create_human_factors_features(df)

        assert "aircraft_daily_util_min" in df.columns
        assert df["aircraft_daily_util_min"].min() > 0

    def test_human_factors_without_tail_num(self, sample_flight_data):
        """Without TAIL_NUM, should fall back to carrier-level proxy."""
        eng = FeatureEngineer()
        df = sample_flight_data.drop(columns=["TAIL_NUM"])
        df = eng.create_temporal_features(df)
        df = eng.create_human_factors_features(df)

        # Should still create basic features
        assert "aircraft_daily_legs" in df.columns
        assert "is_late_night_op" in df.columns
        assert "origin_hourly_density" in df.columns


class TestLeakagePrevention:
    """CRITICAL: Verify no post-flight features leak into training data."""

    def test_no_leakage_in_selected_features(self, sample_flight_data):
        """
        After full pipeline, select_features_for_training must NOT
        include ARR_TIME, DEP_TIME, CANCELLED, DIVERTED, ARR_DELAY, DEP_DELAY.
        """
        eng = FeatureEngineer()
        tg = TargetGenerator()
        df = tg.create_target_variables(sample_flight_data)
        df = eng.create_all_features(df, fit_encoders=True)
        X, y = eng.select_features_for_training(df, target_col="IS_DELAYED")

        forbidden_features = [
            "ARR_TIME",
            "DEP_TIME",
            "ARR_DELAY",
            "DEP_DELAY",
            "CANCELLED",
            "DIVERTED",
            "ArrTime",
            "DepTime",
            "ArrDelay",
            "DepDelay",
            "Cancelled",
            "Diverted",
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",
        ]

        for feat in forbidden_features:
            assert feat not in X.columns, f"LEAKAGE: {feat} found in training features!"

    def test_target_not_in_features(self, sample_flight_data):
        """IS_DELAYED target must not appear in feature matrix."""
        eng = FeatureEngineer()
        tg = TargetGenerator()
        df = tg.create_target_variables(sample_flight_data)
        df = eng.create_all_features(df, fit_encoders=True)
        X, y = eng.select_features_for_training(df, target_col="IS_DELAYED")

        assert "IS_DELAYED" not in X.columns
        assert y is not None
        assert len(y) == len(X)

    def test_features_are_numeric(self, sample_flight_data):
        """All selected features must be numeric."""
        eng = FeatureEngineer()
        tg = TargetGenerator()
        df = tg.create_target_variables(sample_flight_data)
        df = eng.create_all_features(df, fit_encoders=True)
        X, y = eng.select_features_for_training(df, target_col="IS_DELAYED")

        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), (
                f"Non-numeric feature: {col} (dtype={X[col].dtype})"
            )


class TestFullPipeline:
    def test_create_all_features_runs(self, sample_flight_data):
        """Full pipeline should run without errors."""
        eng = FeatureEngineer()
        tg = TargetGenerator()
        df = tg.create_target_variables(sample_flight_data)
        df = eng.create_all_features(df, fit_encoders=True)

        # Should have more columns than input
        assert df.shape[1] > sample_flight_data.shape[1]
        # Should have engineered features tracked
        assert len(eng.engineered_features) > 0

    def test_feature_count_reasonable(self, sample_flight_data):
        """Should create a reasonable number of features (20-60)."""
        eng = FeatureEngineer()
        tg = TargetGenerator()
        df = tg.create_target_variables(sample_flight_data)
        df = eng.create_all_features(df, fit_encoders=True)
        X, y = eng.select_features_for_training(df)

        assert X.shape[1] >= 20, f"Too few features: {X.shape[1]}"
        assert X.shape[1] <= 60, f"Too many features: {X.shape[1]}"
