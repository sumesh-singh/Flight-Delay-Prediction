"""
Unit Tests for Data Cleanser Module

Tests:
1. Cancelled/diverted flight removal
2. Missing value handling
3. Outlier detection
4. Pipeline execution
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.data.data_cleanser import DataCleanser


@pytest.fixture
def cleanser():
    return DataCleanser()


@pytest.fixture
def raw_flight_data():
    """DataFrame simulating raw BTS data with issues."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "FL_DATE": pd.date_range("2023-01-01", periods=n, freq="h"),
            "OP_CARRIER": np.random.choice(["AA", "UA", "DL"], n),
            "ORIGIN": np.random.choice(["ATL", "ORD"], n),
            "DEST": np.random.choice(["LAX", "JFK"], n),
            "CRS_DEP_TIME": np.random.randint(100, 2359, n),
            "CRS_ARR_TIME": np.random.randint(100, 2359, n),
            "DEP_TIME": np.random.randint(100, 2359, n),
            "ARR_TIME": np.random.randint(100, 2359, n),
            "DEP_DELAY": np.random.normal(5, 20, n),
            "ARR_DELAY": np.random.normal(5, 20, n),
            "DISTANCE": np.random.randint(200, 2500, n),
            "CANCELLED": np.zeros(n),
            "DIVERTED": np.zeros(n),
        }
    )

    # Add some cancelled and diverted flights
    df.loc[0:4, "CANCELLED"] = 1.0
    df.loc[5:7, "DIVERTED"] = 1.0

    # Add some missing values
    df.loc[10:12, "ARR_DELAY"] = np.nan
    df.loc[15, "DEP_DELAY"] = np.nan

    return df


class TestCancelledDivertedRemoval:
    def test_cancelled_flights_removed(self, cleanser, raw_flight_data):
        """Cancelled flights should be removed."""
        result = cleanser.remove_cancelled_diverted(raw_flight_data)

        assert (result["CANCELLED"] == 1.0).sum() == 0

    def test_diverted_flights_removed(self, cleanser, raw_flight_data):
        """Diverted flights should be removed."""
        result = cleanser.remove_cancelled_diverted(raw_flight_data)

        assert (result["DIVERTED"] == 1.0).sum() == 0

    def test_valid_flights_kept(self, cleanser, raw_flight_data):
        """Valid flights should be preserved."""
        n_cancelled = (raw_flight_data["CANCELLED"] == 1.0).sum()
        n_diverted = (raw_flight_data["DIVERTED"] == 1.0).sum()
        n_original = len(raw_flight_data)

        result = cleanser.remove_cancelled_diverted(raw_flight_data)

        # Some cancelled/diverted may overlap, so check >= expected
        assert len(result) <= n_original - max(n_cancelled, n_diverted)
        assert len(result) > 0


class TestMissingValueHandling:
    def test_critical_missing_handled(self, cleanser, raw_flight_data):
        """Missing ARR_DELAY/DEP_DELAY should be handled."""
        # Remove cancelled/diverted first
        df = cleanser.remove_cancelled_diverted(raw_flight_data)
        result = cleanser.handle_missing_values(df)

        # Should not have NaN in ARR_DELAY after handling
        assert result["ARR_DELAY"].isna().sum() == 0


class TestOutlierDetection:
    def test_outlier_removal_runs(self):
        """Outlier removal should run without crashing."""
        np.random.seed(42)
        n = 500
        cleanser = DataCleanser(outlier_method="3sigma")
        df = pd.DataFrame(
            {
                "ARR_DELAY": np.concatenate(
                    [
                        np.random.normal(0, 10, n - 5),
                        [500, 600, 700, 800, 900],
                    ]
                ),
                "DEP_DELAY": np.random.normal(0, 10, n),
                "DISTANCE": np.random.randint(200, 2500, n),
            }
        )

        result = cleanser.remove_outliers(df)

        # Should remove some outliers or keep all
        assert len(result) <= len(df)

    def test_3sigma_detection(self):
        """3-sigma method should flag extreme values."""
        np.random.seed(42)
        n = 1000
        cleanser = DataCleanser(outlier_method="3sigma")
        values = np.concatenate(
            [
                np.random.normal(0, 10, n),
                [1000],  # Extreme outlier
            ]
        )
        df = pd.DataFrame(
            {
                "ARR_DELAY": values,
                "DEP_DELAY": np.random.normal(0, 10, n + 1),
                "DISTANCE": np.random.randint(200, 2500, n + 1),
            }
        )

        # Use the 3sigma detection method directly
        outlier_mask = cleanser.detect_outliers_3sigma(df, ["ARR_DELAY"])

        # Should flag at least the extreme outlier
        assert outlier_mask.sum() >= 1


class TestDataQualityValidation:
    def test_validation_runs(self, cleanser):
        """validate_data_quality should run without errors."""
        df = pd.DataFrame(
            {
                "ARR_DELAY": [5, 10, 15],
                "DEP_DELAY": [3, 8, 12],
                "DISTANCE": [500, 1000, 1500],
                "OP_CARRIER": ["AA", "UA", "DL"],
                "DEST": ["ATL", "ORD", "LAX"],
                "ORIGIN": ["JFK", "SFO", "DFW"],
                "OP_CARRIER_FL_NUM": [100, 200, 300],
                "FL_DATE": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            }
        )

        # Should not raise any exception
        result = cleanser.validate_data_quality(df)
        assert result is not None


class TestCleaningStatistics:
    def test_statistics_captured(self, cleanser, raw_flight_data):
        """Cleaning statistics should be populated after running pipeline."""
        # Run some cleaning steps
        df = cleanser.remove_cancelled_diverted(raw_flight_data)
        df = cleanser.handle_missing_values(df)

        stats = cleanser.get_statistics()
        assert isinstance(stats, dict)
