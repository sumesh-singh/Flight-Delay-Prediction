"""
Unit Tests for Target Generator Module

Tests:
1. Binary target threshold correctness
2. Multiclass category boundaries
3. Regression target preservation
4. prepare_training_data leakage exclusions
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.features.target_generator import TargetGenerator, prepare_training_data


class TestBinaryTarget:
    def test_threshold_correctness(self):
        """Delays > threshold should be 1, otherwise 0."""
        tg = TargetGenerator(binary_threshold=15)
        df = pd.DataFrame({"ARR_DELAY": [-10, 0, 14, 15, 16, 100]})
        target = tg.create_binary_target(df)

        expected = pd.Series([0, 0, 0, 0, 1, 1], name="IS_DELAYED")
        pd.testing.assert_series_equal(target, expected, check_dtype=False)

    def test_custom_threshold(self):
        """Custom threshold should work correctly."""
        tg = TargetGenerator(binary_threshold=30)
        df = pd.DataFrame({"ARR_DELAY": [10, 29, 30, 31, 60]})
        target = tg.create_binary_target(df)

        expected = pd.Series([0, 0, 0, 1, 1], name="IS_DELAYED")
        pd.testing.assert_series_equal(target, expected, check_dtype=False)

    def test_all_on_time(self):
        """All negative delays should produce all zeros."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [-20, -10, -5, 0, 5]})
        target = tg.create_binary_target(df)

        assert target.sum() == 0

    def test_all_delayed(self):
        """All large delays should produce all ones."""
        tg = TargetGenerator(binary_threshold=15)
        df = pd.DataFrame({"ARR_DELAY": [20, 30, 60, 120]})
        target = tg.create_binary_target(df)

        assert target.sum() == 4

    def test_nan_handling(self):
        """NaN values should be treated as on-time (0)."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [30, np.nan, -5]})
        target = tg.create_binary_target(df)

        # NaN > 15 is False, so should be 0
        assert target.iloc[1] == 0

    def test_missing_column_raises(self):
        """Missing target column should raise ValueError."""
        tg = TargetGenerator()
        df = pd.DataFrame({"DISTANCE": [100, 200]})

        with pytest.raises(ValueError, match="not found"):
            tg.create_binary_target(df)


class TestMulticlassTarget:
    def test_category_boundaries(self):
        """Test default category boundaries."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [-20, 5, 25, 60]})
        target = tg.create_multiclass_target(df)

        assert target.iloc[0] == 0  # Early/OnTime (< 0)
        assert target.iloc[1] == 1  # Minor (0-15)
        assert target.iloc[2] == 2  # Moderate (15-45)
        assert target.iloc[3] == 3  # Severe (> 45)

    def test_four_categories_exist(self):
        """Should produce up to 4 categories."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [-50, 5, 25, 100]})
        target = tg.create_multiclass_target(df)

        assert set(target.unique()) == {0, 1, 2, 3}


class TestRegressionTarget:
    def test_preserves_values(self):
        """Regression target should preserve ARR_DELAY values."""
        tg = TargetGenerator()
        delays = [-10.5, 0, 15.3, 45.7]
        df = pd.DataFrame({"ARR_DELAY": delays})
        target = tg.create_regression_target(df)

        np.testing.assert_array_almost_equal(target.values, delays)

    def test_negative_values_preserved(self):
        """Negative delays (early arrivals) should be kept."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [-30, -15, -5]})
        target = tg.create_regression_target(df)

        assert all(target < 0)


class TestCreateAllTargets:
    def test_all_three_targets_created(self):
        """create_all_targets should produce 3 columns."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [-10, 5, 20, 60]})
        targets = tg.create_all_targets(df)

        assert "IS_DELAYED" in targets.columns
        assert "DELAY_CATEGORY" in targets.columns
        assert "ARR_DELAY" in targets.columns


class TestPrepareTrainingData:
    def test_arr_delay_excluded(self):
        """ARR_DELAY should not be in features."""
        df = pd.DataFrame(
            {
                "ARR_DELAY": [10, 20, 30],
                "DISTANCE": [100, 200, 300],
                "hour": [8, 14, 20],
            }
        )
        X, y = prepare_training_data(df, target_type="binary")

        assert "ARR_DELAY" not in X.columns

    def test_dep_delay_excluded(self):
        """DEP_DELAY should not be in features."""
        df = pd.DataFrame(
            {
                "ARR_DELAY": [10, 20, 30],
                "DEP_DELAY": [5, 15, 25],
                "DISTANCE": [100, 200, 300],
            }
        )
        X, y = prepare_training_data(df, target_type="binary")

        assert "DEP_DELAY" not in X.columns

    def test_target_excluded(self):
        """IS_DELAYED should not be in features."""
        df = pd.DataFrame(
            {
                "ARR_DELAY": [10, 20, 30],
                "IS_DELAYED": [0, 1, 1],
                "DISTANCE": [100, 200, 300],
            }
        )
        X, y = prepare_training_data(df, target_type="binary")

        assert "IS_DELAYED" not in X.columns

    def test_invalid_target_type_raises(self):
        """Invalid target_type should raise ValueError."""
        df = pd.DataFrame({"ARR_DELAY": [10], "DISTANCE": [100]})

        with pytest.raises(ValueError, match="Invalid target_type"):
            prepare_training_data(df, target_type="invalid")


class TestCreateTargetVariables:
    def test_adds_is_delayed_column(self):
        """create_target_variables should add IS_DELAYED to DataFrame."""
        tg = TargetGenerator()
        df = pd.DataFrame({"ARR_DELAY": [5, 20, -3, 50]})
        result = tg.create_target_variables(df)

        assert "IS_DELAYED" in result.columns
        assert result["IS_DELAYED"].iloc[0] == 0  # 5 <= 15
        assert result["IS_DELAYED"].iloc[1] == 1  # 20 > 15
