"""
Shared pytest fixtures for Flight Delay Prediction tests.
Creates minimal BTS-like DataFrames for consistent testing.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_flight_data():
    """
    Create a minimal BTS-like DataFrame with all required columns
    for testing data processing, feature engineering, and model training.

    Returns a DataFrame with 200 rows spanning 5 airlines across 3 airports,
    realistic delay distributions, and TAIL_NUM for aircraft tracking.
    """
    np.random.seed(42)
    n = 200

    airlines = ["AA", "UA", "DL", "WN", "B6"]
    airports = ["ATL", "ORD", "LAX"]

    # Simulate realistic delay distribution
    delays = np.concatenate(
        [
            np.random.normal(-5, 8, int(n * 0.5)),  # 50% early/on-time
            np.random.normal(10, 5, int(n * 0.3)),  # 30% minor delays
            np.random.normal(40, 15, int(n * 0.2)),  # 20% moderate/severe
        ]
    )

    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "FL_DATE": np.random.choice(dates, n),
            "OP_CARRIER": np.random.choice(airlines, n),
            "TAIL_NUM": [
                f"N{np.random.randint(100, 999)}{c}"
                for c in np.random.choice(["AA", "UA", "DL"], n)
            ],
            "ORIGIN": np.random.choice(airports, n),
            "DEST": np.random.choice(airports, n),
            "CRS_DEP_TIME": np.random.choice(
                [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 100], n
            ),
            "CRS_ARR_TIME": np.random.choice(
                [900, 1100, 1300, 1500, 1700, 1900, 2100, 2300], n
            ),
            "DEP_TIME": np.random.choice([610, 815, 1005, 1215], n),
            "ARR_TIME": np.random.choice([910, 1115, 1305, 1515], n),
            "DEP_DELAY": delays * 0.8 + np.random.normal(0, 3, n),
            "ARR_DELAY": delays,
            "CANCELLED": np.zeros(n),
            "DIVERTED": np.zeros(n),
            "DISTANCE": np.random.randint(200, 2500, n),
            "CRSElapsedTime": np.random.randint(60, 360, n),
        }
    )

    # Ensure FL_DATE is datetime
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    return df


@pytest.fixture
def sample_engineered_data(sample_flight_data):
    """
    Return a DataFrame that has been through feature engineering.
    Also returns the y target for convenience.
    """
    from src.features.feature_engineer import FeatureEngineer
    from src.features.target_generator import TargetGenerator

    eng = FeatureEngineer(use_external_data=False)
    tg = TargetGenerator()

    df = tg.create_target_variables(sample_flight_data)
    df = eng.create_all_features(df, fit_encoders=True)

    return df, eng


@pytest.fixture
def small_flight_data():
    """Very small DataFrame (20 rows) for quick unit tests."""
    np.random.seed(0)
    n = 20
    return pd.DataFrame(
        {
            "FL_DATE": pd.date_range("2023-06-01", periods=n, freq="h"),
            "OP_CARRIER": np.random.choice(["AA", "UA"], n),
            "TAIL_NUM": [f"N{i:03d}AA" for i in range(n)],
            "ORIGIN": np.random.choice(["ATL", "ORD"], n),
            "DEST": np.random.choice(["LAX", "JFK"], n),
            "CRS_DEP_TIME": np.random.choice([800, 1400, 2000], n),
            "CRS_ARR_TIME": np.random.choice([1100, 1700, 2300], n),
            "ARR_DELAY": np.random.normal(5, 20, n),
            "DEP_DELAY": np.random.normal(3, 15, n),
            "DISTANCE": np.random.randint(300, 2000, n),
            "CRSElapsedTime": np.random.randint(90, 300, n),
            "CANCELLED": np.zeros(n),
            "DIVERTED": np.zeros(n),
        }
    )
