import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[1]))

from src.features.feature_engineer import FeatureEngineer
from src.data.external.external_manager import ExternalManager


def test_feature_engineer_integration():
    print("Initializing FeatureEngineer with use_external_data=True")
    engineer = FeatureEngineer(use_external_data=True)

    print(f"Engineer State: use_external_data={engineer.use_external_data}")
    print(f"Engineer State: external_manager={engineer.external_manager}")

    # Create dummy dataframe
    df = pd.DataFrame(
        {
            "FL_DATE": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "ORIGIN": ["JFK", "LAX"],
            "DEST": ["LAX", "JFK"],
            "CRS_DEP_TIME": [1200, 1800],
            "OP_CARRIER": ["AA", "DL"],
            "Reporting_Airline": ["AA", "DL"],
            "Flight_Number_Reporting_Airline": [100, 200],
            "OriginAirportID": [10001, 10002],
            "DestAirportID": [10002, 10001],
            "DISTANCE": [2500, 2500],
            "ARR_DELAY": [0, 0],
            "DEP_DELAY": [0, 0],
        }
    )

    print("\nRunning create_all_features...")
    try:
        df_out = engineer.create_all_features(df, fit_encoders=True)
        print("\nFeature Engineering Complete.")
        print("Columns:", df_out.columns.tolist())

        ext_cols = [
            c
            for c in df_out.columns
            if "ORIGIN_TMAX" in c or "ORIGIN_AIRPORT_TRAFFIC" in c
        ]
        if ext_cols:
            print(f"SUCCESS: Found {len(ext_cols)} external features: {ext_cols}")
        else:
            print("FAILURE: No external features found.")

    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_feature_engineer_integration()
