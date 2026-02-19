import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multiyear_loader import MultiYearDataLoader
from src.features.feature_engineer import FeatureEngineer


def verify_tail_num_integration():
    print("=" * 60)
    print("VERIFYING TAIL_NUM INTEGRATION (HUMAN FACTORS PROXY)")
    print("=" * 60)

    # 1. Verify Loader Configuration
    loader = MultiYearDataLoader()
    if "Tail_Number" in loader.ESSENTIAL_COLS:
        print("✓ 'Tail_Number' found in MultiYearDataLoader.ESSENTIAL_COLS")
    else:
        print("✗ 'Tail_Number' NOT found in MultiYearDataLoader.ESSENTIAL_COLS")
        return

    if "Tail_Number" in loader.DTYPE_MAP:
        print("✓ 'Tail_Number' found in MultiYearDataLoader.DTYPE_MAP")
    else:
        print("✗ 'Tail_Number' NOT found in MultiYearDataLoader.DTYPE_MAP")
        return

    # 2. Verify Data Loading (Mock or Real)
    # Create a small mock CSV to simulate loading if real data is large/slow
    # But better to try to load a chunk of 2023 data if available
    try:
        print("\nAttempting to load actual data chunk...")
        files = loader.get_available_files(2023)
        if not files:
            print("⚠ No 2023 data found. Skipping live load test.")
        else:
            df_chunk = loader.load_month_optimized(2023, 1, nrows=100)
            print(f"✓ Loaded chunk with {len(df_chunk)} rows")
            if "Tail_Number" in df_chunk.columns:
                print("✓ 'Tail_Number' column present in loaded DataFrame")
            else:
                print("✗ 'Tail_Number' column MISSING from loaded DataFrame")
                print(f"Columns: {df_chunk.columns.tolist()}")

            # Normalize column names (as per experiment runner)
            df_chunk = df_chunk.rename(
                columns={
                    "Tail_Number": "TAIL_NUM",
                    "FlightDate": "FL_DATE",
                    "Reporting_Airline": "OP_CARRIER",
                    "Origin": "ORIGIN",
                    "Dest": "DEST",
                    "CRSDepTime": "CRS_DEP_TIME",
                    "ArrDelay": "ARR_DELAY",
                }
            )

            # 3. Verify Feature Engineering
            print("\nTesting Feature Engineering...")
            engineer = FeatureEngineer(use_external_data=False)
            # We strictly test the internal Human Factor proxy logic here, preventing external API calls

            # Ensure required columns for network features
            df_chunk["CRS_DEP_TIME"] = df_chunk["CRS_DEP_TIME"].fillna(0)
            df_chunk["ARR_DELAY"] = df_chunk["ARR_DELAY"].fillna(0)

            # Run specific feature generation
            df_features = engineer.create_network_features(df_chunk)

            if "prev_flight_delay" in df_features.columns:
                print("✓ 'prev_flight_delay' created")

            # Check log for "Using TAIL_NUM" message would be ideal, but hard to capture here.
            # Instead, check if sorting worked (indirectly)
            print("✓ Feature Engineering ran without error")

    except Exception as e:
        print(f"✗ Error during data load/processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_tail_num_integration()
