"""
Example: Basic Data Validation
Demonstrates simple validation of a flight delay dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import validation framework
from src.validation.validation_pipeline import validate_dataframe


def create_sample_data():
    """Create sample flight data for demonstration"""
    np.random.seed(42)
    n_samples = 5000

    # Create realistic sample data
    data = pd.DataFrame(
        {
            "FL_DATE": pd.date_range("2024-01-01", periods=n_samples, freq="5min"),
            "CRS_DEP_TIME": np.random.randint(500, 2200, n_samples),
            "CRS_ARR_TIME": np.random.randint(600, 2300, n_samples),
            "OP_CARRIER": np.random.choice(["AA", "DL", "UA", "WN", "B6"], n_samples),
            "ORIGIN": np.random.choice(["ATL", "ORD", "DFW", "LAX", "DEN"], n_samples),
            "DEST": np.random.choice(["ATL", "ORD", "DFW", "LAX", "DEN"], n_samples),
            "DISTANCE": np.random.randint(200, 2500, n_samples),
            "ARR_DELAY": np.random.normal(10, 25, n_samples),
        }
    )

    return data


def example_1_basic_validation():
    """Example 1: Basic validation - should PASS"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Validation (Clean Data)")
    print("=" * 70)

    # Create clean data
    df = create_sample_data()

    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Validate
    report = validate_dataframe(df, log_level="INFO")

    print(f"\n{'=' * 70}")
    print(f"RESULT: {report['status']}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"{'=' * 70}")

    return report


def example_2_missing_columns():
    """Example 2: Missing required columns - should FAIL"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Missing Required Columns")
    print("=" * 70)

    # Create data missing required columns
    df = create_sample_data()
    df = df.drop(columns=["ARR_DELAY", "DISTANCE"])  # Remove required columns

    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Validate
    report = validate_dataframe(df, log_level="WARNING")  # Less verbose

    print(f"\n{'=' * 70}")
    print(f"RESULT: {report['status']}")
    print(f"Errors: {report['errors']}")
    print(f"{'=' * 70}")

    return report


def example_3_data_leakage():
    """Example 3: Data leakage columns present - should FAIL"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Data Leakage Detection")
    print("=" * 70)

    # Create data with leakage columns
    df = create_sample_data()

    # Add forbidden leakage columns
    df["DEP_DELAY"] = np.random.normal(5, 15, len(df))  # ❌ Temporal leakage!
    df["CARRIER_DELAY"] = np.random.normal(
        10, 20, len(df)
    )  # ❌ Post-flight attribution!

    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print("\n⚠️  WARNING: Dataset contains DEP_DELAY and CARRIER_DELAY (leakage!)")

    # Validate
    report = validate_dataframe(df, log_level="WARNING")

    print(f"\n{'=' * 70}")
    print(f"RESULT: {report['status']}")
    print(f"Errors:")
    for error in report["errors"]:
        print(f"  - {error}")
    print(f"{'=' * 70}")

    return report


def example_4_range_violations():
    """Example 4: Range violations - should FAIL (strict mode)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Range Violations")
    print("=" * 70)

    # Create data with range violations
    df = create_sample_data()

    # Introduce range violations
    df.loc[0:50, "CRS_DEP_TIME"] = 2500  # Invalid time (>2359)
    df.loc[100:110, "DISTANCE"] = -500  # Negative distance!
    df.loc[200:220, "ARR_DELAY"] = 2000  # Extreme delay (>1440 min)

    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\n⚠️  Introduced violations:")
    print("  - 51 invalid CRS_DEP_TIME values (>2359)")
    print("  - 11 negative DISTANCE values")
    print("  - 21 extreme ARR_DELAY values (>1440 min)")

    # Validate
    report = validate_dataframe(df, log_level="WARNING")

    print(f"\n{'=' * 70}")
    print(f"RESULT: {report['status']}")
    print(f"Errors:")
    for error in report["errors"]:
        print(f"  - {error}")
    print(f"{'=' * 70}")

    return report


def example_5_drift_detection():
    """Example 5: Distribution drift - should WARN"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Drift Detection")
    print("=" * 70)

    # Create data with drift
    df = create_sample_data()

    # Introduce heavy delays (shift distribution)
    df["ARR_DELAY"] = df["ARR_DELAY"] + 30  # Shift all delays by 30 min

    # Define baseline statistics (historical)
    baseline_stats = {
        "delay_rate": 0.25,  # Historically 25% delay rate
        "OP_CARRIER": {"cardinality": 5},
        "ORIGIN": {"cardinality": 5},
    }

    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nBaseline delay rate: {baseline_stats['delay_rate'] * 100:.1f}%")

    # Calculate current delay rate
    current_delay_rate = (df["ARR_DELAY"] > 15).mean()
    print(f"Current delay rate: {current_delay_rate * 100:.1f}%")
    print(f"Change: +{(current_delay_rate - baseline_stats['delay_rate']) * 100:.1f}%")

    # Validate (with baseline)
    from src.validation.validation_pipeline import DataValidationPipeline

    pipeline = DataValidationPipeline(baseline_stats=baseline_stats)
    report = pipeline.validate(df)

    print(f"\n{'=' * 70}")
    print(f"RESULT: {report['status']}")
    print(f"Warnings:")
    for warning in report["warnings"]:
        print(f"  - {warning}")
    print(f"{'=' * 70}")

    return report


def example_6_production_pipeline():
    """Example 6: Production pipeline with validation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Production Pipeline Integration")
    print("=" * 70)

    # Simulate production pipeline
    print("\n📥 Loading raw data...")
    df_raw = create_sample_data()
    print(f"   Loaded: {df_raw.shape[0]:,} rows")

    print("\n🔍 Step 1: Validating data...")
    report = validate_dataframe(df_raw, log_level="WARNING")

    if report["status"] == "FAIL":
        print("\n❌ PIPELINE STOPPED: Validation failed!")
        print(f"   Errors: {report['errors']}")
        return None

    if report["status"] == "WARN":
        print(f"\n⚠️  Validation passed with {len(report['warnings'])} warnings")
    else:
        print("\n✅ Validation passed")

    print("\n🧹 Step 2: Data cleansing...")
    from src.data.data_cleanser import DataCleanser

    cleanser = DataCleanser()
    df_clean = cleanser.full_pipeline(df_raw)
    print(f"   Cleaned: {df_clean.shape[0]:,} rows")

    print("\n⚙️  Step 3: Feature engineering...")
    from src.features.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    print(f"   Features: {df_features.shape[1]} columns")

    print("\n🎯 Step 4: Target generation...")
    from src.features.target_generator import prepare_training_data

    X, y = prepare_training_data(df_features, target_type="binary")
    print(f"   X: {X.shape}, y: {y.shape}")
    print(f"   Delay rate: {y.mean() * 100:.1f}%")

    print(f"\n{'=' * 70}")
    print("✅ PIPELINE COMPLETE - Ready for model training!")
    print(f"{'=' * 70}")

    return X, y


if __name__ == "__main__":
    """Run all examples"""

    print("=" * 70)
    print("DATA VALIDATION FRAMEWORK - EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the data validation framework with")
    print("6 different scenarios:")
    print("\n1. Basic validation (clean data)")
    print("2. Missing required columns")
    print("3. Data leakage detection")
    print("4. Range violations")
    print("5. Drift detection")
    print("6. Production pipeline integration")

    input("\nPress Enter to start examples...")

    # Run examples
    try:
        example_1_basic_validation()
        input("\nPress Enter to continue to Example 2...")

        example_2_missing_columns()
        input("\nPress Enter to continue to Example 3...")

        example_3_data_leakage()
        input("\nPress Enter to continue to Example 4...")

        example_4_range_violations()
        input("\nPress Enter to continue to Example 5...")

        example_5_drift_detection()
        input("\nPress Enter to continue to Example 6...")

        example_6_production_pipeline()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
