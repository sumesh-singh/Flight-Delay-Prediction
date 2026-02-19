import joblib
import sys
from pathlib import Path

# Path to the specific experiment result
result_dir = Path("experiments/multiyear_results_20260216_015937")
feature_file = result_dir / "feature_names.joblib"

if feature_file.exists():
    features = joblib.load(feature_file)
    print(f"Total Features: {len(features)}")

    traffic_features = [f for f in features if "TRAFFIC" in f or "OPENSKY" in f]
    if traffic_features:
        print("\n[CONFIRMED] Traffic features found:")
        for f in traffic_features:
            print(f" - {f}")
    else:
        print("\n[WARNING] No traffic features found.")
else:
    print(f"File not found: {feature_file}")
