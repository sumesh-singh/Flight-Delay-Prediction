"""
Export Phase 3 artifacts for Streamlit UI (Phase 4)

Generates:
- feature_order.json (already exists as *_features_*.json)
- label_encoders.pkl
- historical_stats.json
- distance_cache.json
- metadata.json

These artifacts enable inference-time feature computation in the Streamlit UI.
"""

import sys
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import RAW_DATA_DIR, MODELS_DIR


def export_label_encoders(df_raw: pd.DataFrame, model_dir: Path):
    """Export label encoders for categorical features."""
    print("\n[1/4] Exporting label encoders...")

    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    categorical_cols = ["OP_CARRIER", "ORIGIN", "DEST"]

    for col in categorical_cols:
        if col in df_raw.columns:
            le = LabelEncoder()
            le.fit(df_raw[col].astype(str))
            encoders[col] = le
            print(f"  {col}: {len(le.classes_)} unique values")

    output_path = model_dir / "label_encoders.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(encoders, f)

    print(f"[OK] Saved to: {output_path}")
    return encoders


def export_historical_stats(df_raw: pd.DataFrame, model_dir: Path):
    """Export historical aggregate statistics for inference-time lookup."""
    print("\n[2/4] Exporting historical statistics...")

    stats = {}

    # Carrier statistics (if DEP_DELAY available in raw)
    if "DEP_DELAY" in df_raw.columns and "OP_CARRIER" in df_raw.columns:
        carrier_stats = (
            df_raw.groupby("OP_CARRIER")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("std_delay", "std"),
                ]
            )
            .to_dict("index")
        )

        for carrier, vals in carrier_stats.items():
            stats[f"carrier_{carrier}_avg_delay"] = float(vals["avg_delay"])
            stats[f"carrier_{carrier}_delay_rate"] = float(vals["delay_rate"])
            stats[f"carrier_{carrier}_std_delay"] = (
                float(vals["std_delay"]) if not pd.isna(vals["std_delay"]) else 0.0
            )

        print(f"  Carrier stats: {len(carrier_stats)} carriers")

    # Airport statistics (origin)
    if "DEP_DELAY" in df_raw.columns and "ORIGIN" in df_raw.columns:
        origin_stats = (
            df_raw.groupby("ORIGIN")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("flight_count", "count"),
                ]
            )
            .to_dict("index")
        )

        for airport, vals in origin_stats.items():
            stats[f"origin_{airport}_avg_delay"] = float(vals["avg_delay"])
            stats[f"origin_{airport}_delay_rate"] = float(vals["delay_rate"])
            stats[f"origin_{airport}_flight_count"] = int(vals["flight_count"])

        print(f"  Origin airport stats: {len(origin_stats)} airports")

    # Route statistics
    if (
        "DEP_DELAY" in df_raw.columns
        and "ORIGIN" in df_raw.columns
        and "DEST" in df_raw.columns
    ):
        route_stats = (
            df_raw.groupby(["ORIGIN", "DEST"])["DEP_DELAY"]
            .agg([("avg_delay", "mean"), ("delay_rate", lambda x: (x > 15).mean())])
            .reset_index()
        )

        for _, row in route_stats.iterrows():
            route_key = f"route_{row['ORIGIN']}_{row['DEST']}"
            stats[f"{route_key}_avg_delay"] = float(row["avg_delay"])
            stats[f"{route_key}_delay_rate"] = float(row["delay_rate"])

        print(f"  Route stats: {len(route_stats)} routes")

    # Global fallbacks
    if "DEP_DELAY" in df_raw.columns:
        stats["global_avg_delay"] = float(df_raw["DEP_DELAY"].mean())
        stats["global_delay_rate"] = float((df_raw["DEP_DELAY"] > 15).mean())
        stats["global_std_delay"] = float(df_raw["DEP_DELAY"].std())
        print(
            f"  Global stats: avg_delay={stats['global_avg_delay']:.2f}, delay_rate={stats['global_delay_rate']:.2%}"
        )

    output_path = model_dir / "historical_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return stats


def export_distance_cache(df_raw: pd.DataFrame, model_dir: Path):
    """Export route distance cache for inference-time lookup."""
    print("\n[3/4] Exporting distance cache...")

    if (
        "ORIGIN" not in df_raw.columns
        or "DEST" not in df_raw.columns
        or "DISTANCE" not in df_raw.columns
    ):
        print("  [WARN] Required columns not found, skipping...")
        return {}

    # Get unique (origin, dest) → distance mapping
    distance_cache = df_raw.groupby(["ORIGIN", "DEST"])["DISTANCE"].first().to_dict()

    # Convert tuple keys to strings for JSON
    distance_cache_json = {f"{k[0]}_{k[1]}": int(v) for k, v in distance_cache.items()}

    # Add median distance as fallback
    distance_cache_json["_median_distance"] = int(df_raw["DISTANCE"].median())

    output_path = model_dir / "distance_cache.json"
    with open(output_path, "w") as f:
        json.dump(distance_cache_json, f, indent=2)

    print(f"  Cached {len(distance_cache)} unique routes")
    print(f"  Median distance: {distance_cache_json['_median_distance']} miles")
    print(f"[OK] Saved to: {output_path}")
    return distance_cache_json


def export_metadata(model_type: str, model_dir: Path, df_raw: pd.DataFrame):
    """Export model metadata (training info, metrics)."""
    print("\n[4/4] Exporting metadata...")

    # Load latest baseline report to get metrics
    baseline_dir = Path("experiments/baseline")
    report_files = list(baseline_dir.glob("baseline_report_*.json"))

    metadata = {
        "model_type": model_type,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_raw),
        "feature_count": None,  # Will be populated from features JSON
        "metrics": {},
    }

    if report_files:
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        with open(latest_report, "r") as f:
            report = json.load(f)

        # Extract relevant metrics
        run_key = f"{model_type.replace('_', '')}_run1"
        if run_key in report:
            metadata["metrics"] = report[run_key].get("test_metrics", {})
            metadata["training_time"] = report[run_key].get("training_time")

        print(f"  Loaded metrics from: {latest_report.name}")

    # Load feature count from features JSON
    features_file = list(model_dir.glob(f"{model_type}_features_*.json"))
    if features_file:
        with open(features_file[0], "r") as f:
            features_data = json.load(f)
            metadata["feature_count"] = len(features_data.get("features", []))

    output_path = model_dir / "metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return metadata


def main():
    print("=" * 70)
    print("EXPORTING PHASE 3 ARTIFACTS FOR STREAMLIT UI")
    print("=" * 70)

    # Load raw data (sample for statistics)
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("[ERROR] No raw data files found")
        return

    print(f"\nLoading raw data: {raw_files[0].name}")
    df_raw = pd.read_csv(raw_files[0], nrows=20000, low_memory=False)

    # Rename columns
    df_raw = df_raw.rename(columns={"Flight_Date_Year_Month_Day": "FL_DATE"})

    print(f"Loaded {len(df_raw):,} rows")

    # Export for both model types
    for model_type in ["logistic_regression", "random_forest"]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'=' * 70}")

        model_dir = MODELS_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        export_label_encoders(df_raw, model_dir)
        export_historical_stats(df_raw, model_dir)
        export_distance_cache(df_raw, model_dir)
        export_metadata(model_type, model_dir, df_raw)

    print("\n" + "=" * 70)
    print("[OK] ALL ARTIFACTS EXPORTED SUCCESSFULLY")
    print("=" * 70)
    print("\nStreamlit UI can now use these artifacts for inference.")
    return encoders


def export_historical_stats(df_raw: pd.DataFrame, model_dir: Path):
    """Export historical aggregate statistics for inference-time lookup."""
    print("\n[2/4] Exporting historical statistics...")

    stats = {}

    # Carrier statistics (if DEP_DELAY available in raw)
    if "DEP_DELAY" in df_raw.columns and "OP_CARRIER" in df_raw.columns:
        carrier_stats = (
            df_raw.groupby("OP_CARRIER")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("std_delay", "std"),
                ]
            )
            .to_dict("index")
        )

        for carrier, vals in carrier_stats.items():
            stats[f"carrier_{carrier}_avg_delay"] = float(vals["avg_delay"])
            stats[f"carrier_{carrier}_delay_rate"] = float(vals["delay_rate"])
            stats[f"carrier_{carrier}_std_delay"] = (
                float(vals["std_delay"]) if not pd.isna(vals["std_delay"]) else 0.0
            )

        print(f"  Carrier stats: {len(carrier_stats)} carriers")

    # Airport statistics (origin)
    if "DEP_DELAY" in df_raw.columns and "ORIGIN" in df_raw.columns:
        origin_stats = (
            df_raw.groupby("ORIGIN")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("flight_count", "count"),
                ]
            )
            .to_dict("index")
        )

        for airport, vals in origin_stats.items():
            stats[f"origin_{airport}_avg_delay"] = float(vals["avg_delay"])
            stats[f"origin_{airport}_delay_rate"] = float(vals["delay_rate"])
            stats[f"origin_{airport}_flight_count"] = int(vals["flight_count"])

        print(f"  Origin airport stats: {len(origin_stats)} airports")

    # Route statistics
    if (
        "DEP_DELAY" in df_raw.columns
        and "ORIGIN" in df_raw.columns
        and "DEST" in df_raw.columns
    ):
        route_stats = (
            df_raw.groupby(["ORIGIN", "DEST"])["DEP_DELAY"]
            .agg([("avg_delay", "mean"), ("delay_rate", lambda x: (x > 15).mean())])
            .reset_index()
        )

        for _, row in route_stats.iterrows():
            route_key = f"route_{row['ORIGIN']}_{row['DEST']}"
            stats[f"{route_key}_avg_delay"] = float(row["avg_delay"])
            stats[f"{route_key}_delay_rate"] = float(row["delay_rate"])

        print(f"  Route stats: {len(route_stats)} routes")

    # Global fallbacks
    if "DEP_DELAY" in df_raw.columns:
        stats["global_avg_delay"] = float(df_raw["DEP_DELAY"].mean())
        stats["global_delay_rate"] = float((df_raw["DEP_DELAY"] > 15).mean())
        stats["global_std_delay"] = float(df_raw["DEP_DELAY"].std())
        print(
            f"  Global stats: avg_delay={stats['global_avg_delay']:.2f}, delay_rate={stats['global_delay_rate']:.2%}"
        )

    output_path = model_dir / "historical_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return encoders


def export_historical_stats(df_raw: pd.DataFrame, model_dir: Path):
    """Export historical aggregate statistics for inference-time lookup."""
    print("\n[2/4] Exporting historical statistics...")

    stats = {}

    # Carrier statistics (if DEP_DELAY available in raw)
    if "DEP_DELAY" in df_raw.columns and "OP_CARRIER" in df_raw.columns:
        carrier_stats = (
            df_raw.groupby("OP_CARRIER")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("std_delay", "std"),
                ]
            )
            .to_dict("index")
        )

        for carrier, vals in carrier_stats.items():
            stats[f"carrier_{carrier}_avg_delay"] = float(vals["avg_delay"])
            stats[f"carrier_{carrier}_delay_rate"] = float(vals["delay_rate"])
            stats[f"carrier_{carrier}_std_delay"] = (
                float(vals["std_delay"]) if not pd.isna(vals["std_delay"]) else 0.0
            )

        print(f"  Carrier stats: {len(carrier_stats)} carriers")

    # Airport statistics (origin)
    if "DEP_DELAY" in df_raw.columns and "ORIGIN" in df_raw.columns:
        origin_stats = (
            df_raw.groupby("ORIGIN")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("flight_count", "count"),
                ]
            )
            .to_dict("index")
        )

        for airport, vals in origin_stats.items():
            stats[f"origin_{airport}_avg_delay"] = float(vals["avg_delay"])
            stats[f"origin_{airport}_delay_rate"] = float(vals["delay_rate"])
            stats[f"origin_{airport}_flight_count"] = int(vals["flight_count"])

        print(f"  Origin airport stats: {len(origin_stats)} airports")

    # Route statistics
    if (
        "DEP_DELAY" in df_raw.columns
        and "ORIGIN" in df_raw.columns
        and "DEST" in df_raw.columns
    ):
        route_stats = (
            df_raw.groupby(["ORIGIN", "DEST"])["DEP_DELAY"]
            .agg([("avg_delay", "mean"), ("delay_rate", lambda x: (x > 15).mean())])
            .reset_index()
        )

        for _, row in route_stats.iterrows():
            route_key = f"route_{row['ORIGIN']}_{row['DEST']}"
            stats[f"{route_key}_avg_delay"] = float(row["avg_delay"])
            stats[f"{route_key}_delay_rate"] = float(row["delay_rate"])

        print(f"  Route stats: {len(route_stats)} routes")

    # Global fallbacks
    if "DEP_DELAY" in df_raw.columns:
        stats["global_avg_delay"] = float(df_raw["DEP_DELAY"].mean())
        stats["global_delay_rate"] = float((df_raw["DEP_DELAY"] > 15).mean())
        stats["global_std_delay"] = float(df_raw["DEP_DELAY"].std())
        print(
            f"  Global stats: avg_delay={stats['global_avg_delay']:.2f}, delay_rate={stats['global_delay_rate']:.2%}"
        )

    output_path = model_dir / "historical_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return stats


def export_distance_cache(df_raw: pd.DataFrame, model_dir: Path):
    """Export route distance cache for inference-time lookup."""
    print("\n[3/4] Exporting distance cache...")

    if (
        "ORIGIN" not in df_raw.columns
        or "DEST" not in df_raw.columns
        or "DISTANCE" not in df_raw.columns
    ):
        print("  [WARN] Required columns not found, skipping...")
        return {}

    # Get unique (origin, dest) → distance mapping
    distance_cache = df_raw.groupby(["ORIGIN", "DEST"])["DISTANCE"].first().to_dict()

    # Convert tuple keys to strings for JSON
    distance_cache_json = {f"{k[0]}_{k[1]}": int(v) for k, v in distance_cache.items()}

    # Add median distance as fallback
    distance_cache_json["_median_distance"] = int(df_raw["DISTANCE"].median())

    output_path = model_dir / "distance_cache.json"
    with open(output_path, "w") as f:
        json.dump(distance_cache_json, f, indent=2)

    print(f"  Cached {len(distance_cache)} unique routes")
    print(f"  Median distance: {distance_cache_json['_median_distance']} miles")
    print(f"[OK] Saved to: {output_path}")
    return distance_cache_json


def export_metadata(model_type: str, model_dir: Path, df_raw: pd.DataFrame):
    """Export model metadata (training info, metrics)."""
    print("\n[4/4] Exporting metadata...")

    # Load latest baseline report to get metrics
    baseline_dir = Path("experiments/baseline")
    report_files = list(baseline_dir.glob("baseline_report_*.json"))

    metadata = {
        "model_type": model_type,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_raw),
        "feature_count": None,  # Will be populated from features JSON
        "metrics": {},
    }

    if report_files:
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        with open(latest_report, "r") as f:
            report = json.load(f)

        # Extract relevant metrics
        run_key = f"{model_type.replace('_', '')}_run1"
        if run_key in report:
            metadata["metrics"] = report[run_key].get("test_metrics", {})
            metadata["training_time"] = report[run_key].get("training_time")

        print(f"  Loaded metrics from: {latest_report.name}")

    # Load feature count from features JSON
    features_file = list(model_dir.glob(f"{model_type}_features_*.json"))
    if features_file:
        with open(features_file[0], "r") as f:
            features_data = json.load(f)
            metadata["feature_count"] = len(features_data.get("features", []))

    output_path = model_dir / "metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return metadata


def main():
    print("=" * 70)
    print("EXPORTING PHASE 3 ARTIFACTS FOR STREAMLIT UI")
    print("=" * 70)

    # Load raw data (sample for statistics)
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("[ERROR] No raw data files found")
        return

    print(f"\nLoading raw data: {raw_files[0].name}")
    df_raw = pd.read_csv(raw_files[0], nrows=20000, low_memory=False)

    # Rename columns
    df_raw = df_raw.rename(columns={"Flight_Date_Year_Month_Day": "FL_DATE"})

    print(f"Loaded {len(df_raw):,} rows")

    # Export for both model types
    for model_type in ["logistic_regression", "random_forest"]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'=' * 70}")

        model_dir = MODELS_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        export_label_encoders(df_raw, model_dir)
        export_historical_stats(df_raw, model_dir)
        export_distance_cache(df_raw, model_dir)
        export_metadata(model_type, model_dir, df_raw)

    print("\n" + "=" * 70)
    print("[OK] ALL ARTIFACTS EXPORTED SUCCESSFULLY")
    print("=" * 70)
    print("\nStreamlit UI can now use these artifacts for inference.")
    return stats


def export_distance_cache(df_raw: pd.DataFrame, model_dir: Path):
    """Export route distance cache for inference-time lookup."""
    print("\n[3/4] Exporting distance cache...")

    if (
        "ORIGIN" not in df_raw.columns
        or "DEST" not in df_raw.columns
        or "DISTANCE" not in df_raw.columns
    ):
        print("  ⚠️  Required columns not found, skipping...")
        return {}

    # Get unique (origin, dest) → distance mapping
    distance_cache = df_raw.groupby(["ORIGIN", "DEST"])["DISTANCE"].first().to_dict()

    # Convert tuple keys to strings for JSON
    distance_cache_json = {f"{k[0]}_{k[1]}": int(v) for k, v in distance_cache.items()}

    # Add median distance as fallback
    distance_cache_json["_median_distance"] = int(df_raw["DISTANCE"].median())

    output_path = model_dir / "distance_cache.json"
    with open(output_path, "w") as f:
        json.dump(distance_cache_json, f, indent=2)

    print(f"  Cached {len(distance_cache)} unique routes")
    print(f"  Median distance: {distance_cache_json['_median_distance']} miles")
    print(f"[OK] Saved to: {output_path}")
    return encoders


def export_historical_stats(df_raw: pd.DataFrame, model_dir: Path):
    """Export historical aggregate statistics for inference-time lookup."""
    print("\n[2/4] Exporting historical statistics...")

    stats = {}

    # Carrier statistics (if DEP_DELAY available in raw)
    if "DEP_DELAY" in df_raw.columns and "OP_CARRIER" in df_raw.columns:
        carrier_stats = (
            df_raw.groupby("OP_CARRIER")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("std_delay", "std"),
                ]
            )
            .to_dict("index")
        )

        for carrier, vals in carrier_stats.items():
            stats[f"carrier_{carrier}_avg_delay"] = float(vals["avg_delay"])
            stats[f"carrier_{carrier}_delay_rate"] = float(vals["delay_rate"])
            stats[f"carrier_{carrier}_std_delay"] = (
                float(vals["std_delay"]) if not pd.isna(vals["std_delay"]) else 0.0
            )

        print(f"  Carrier stats: {len(carrier_stats)} carriers")

    # Airport statistics (origin)
    if "DEP_DELAY" in df_raw.columns and "ORIGIN" in df_raw.columns:
        origin_stats = (
            df_raw.groupby("ORIGIN")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("flight_count", "count"),
                ]
            )
            .to_dict("index")
        )

        for airport, vals in origin_stats.items():
            stats[f"origin_{airport}_avg_delay"] = float(vals["avg_delay"])
            stats[f"origin_{airport}_delay_rate"] = float(vals["delay_rate"])
            stats[f"origin_{airport}_flight_count"] = int(vals["flight_count"])

        print(f"  Origin airport stats: {len(origin_stats)} airports")

    # Route statistics
    if (
        "DEP_DELAY" in df_raw.columns
        and "ORIGIN" in df_raw.columns
        and "DEST" in df_raw.columns
    ):
        route_stats = (
            df_raw.groupby(["ORIGIN", "DEST"])["DEP_DELAY"]
            .agg([("avg_delay", "mean"), ("delay_rate", lambda x: (x > 15).mean())])
            .reset_index()
        )

        for _, row in route_stats.iterrows():
            route_key = f"route_{row['ORIGIN']}_{row['DEST']}"
            stats[f"{route_key}_avg_delay"] = float(row["avg_delay"])
            stats[f"{route_key}_delay_rate"] = float(row["delay_rate"])

        print(f"  Route stats: {len(route_stats)} routes")

    # Global fallbacks
    if "DEP_DELAY" in df_raw.columns:
        stats["global_avg_delay"] = float(df_raw["DEP_DELAY"].mean())
        stats["global_delay_rate"] = float((df_raw["DEP_DELAY"] > 15).mean())
        stats["global_std_delay"] = float(df_raw["DEP_DELAY"].std())
        print(
            f"  Global stats: avg_delay={stats['global_avg_delay']:.2f}, delay_rate={stats['global_delay_rate']:.2%}"
        )

    output_path = model_dir / "historical_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return stats


def export_distance_cache(df_raw: pd.DataFrame, model_dir: Path):
    """Export route distance cache for inference-time lookup."""
    print("\n[3/4] Exporting distance cache...")

    if (
        "ORIGIN" not in df_raw.columns
        or "DEST" not in df_raw.columns
        or "DISTANCE" not in df_raw.columns
    ):
        print("  [WARN] Required columns not found, skipping...")
        return {}

    # Get unique (origin, dest) → distance mapping
    distance_cache = df_raw.groupby(["ORIGIN", "DEST"])["DISTANCE"].first().to_dict()

    # Convert tuple keys to strings for JSON
    distance_cache_json = {f"{k[0]}_{k[1]}": int(v) for k, v in distance_cache.items()}

    # Add median distance as fallback
    distance_cache_json["_median_distance"] = int(df_raw["DISTANCE"].median())

    output_path = model_dir / "distance_cache.json"
    with open(output_path, "w") as f:
        json.dump(distance_cache_json, f, indent=2)

    print(f"  Cached {len(distance_cache)} unique routes")
    print(f"  Median distance: {distance_cache_json['_median_distance']} miles")
    print(f"[OK] Saved to: {output_path}")
    return distance_cache_json


def export_metadata(model_type: str, model_dir: Path, df_raw: pd.DataFrame):
    """Export model metadata (training info, metrics)."""
    print("\n[4/4] Exporting metadata...")

    # Load latest baseline report to get metrics
    baseline_dir = Path("experiments/baseline")
    report_files = list(baseline_dir.glob("baseline_report_*.json"))

    metadata = {
        "model_type": model_type,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_raw),
        "feature_count": None,  # Will be populated from features JSON
        "metrics": {},
    }

    if report_files:
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        with open(latest_report, "r") as f:
            report = json.load(f)

        # Extract relevant metrics
        run_key = f"{model_type.replace('_', '')}_run1"
        if run_key in report:
            metadata["metrics"] = report[run_key].get("test_metrics", {})
            metadata["training_time"] = report[run_key].get("training_time")

        print(f"  Loaded metrics from: {latest_report.name}")

    # Load feature count from features JSON
    features_file = list(model_dir.glob(f"{model_type}_features_*.json"))
    if features_file:
        with open(features_file[0], "r") as f:
            features_data = json.load(f)
            metadata["feature_count"] = len(features_data.get("features", []))

    output_path = model_dir / "metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return metadata


def main():
    print("=" * 70)
    print("EXPORTING PHASE 3 ARTIFACTS FOR STREAMLIT UI")
    print("=" * 70)

    # Load raw data (sample for statistics)
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("[ERROR] No raw data files found")
        return

    print(f"\nLoading raw data: {raw_files[0].name}")
    df_raw = pd.read_csv(raw_files[0], nrows=20000, low_memory=False)

    # Rename columns
    df_raw = df_raw.rename(columns={"Flight_Date_Year_Month_Day": "FL_DATE"})

    print(f"Loaded {len(df_raw):,} rows")

    # Export for both model types
    for model_type in ["logistic_regression", "random_forest"]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'=' * 70}")

        model_dir = MODELS_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        export_label_encoders(df_raw, model_dir)
        export_historical_stats(df_raw, model_dir)
        export_distance_cache(df_raw, model_dir)
        export_metadata(model_type, model_dir, df_raw)

    print("\n" + "=" * 70)
    print("[OK] ALL ARTIFACTS EXPORTED SUCCESSFULLY")
    print("=" * 70)
    print("\nStreamlit UI can now use these artifacts for inference.")
    return distance_cache_json


def export_metadata(model_type: str, model_dir: Path, df_raw: pd.DataFrame):
    """Export model metadata (training info, metrics)."""
    print("\n[4/4] Exporting metadata...")

    # Load latest baseline report to get metrics
    baseline_dir = Path("experiments/baseline")
    report_files = list(baseline_dir.glob("baseline_report_*.json"))

    metadata = {
        "model_type": model_type,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_raw),
        "feature_count": None,  # Will be populated from features JSON
        "metrics": {},
    }

    if report_files:
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        with open(latest_report, "r") as f:
            report = json.load(f)

        # Extract relevant metrics
        run_key = f"{model_type.replace('_', '')}_run1"
        if run_key in report:
            metadata["metrics"] = report[run_key].get("test_metrics", {})
            metadata["training_time"] = report[run_key].get("training_time")

        print(f"  Loaded metrics from: {latest_report.name}")

    # Load feature count from features JSON
    features_file = list(model_dir.glob(f"{model_type}_features_*.json"))
    if features_file:
        with open(features_file[0], "r") as f:
            features_data = json.load(f)
            metadata["feature_count"] = len(features_data.get("features", []))

    output_path = model_dir / "metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return encoders


def export_historical_stats(df_raw: pd.DataFrame, model_dir: Path):
    """Export historical aggregate statistics for inference-time lookup."""
    print("\n[2/4] Exporting historical statistics...")

    stats = {}

    # Carrier statistics (if DEP_DELAY available in raw)
    if "DEP_DELAY" in df_raw.columns and "OP_CARRIER" in df_raw.columns:
        carrier_stats = (
            df_raw.groupby("OP_CARRIER")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("std_delay", "std"),
                ]
            )
            .to_dict("index")
        )

        for carrier, vals in carrier_stats.items():
            stats[f"carrier_{carrier}_avg_delay"] = float(vals["avg_delay"])
            stats[f"carrier_{carrier}_delay_rate"] = float(vals["delay_rate"])
            stats[f"carrier_{carrier}_std_delay"] = (
                float(vals["std_delay"]) if not pd.isna(vals["std_delay"]) else 0.0
            )

        print(f"  Carrier stats: {len(carrier_stats)} carriers")

    # Airport statistics (origin)
    if "DEP_DELAY" in df_raw.columns and "ORIGIN" in df_raw.columns:
        origin_stats = (
            df_raw.groupby("ORIGIN")["DEP_DELAY"]
            .agg(
                [
                    ("avg_delay", "mean"),
                    ("delay_rate", lambda x: (x > 15).mean()),
                    ("flight_count", "count"),
                ]
            )
            .to_dict("index")
        )

        for airport, vals in origin_stats.items():
            stats[f"origin_{airport}_avg_delay"] = float(vals["avg_delay"])
            stats[f"origin_{airport}_delay_rate"] = float(vals["delay_rate"])
            stats[f"origin_{airport}_flight_count"] = int(vals["flight_count"])

        print(f"  Origin airport stats: {len(origin_stats)} airports")

    # Route statistics
    if (
        "DEP_DELAY" in df_raw.columns
        and "ORIGIN" in df_raw.columns
        and "DEST" in df_raw.columns
    ):
        route_stats = (
            df_raw.groupby(["ORIGIN", "DEST"])["DEP_DELAY"]
            .agg([("avg_delay", "mean"), ("delay_rate", lambda x: (x > 15).mean())])
            .reset_index()
        )

        for _, row in route_stats.iterrows():
            route_key = f"route_{row['ORIGIN']}_{row['DEST']}"
            stats[f"{route_key}_avg_delay"] = float(row["avg_delay"])
            stats[f"{route_key}_delay_rate"] = float(row["delay_rate"])

        print(f"  Route stats: {len(route_stats)} routes")

    # Global fallbacks
    if "DEP_DELAY" in df_raw.columns:
        stats["global_avg_delay"] = float(df_raw["DEP_DELAY"].mean())
        stats["global_delay_rate"] = float((df_raw["DEP_DELAY"] > 15).mean())
        stats["global_std_delay"] = float(df_raw["DEP_DELAY"].std())
        print(
            f"  Global stats: avg_delay={stats['global_avg_delay']:.2f}, delay_rate={stats['global_delay_rate']:.2%}"
        )

    output_path = model_dir / "historical_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return stats


def export_distance_cache(df_raw: pd.DataFrame, model_dir: Path):
    """Export route distance cache for inference-time lookup."""
    print("\n[3/4] Exporting distance cache...")

    if (
        "ORIGIN" not in df_raw.columns
        or "DEST" not in df_raw.columns
        or "DISTANCE" not in df_raw.columns
    ):
        print("  [WARN] Required columns not found, skipping...")
        return {}

    # Get unique (origin, dest) → distance mapping
    distance_cache = df_raw.groupby(["ORIGIN", "DEST"])["DISTANCE"].first().to_dict()

    # Convert tuple keys to strings for JSON
    distance_cache_json = {f"{k[0]}_{k[1]}": int(v) for k, v in distance_cache.items()}

    # Add median distance as fallback
    distance_cache_json["_median_distance"] = int(df_raw["DISTANCE"].median())

    output_path = model_dir / "distance_cache.json"
    with open(output_path, "w") as f:
        json.dump(distance_cache_json, f, indent=2)

    print(f"  Cached {len(distance_cache)} unique routes")
    print(f"  Median distance: {distance_cache_json['_median_distance']} miles")
    print(f"[OK] Saved to: {output_path}")
    return distance_cache_json


def export_metadata(model_type: str, model_dir: Path, df_raw: pd.DataFrame):
    """Export model metadata (training info, metrics)."""
    print("\n[4/4] Exporting metadata...")

    # Load latest baseline report to get metrics
    baseline_dir = Path("experiments/baseline")
    report_files = list(baseline_dir.glob("baseline_report_*.json"))

    metadata = {
        "model_type": model_type,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_raw),
        "feature_count": None,  # Will be populated from features JSON
        "metrics": {},
    }

    if report_files:
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        with open(latest_report, "r") as f:
            report = json.load(f)

        # Extract relevant metrics
        run_key = f"{model_type.replace('_', '')}_run1"
        if run_key in report:
            metadata["metrics"] = report[run_key].get("test_metrics", {})
            metadata["training_time"] = report[run_key].get("training_time")

        print(f"  Loaded metrics from: {latest_report.name}")

    # Load feature count from features JSON
    features_file = list(model_dir.glob(f"{model_type}_features_*.json"))
    if features_file:
        with open(features_file[0], "r") as f:
            features_data = json.load(f)
            metadata["feature_count"] = len(features_data.get("features", []))

    output_path = model_dir / "metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved to: {output_path}")
    return metadata


def main():
    print("=" * 70)
    print("EXPORTING PHASE 3 ARTIFACTS FOR STREAMLIT UI")
    print("=" * 70)

    # Load raw data (sample for statistics)
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("[ERROR] No raw data files found")
        return

    print(f"\nLoading raw data: {raw_files[0].name}")
    df_raw = pd.read_csv(raw_files[0], nrows=20000, low_memory=False)

    # Rename columns
    df_raw = df_raw.rename(columns={"Flight_Date_Year_Month_Day": "FL_DATE"})

    print(f"Loaded {len(df_raw):,} rows")

    # Export for both model types
    for model_type in ["logistic_regression", "random_forest"]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'=' * 70}")

        model_dir = MODELS_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        export_label_encoders(df_raw, model_dir)
        export_historical_stats(df_raw, model_dir)
        export_distance_cache(df_raw, model_dir)
        export_metadata(model_type, model_dir, df_raw)

    print("\n" + "=" * 70)
    print("[OK] ALL ARTIFACTS EXPORTED SUCCESSFULLY")
    print("=" * 70)
    print("\nStreamlit UI can now use these artifacts for inference.")
    return metadata


def main():
    print("=" * 70)
    print("EXPORTING PHASE 3 ARTIFACTS FOR STREAMLIT UI")
    print("=" * 70)

    # Load raw data (sample for statistics)
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("❌ No raw data files found")
        return

    print(f"\nLoading raw data: {raw_files[0].name}")
    df_raw = pd.read_csv(raw_files[0], nrows=20000)

    # Rename columns
    df_raw = df_raw.rename(columns={"Flight_Date_Year_Month_Day": "FL_DATE"})

    print(f"Loaded {len(df_raw):,} rows")

    # Export for both model types
    for model_type in ["logistic_regression", "random_forest"]:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'=' * 70}")

        model_dir = MODELS_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        export_label_encoders(df_raw, model_dir)
        export_historical_stats(df_raw, model_dir)
        export_distance_cache(df_raw, model_dir)
        export_metadata(model_type, model_dir, df_raw)

    print("\n" + "=" * 70)
    print("✓ ALL ARTIFACTS EXPORTED SUCCESSFULLY")
    print("=" * 70)
    print("\nStreamlit UI can now use these artifacts for inference.")


if __name__ == "__main__":
    main()
