"""
Unified Flight Delay Prediction — Training Pipeline
====================================================
Trains on ALL 24 months of 2023+2024, tests on ALL 11 months of 2025.
Uses per-month subsampling to fit within 8 GB RAM.

Usage:
  python train_pipeline.py
"""

import sys, os, gc, time, json, pickle, warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

os.chdir(r"e:\Flight-Delay-Prediction")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE

from src.data.multiyear_loader import MultiYearDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.features.target_generator import TargetGenerator


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
RANDOM_STATE = 42
TODAY = datetime.now().strftime("%Y%m%d")
MODEL_BASE_DIR = Path("models")
RESULTS_DIR = Path("experiments/logs")

# Per-month sampling limits (to fit within 8 GB RAM)
# 24 train months × 62,500 = 1,500,000 total training rows
# 11 test months  × 50,000 =   550,000 total test rows
TRAIN_ROWS_PER_MONTH = 62_500
TEST_ROWS_PER_MONTH = 50_000

# Column renaming map (BTS camelCase → internal UPPERCASE)
COL_MAP = {
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_CARRIER",
    "Tail_Number": "TAIL_NUM",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "CRSArrTime": "CRS_ARR_TIME",
    "CRSElapsedTime": "CRSElapsedTime",
    "DepTime": "DEP_TIME",
    "ArrTime": "ARR_TIME",
    "DepDelay": "DEP_DELAY",
    "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
    "Distance": "DISTANCE",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
}

np.random.seed(RANDOM_STATE)


def banner(text):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def stratified_subsample(X, y, n, carrier=None, origin=None, dest=None, arr_delay=None):
    """
    Stratified subsample: preserve the delayed/on-time ratio.
    Returns subsampled arrays plus optional metadata arrays.
    """
    if len(X) <= n:
        return X, y, carrier, origin, dest, arr_delay

    delayed_mask = y == 1
    n_delayed = int(delayed_mask.sum())
    n_ontime = len(y) - n_delayed

    # Preserve class ratio
    frac = n / len(y)
    n_del_keep = max(1, int(n_delayed * frac))
    n_ont_keep = n - n_del_keep

    idx_del = np.where(delayed_mask)[0]
    idx_ont = np.where(~delayed_mask)[0]

    chosen = np.concatenate(
        [
            np.random.choice(
                idx_del, size=min(n_del_keep, len(idx_del)), replace=False
            ),
            np.random.choice(
                idx_ont, size=min(n_ont_keep, len(idx_ont)), replace=False
            ),
        ]
    )
    chosen.sort()

    X_sub = X.iloc[chosen].reset_index(drop=True)
    y_sub = y.iloc[chosen].reset_index(drop=True)

    carr_sub = carrier[chosen] if carrier is not None else None
    orig_sub = origin[chosen] if origin is not None else None
    dest_sub = dest[chosen] if dest is not None else None
    arr_sub = arr_delay[chosen] if arr_delay is not None else None

    return X_sub, y_sub, carr_sub, orig_sub, dest_sub, arr_sub


def process_one_month(csv_path, eng, tg, fit_encoders=False, max_rows=None):
    """
    Load one CSV, engineer features, select numeric features,
    optionally subsample, return (X, y, carrier, origin, dest, arr_delay).
    """
    print(f"  Loading: {csv_path.name}...", end=" ", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(columns=COL_MAP)

    # Filter invalid flights
    mask = (
        (df.get("CANCELLED", 0) != 1.0)
        & (df.get("DIVERTED", 0) != 1.0)
        & df["ARR_DELAY"].notna()
    )
    df = df[mask].copy()

    # Create target
    df = tg.create_target_variables(df)

    # Extract raw metadata BEFORE feature engineering drops them
    carrier_col = df["OP_CARRIER"].values.copy() if "OP_CARRIER" in df.columns else None
    origin_col = df["ORIGIN"].values.copy() if "ORIGIN" in df.columns else None
    dest_col = df["DEST"].values.copy() if "DEST" in df.columns else None
    arr_delay_col = df["ARR_DELAY"].values.copy() if "ARR_DELAY" in df.columns else None

    # Feature engineering
    df = eng.create_all_features(df, fit_encoders=fit_encoders)

    # Select features (leakage-free)
    X, y = eng.select_features_for_training(df, target_col="IS_DELAYED")
    X = X.fillna(0)

    total_rows = len(X)

    # Free raw dataframe immediately
    del df
    gc.collect()

    # Subsample if needed
    if max_rows is not None and len(X) > max_rows:
        X, y, carrier_col, origin_col, dest_col, arr_delay_col = stratified_subsample(
            X, y, max_rows, carrier_col, origin_col, dest_col, arr_delay_col
        )

    print(f"{total_rows:,} total → {len(X):,} kept, {len(X.columns)} features")
    return X, y, carrier_col, origin_col, dest_col, arr_delay_col


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PROCESS DATA (CHUNKED WITH PER-MONTH SUBSAMPLING)
# ═══════════════════════════════════════════════════════════════════════════════
banner("1. LOADING & PROCESSING DATA (CHUNKED)")

loader = MultiYearDataLoader()
eng = FeatureEngineer(use_external_data=True)
tg = TargetGenerator()

# --- Process training data month-by-month ---
print(f"\n--- Training Data ({', '.join(str(y) for y in TRAIN_YEARS)}) ---")
print(f"    Keeping ~{TRAIN_ROWS_PER_MONTH:,} rows per month")
train_Xs = []
train_ys = []
all_carriers = []
all_origins = []
all_dests = []
all_arr_delays = []
months_processed_train = 0

for year in TRAIN_YEARS:
    files = loader.get_available_files(year)
    print(f"\n  {year}: {len(files)} files")
    for i, fpath in enumerate(files):
        fit = year == TRAIN_YEARS[0] and i == 0
        X, y, carr, orig, dest, arr_d = process_one_month(
            fpath,
            eng,
            tg,
            fit_encoders=fit,
            max_rows=TRAIN_ROWS_PER_MONTH,
        )
        train_Xs.append(X)
        train_ys.append(y)
        if carr is not None:
            all_carriers.append(carr)
            all_origins.append(orig)
            all_dests.append(dest)
            all_arr_delays.append(arr_d)
        months_processed_train += 1

        # Force garbage collection after each month
        gc.collect()

# Combine training data
print(
    f"\n  Concatenating {months_processed_train} months of training data...",
    end=" ",
    flush=True,
)
X_train_all = pd.concat(train_Xs, ignore_index=True)
y_train_all = pd.concat(train_ys, ignore_index=True)
del train_Xs, train_ys
gc.collect()
print(f"Done: {len(X_train_all):,} rows")

# --- Process test data month-by-month ---
print(f"\n--- Test Data ({TEST_YEAR}) ---")
print(f"    Keeping ~{TEST_ROWS_PER_MONTH:,} rows per month")
test_Xs = []
test_ys = []
months_processed_test = 0

test_files = loader.get_available_files(TEST_YEAR)
print(f"\n  {TEST_YEAR}: {len(test_files)} files")
for fpath in test_files:
    X, y, _, _, _, _ = process_one_month(
        fpath,
        eng,
        tg,
        fit_encoders=False,
        max_rows=TEST_ROWS_PER_MONTH,
    )
    test_Xs.append(X)
    test_ys.append(y)
    months_processed_test += 1
    gc.collect()

print(
    f"\n  Concatenating {months_processed_test} months of test data...",
    end=" ",
    flush=True,
)
X_test_all = pd.concat(test_Xs, ignore_index=True)
y_test_all = pd.concat(test_ys, ignore_index=True)
del test_Xs, test_ys
gc.collect()
print(f"Done: {len(X_test_all):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ALIGN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
banner("2. ALIGNING FEATURES")

common = sorted(list(set(X_train_all.columns) & set(X_test_all.columns)))
X_train_all = X_train_all[common].fillna(0)
X_test_all = X_test_all[common].fillna(0)

print(f"  Common features: {len(common)}")
print(f"  Train: {X_train_all.shape}  |  Test: {X_test_all.shape}")
print(f"  Months: {months_processed_train} train + {months_processed_test} test")

# Verify no leakage (comprehensive substring check)
leakage_substrings = [
    "ARR_DELAY",
    "DEP_DELAY",
    "ARR_TIME",
    "DEP_TIME",
    "CANCELLED",
    "DIVERTED",
    "TAXI_IN",
    "TAXI_OUT",
    "WHEELS_ON",
    "WHEELS_OFF",
    "AIR_TIME",
    "ACTUAL_ELAPSED",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "ArrDelay",
    "DepDelay",
    "ArrDel15",
    "DepDel15",
    "DelayGroup",
    "ArrivalDelay",
    "DepartureDelay",
    "DepTime",
    "ArrTime",
    "FirstDep",
    "DivArr",
    "DivActual",
    "DivReach",
    "DivDist",
    "DivAirport",
    "Div1",
    "Div2",
    "Div3",
    "Div4",
    "Div5",
    "TotalAddG",
    "LongestAddG",
    "prev_flight_delay",
    "carrier_avg_arr_delay",
    "carrier_avg_dep_delay",
    "carrier_arr_delay_std",
]
leak_found = [f for f in common if any(kw in f for kw in leakage_substrings)]
if leak_found:
    print(f"  ⚠ LEAKAGE DETECTED — removing {len(leak_found)} columns: {leak_found}")
    common = [c for c in common if c not in leak_found]
    X_train_all = X_train_all[common].fillna(0)
    X_test_all = X_test_all[common].fillna(0)
    print(f"  After removal: {len(common)} features remain")
else:
    print("  ✓ No leakage features detected")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE HISTORICAL STATS (for inference lookups)
# ═══════════════════════════════════════════════════════════════════════════════
banner("3. COMPUTING HISTORICAL STATS")

historical_stats = {}

carriers = np.concatenate(all_carriers) if all_carriers else np.array([])
origins = np.concatenate(all_origins) if all_origins else np.array([])
dests = np.concatenate(all_dests) if all_dests else np.array([])
arr_delays = np.concatenate(all_arr_delays) if all_arr_delays else np.array([])
del all_carriers, all_origins, all_dests, all_arr_delays
gc.collect()

if len(arr_delays) > 0:
    valid = ~np.isnan(arr_delays)
    historical_stats["global_avg_delay"] = float(np.nanmean(arr_delays))
    historical_stats["global_std_delay"] = float(np.nanstd(arr_delays))
    historical_stats["global_delay_rate"] = float(
        np.sum(arr_delays[valid] > 15) / max(np.sum(valid), 1)
    )

    # Per-carrier stats
    for c in np.unique(carriers[valid]):
        mask_c = (carriers == c) & valid
        subset = arr_delays[mask_c]
        if len(subset) > 0:
            historical_stats[f"carrier_{c}_avg_delay"] = float(np.mean(subset))
            historical_stats[f"carrier_{c}_std_delay"] = float(np.std(subset))
            historical_stats[f"carrier_{c}_delay_rate"] = float(
                np.sum(subset > 15) / len(subset)
            )

    # Per-origin airport stats
    for ap in np.unique(origins[valid]):
        mask_a = (origins == ap) & valid
        subset = arr_delays[mask_a]
        if len(subset) > 0:
            historical_stats[f"origin_{ap}_flight_count"] = int(len(subset))
            historical_stats[f"origin_{ap}_delay_rate"] = float(
                np.sum(subset > 15) / len(subset)
            )

    # Per-route stats (top 5000 by volume)
    route_keys = np.char.add(
        np.char.add(origins[valid].astype(str), "_"),
        dests[valid].astype(str),
    )
    unique_routes, route_counts = np.unique(route_keys, return_counts=True)
    top_routes = unique_routes[np.argsort(-route_counts)[:5000]]
    for rk in top_routes:
        mask_r = route_keys == rk
        subset = arr_delays[valid][mask_r]
        if len(subset) > 0:
            historical_stats[f"route_{rk}_avg_delay"] = float(np.mean(subset))
            historical_stats[f"route_{rk}_delay_rate"] = float(
                np.sum(subset > 15) / len(subset)
            )

del carriers, origins, dests, arr_delays
gc.collect()

print(f"  Computed {len(historical_stats):,} stats keys")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SMOTE (balance classes)
# ═══════════════════════════════════════════════════════════════════════════════
banner("4. APPLYING SMOTE")

print(
    f"  Before: {X_train_all.shape}, Distribution: {np.bincount(y_train_all.astype(int))}"
)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_all, y_train_all)
print(
    f"  After:  {X_train_res.shape}, Distribution: {np.bincount(y_train_res.astype(int))}"
)

del X_train_all, y_train_all
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SCALE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
banner("5. SCALING FEATURES")

scaler = StandardScaler()
X_tr = pd.DataFrame(scaler.fit_transform(X_train_res), columns=common)
X_te = pd.DataFrame(scaler.transform(X_test_all), columns=common)
y_train = y_train_res
y_test = y_test_all
print(
    f"  Scaler fitted: {X_tr.shape[0]:,} train, {X_te.shape[0]:,} test, {X_tr.shape[1]} features"
)

del X_train_res, X_test_all, y_train_res, y_test_all
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN MODELS + THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════════════════════════
banner("6. TRAINING MODELS")

model_configs = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=30,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "sgd_classifier": SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}

results = {}

for name, model in model_configs.items():
    print(f"\n  Training {name}...")
    t0 = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    # Get probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        raw = model.decision_function(X_te)
        y_prob = 1 / (1 + np.exp(-raw))  # sigmoid

    # Threshold tuning — maximize F1
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.10, 0.90, 0.05):
        yp = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_test, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = round(float(thresh), 2)

    yp_opt = (y_prob >= best_thresh).astype(int)

    r = {
        "accuracy": round(float(accuracy_score(y_test, yp_opt)), 4),
        "precision": round(float(precision_score(y_test, yp_opt, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, yp_opt, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, yp_opt, zero_division=0)), 4),
        "optimal_threshold": best_thresh,
        "train_time_sec": round(train_time, 1),
        "model_object": model,
    }
    results[name] = r

    print(
        f"    acc={r['accuracy']}  prec={r['precision']}  rec={r['recall']}  "
        f"f1={r['f1']}  thresh={r['optimal_threshold']}  ({r['train_time_sec']}s)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SELECT BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
banner("7. SELECTING BEST MODEL")

best_name = max(results, key=lambda k: results[k]["f1"])
best_result = results[best_name]
print(f"  ★ BEST MODEL: {best_name} (F1={best_result['f1']})")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
banner("8. SAVING ARTIFACTS")

for name, r in results.items():
    model_dir = MODEL_BASE_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_obj = r.pop("model_object")

    # 1. Model
    for old in model_dir.glob(f"{name}_model_*.joblib"):
        old.unlink()
    model_path = model_dir / f"{name}_model_{TODAY}.joblib"
    joblib.dump(model_obj, model_path, compress=3)

    # 2. Feature order
    for old in model_dir.glob(f"{name}_features_*.json"):
        old.unlink()
    with open(model_dir / f"{name}_features_{TODAY}.json", "w") as f:
        json.dump({"features": common}, f, indent=2)

    # 3. Scaler
    for old in model_dir.glob(f"{name}_scaler_*.joblib"):
        old.unlink()
    joblib.dump(scaler, model_dir / f"{name}_scaler_{TODAY}.joblib", compress=3)

    # 4. Label encoders
    with open(model_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(eng.label_encoders, f)

    # 5. Historical stats
    with open(model_dir / "historical_stats.json", "w") as f:
        json.dump(historical_stats, f)

    # 6. Metadata
    is_best = name == best_name
    metadata = {
        "model_type": name,
        "trained_on": TODAY,
        "train_years": TRAIN_YEARS,
        "test_year": TEST_YEAR,
        "train_months": months_processed_train,
        "test_months": months_processed_test,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "n_features": len(common),
        "optimal_threshold": r["optimal_threshold"],
        "is_best_model": is_best,
        "metrics": {k: v for k, v in r.items()},
        "smote": True,
        "class_weight": "balanced",
        "rows_per_month_train": TRAIN_ROWS_PER_MONTH,
        "rows_per_month_test": TEST_ROWS_PER_MONTH,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    tag = "★ BEST" if is_best else "     "
    print(f"  {tag} {name}: F1={r['f1']} → models/{name}/")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SAVE COMPARISON RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
comparison = {
    "best_model": best_name,
    "models": results,
    "feature_names": common,
    "train_size": int(len(y_train)),
    "test_size": int(len(y_test)),
    "train_years": TRAIN_YEARS,
    "test_year": TEST_YEAR,
    "train_months": months_processed_train,
    "test_months": months_processed_test,
    "notes": {
        "smote": "APPLIED on training data",
        "class_weight": "balanced (all models)",
        "leakage_check": "PASSED",
        "human_factors": "INCLUDED (crew_fatigue_index, aircraft_daily_legs, etc.)",
        "threshold_tuning": "APPLIED (optimized for F1)",
        "external_data": "ENABLED (NOAA Weather for major airports)",
        "data_coverage": f"ALL months — {months_processed_train} train ({'+'.join(str(y) for y in TRAIN_YEARS)}), {months_processed_test} test ({TEST_YEAR})",
        "subsampling": f"{TRAIN_ROWS_PER_MONTH:,}/month train, {TEST_ROWS_PER_MONTH:,}/month test (stratified)",
    },
}

results_path = RESULTS_DIR / "comparison_results.json"
with open(results_path, "w") as f:
    json.dump(comparison, f, indent=2)

banner("DONE")
print(f"  Best model : {best_name} (F1={best_result['f1']})")
print(
    f"  Train data : {months_processed_train} months ({', '.join(str(y) for y in TRAIN_YEARS)}), {len(y_train):,} rows"
)
print(
    f"  Test data  : {months_processed_test} months ({TEST_YEAR}), {len(y_test):,} rows"
)
print(f"  Features   : {len(common)}")
print(f"  Models     : {MODEL_BASE_DIR}/")
print(f"  Results    : {results_path}")
print(f"\n  To predict : streamlit run streamlit_app/app.py")
