import re

log_file = "experiments/leakage_check_v17.log"

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

match = re.search(
    r"\[LEAKAGE CHECK\] Training with \d+ features:\s*\[(.*?)\]", content, re.DOTALL
)
if match:
    features_str = match.group(1)
    # Clean up newlines and quotes
    features = [f.strip().strip("'") for f in features_str.split(",")]
    print(f"Found {len(features)} features:")
    for f in features:
        print(f)
else:
    print("Could not find feature list in log.")
