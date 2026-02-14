"""Quick test of data loader paths"""

from pathlib import Path
import sys

sys.path.insert(0, ".")

# Test what the current loader sees
from src.data.multiyear_loader import MultiYearDataLoader

loader = MultiYearDataLoader()

print(f"data_root attribute: {loader.data_root}")
print(f"data_root exists: {loader.data_root.exists()}")

if loader.data_root.exists():
    import os

    contents = os.listdir(loader.data_root)
    print(f"Contents of data_root: {contents}")

    # Check for year folders
    for year in [2023, 2024, 2025]:
        year_dir = loader.data_root / str(year)
        print(f"  {year}: exists={year_dir.exists()}")
else:
    print("ERROR: data_root directory not found!")

# What it SHOULD be
correct_path = Path(__file__).parent / "data" / "raw"
print(f"\nCorrect path should be: {correct_path.resolve()}")
print(f"Correct path exists: {correct_path.exists()}")
