import pandas as pd
import glob
from pathlib import Path
import json
import sys

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
CONFIG_DIR = PROJECT_ROOT / "config"


def main():
    print("Starting station extraction...")

    # 1. Load Airport Metadata
    airports_path = METADATA_DIR / "airports.csv"
    if not airports_path.exists():
        print(f"Error: {airports_path} not found.")
        return

    print(f"Loading metadata from {airports_path}...")
    try:
        airports_df = pd.read_csv(airports_path)
        # Create IATA -> ICAO mapping (priority to large_airport, then medium)
        # Filter for relevant airports (Project scope mentions US BTS data)
        # We'll map globally but prioritize proper matches

        # Drop rows without iata_code or ident
        airports_df = airports_df.dropna(subset=["iata_code", "ident"])

        # Create dictionary
        iata_to_icao = pd.Series(
            airports_df.ident.values, index=airports_df.iata_code
        ).to_dict()
        print(f"Loaded {len(iata_to_icao)} IATA->ICAO mappings.")

    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # 2. Iterate through Raw Data
    years = ["2023", "2024", "2025"]
    unique_iata = set()

    for year in years:
        year_dir = RAW_DATA_DIR / year
        if not year_dir.exists():
            print(f"Directory {year_dir} does not exist, skipping.")
            continue

        csv_files = list(year_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {year}...")

        for csv_file in csv_files:
            try:
                # Read only 'Origin' column to save memory
                df = pd.read_csv(csv_file, usecols=["Origin"], low_memory=False)
                origins = df["Origin"].dropna().unique()
                unique_iata.update(origins)
                print(
                    f"  Processed {csv_file.name}: Found {len(origins)} unique stations."
                )
            except Exception as e:
                print(f"  Error processing {csv_file.name}: {e}")

    print(f"\nTotal unique IATA stations found in BTS data: {len(unique_iata)}")

    # 3. Map to ICAO
    unique_icao = []
    missing_icao = []

    for iata in unique_iata:
        icao = iata_to_icao.get(iata)
        if icao:
            unique_icao.append(icao)
        else:
            # Try prepending 'K' for US airports if not found (common convention, though not 100%)
            # Looking at the data, most are US.
            potential_icao = "K" + iata
            # Check if this constructed ICAO exists in the ident column direct lookup?
            # (Ideally we rely on the mapping, but the mapping might be incomplete)
            # For now, let's just log missing
            missing_icao.append(iata)

    # Remove duplicates
    unique_icao = sorted(list(set(unique_icao)))

    print(f"Mapped {len(unique_icao)} stations to ICAO codes.")
    if missing_icao:
        print(
            f"Warning: Could not map {len(missing_icao)} IATA codes: {missing_icao[:10]}..."
        )

    # 4. Save to JSON
    output_file = CONFIG_DIR / "stations.json"
    with open(output_file, "w") as f:
        json.dump(unique_icao, f, indent=2)

    print(f"Saved station list to {output_file}")


if __name__ == "__main__":
    main()
