import requests
import zipfile
from pathlib import Path

BASE_URL = "https://transtats.bts.gov/PREZIP/"
YEAR = 2025

# Output directories
RAW_DIR = Path("data/raw/2025")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract(year: int, month: int):
    filename = (
        f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    )
    url = BASE_URL + filename
    zip_path = RAW_DIR / filename

    if zip_path.exists():
        print(f"✓ {filename} already exists, skipping download")
        return

    print(f"⬇ Downloading {year}-{month:02d}...")
    response = requests.get(url, stream=True, timeout=300)

    if response.status_code != 200:
        print(f"✗ Failed to download {filename}")
        return

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✓ Downloaded {filename}")

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print(f"✓ Extracted {filename}\n")


def main():
    for month in range(1, 13):
        download_and_extract(YEAR, month)

    print("✅ All 2025 BTS datasets downloaded and extracted")


if __name__ == "__main__":
    main()
