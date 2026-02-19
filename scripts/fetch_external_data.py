"""
Fetch External Data Script
Pre-fetches weather and traffic data for all airports and dates in the dataset.
Populates the local cache to ensure fast training.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parents[1]))

from src.data.external.airport_mapper import AirportMapper
from src.data.external.noaa_client import NOAAClient
from config.data_config import RAW_DATA_DIR, EXTERNAL_DATA_DIR, BTS_FILENAME_TEMPLATE

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(EXTERNAL_DATA_DIR / "fetch_log.txt"),
    ],
)
logger = logging.getLogger(__name__)


def get_dataset_scope():
    """
    Scan raw data to find unique airports and date range.
    """
    logger.info("Scanning raw datasets to determine scope...")

    unique_airports = set()
    min_date = None
    max_date = None

    # Iterate over available files (2024-2025)
    # We'll just look at a few representative files or all if possible.
    # To be fast, let's look at one file per year/month if many.

    csv_files = list(RAW_DATA_DIR.rglob("*.zip")) + list(RAW_DATA_DIR.rglob("*.csv"))

    if not csv_files:
        logger.error(f"No data files found in {RAW_DATA_DIR}")
        return set(), None, None

    for file_path in csv_files:
        try:
            # Read only essential columns (Raw CSV names: FlightDate, Origin, Dest)
            # Note: BTS raw files have capitalized names like 'FlightDate' etc.
            df = pd.read_csv(
                file_path,
                usecols=["FlightDate", "Origin", "Dest"],
                parse_dates=["FlightDate"],
            )

            # Standardize names
            df = df.rename(
                columns={"FlightDate": "FL_DATE", "Origin": "ORIGIN", "Dest": "DEST"}
            )

            unique_airports.update(df["ORIGIN"].unique())
            unique_airports.update(df["DEST"].unique())

            file_min = df["FL_DATE"].min()
            file_max = df["FL_DATE"].max()

            if min_date is None or file_min < min_date:
                min_date = file_min
            if max_date is None or file_max > max_date:
                max_date = file_max

            logger.info(
                f"Scanned {file_path.name}: {len(df)} rows, Range {file_min.date()} to {file_max.date()}"
            )

        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")

    return unique_airports, min_date, max_date


def fetch_weather_data(airports, start_date, end_date):
    """
    Fetch weather for all airports.
    """
    logger.info(f"--- Starting Weather Fetch for {len(airports)} airports ---")

    mapper = AirportMapper()
    noaa = NOAAClient()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    success_count = 0
    fail_count = 0

    # Prioritize top 50 (mapper has only top 50 coords usually)
    # But mapper.get_coordinates works for all? No, it relies on CSV.
    # Check coverage.

    for i, iata in enumerate(sorted(list(airports))):
        try:
            # Check if we have coords
            lat, lon = mapper.get_coordinates(iata)
            if lat == 0.0 and lon == 0.0:
                logger.debug(f"Skipping {iata} (No coordinates found in mapper)")
                continue

            logger.info(f"[{i + 1}/{len(airports)}] Processing {iata}...")

            # Find Station
            station_id = noaa.find_station(lat, lon)
            if not station_id:
                logger.warning(f"  No weather station found for {iata}")
                fail_count += 1
                continue

            # Fetch Data
            # This handles caching internally
            df_weather = noaa.fetch_daily_weather(station_id, start_str, end_str)

            if not df_weather.empty:
                success_count += 1
            else:
                fail_count += 1

            # Respect Rate Limit (client does it, but extra sleep helps)
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"Error processing {iata}: {e}")
            fail_count += 1

    logger.info(
        f"Weather Fetch Complete. Success: {success_count}, Failed/Skipped: {fail_count}"
    )


def main():
    airports, start_date, end_date = get_dataset_scope()

    if not airports:
        logger.error("No airports found to process.")
        return

    logger.info(
        f"Scope: {len(airports)} Airports from {start_date.date()} to {end_date.date()}"
    )

    # 1. Fetch Weather (NOAA)
    fetch_weather_data(airports, start_date, end_date)

    # 2. Fetch Traffic (OpenSky)
    # Disabled per user instruction due to 401 errors.
    logger.info("--- Traffic Fetch (OpenSky) ---")
    logger.info("Skipping OpenSky fetch (Authentication failed). Using defaults.")


if __name__ == "__main__":
    main()
