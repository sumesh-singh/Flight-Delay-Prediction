"""
NOAA Client
Fetches historical daily weather data (GHCND) from NOAA CDO API.
Handles rate limiting and local caching.
"""

import time
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import config (handle standalone execution)
try:
    from config.api_config import NOAA_ACCESS_TOKEN
    from config.data_config import EXTERNAL_DATA_DIR
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parents[3]))
    from config.api_config import NOAA_ACCESS_TOKEN
    from config.data_config import EXTERNAL_DATA_DIR


class NOAAClient:
    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize NOAA Client with API token and cache directory.
        """
        self.headers = {"token": NOAA_ACCESS_TOKEN}

        if cache_dir is None:
            self.cache_dir = EXTERNAL_DATA_DIR / "weather"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to NOAA API with rate limiting and error handling.
        """
        url = self.BASE_URL + endpoint
        try:
            time.sleep(0.25)  # Max 5 requests per second (conservative 4/s)
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            try:
                if "response" in locals():
                    print(f"Response Body: {response.text}")
            except:
                pass
            return {}

    def fetch_daily_weather(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        local_only: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch daily summaries (GHCND) for a specific station and date range.
        Checks local cache first.

        Args:
            station_id: NOAA Station ID (e.g., 'GHCND:USW00094789' for JFK)
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            local_only: If True, do not make API calls if cache is missing.

        Returns:
            DataFrame with columns: DATE, TMAX, TMIN, PRCP, SNOW, AWND
        """
        cache_file = (
            self.cache_dir
            / f"{station_id.replace(':', '_')}_{start_date}_{end_date}.parquet"
        )

        # Check cache
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        # Break down large date ranges into yearly chunks
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_dfs = []
        current_start = start_dt

        while current_start <= end_dt:
            # Calculate chunk end (1 year later or final end date)
            current_end = min(
                current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end_dt
            )

            chunk_start_str = current_start.strftime("%Y-%m-%d")
            chunk_end_str = current_end.strftime("%Y-%m-%d")

            # Check chunk cache
            chunk_file = (
                self.cache_dir
                / f"{station_id.replace(':', '_')}_{chunk_start_str}_{chunk_end_str}.parquet"
            )

            if chunk_file.exists():
                try:
                    df = pd.read_parquet(chunk_file)
                    all_dfs.append(df)
                    current_start = current_end + pd.DateOffset(days=1)
                    continue
                except:
                    pass

            if local_only:
                # Skip API call
                current_start = current_end + pd.DateOffset(days=1)
                continue

            # Fetch
            print(
                f"Fetching weather for {station_id} ({chunk_start_str} to {chunk_end_str})..."
            )

            params = {
                "datasetid": "GHCND",  # Global Historical Climatology Network - Daily
                "stationid": station_id,
                "startdate": chunk_start_str,
                "enddate": chunk_end_str,
                "limit": 1000,
                "units": "metric",  # Celcius, Millimeters
                "datatypeid": ["TMAX", "TMIN", "PRCP", "SNOW", "AWND"],
            }

            data = self._make_request("data", params)

            if data and "results" in data:
                # Parse results
                records = []
                for result in data["results"]:
                    records.append(
                        {
                            "DATE": result["date"].split("T")[0],
                            "datatype": result["datatype"],
                            "value": result["value"],
                        }
                    )
                if records:
                    all_dfs.append(pd.DataFrame(records))
            elif not data or "results" not in data:
                print(f"No data found for {station_id} in this chunk")

            # Move to next chunk
            current_start = current_end + pd.DateOffset(days=1)

        if not all_dfs:
            return pd.DataFrame()

        df_long = pd.concat(all_dfs, ignore_index=True)

        if df_long.empty:
            return pd.DataFrame()

        # Pivot to wide format (one row per date)
        # Drop duplicates just in case
        df_long = df_long.drop_duplicates(subset=["DATE", "datatype"])

        df_wide = df_long.pivot(
            index="DATE", columns="datatype", values="value"
        ).reset_index()

        # Ensure all expected columns exist
        expected_cols = ["TMAX", "TMIN", "PRCP", "SNOW", "AWND"]
        for col in expected_cols:
            if col not in df_wide.columns:
                df_wide[col] = 0.0  # Fill missing metrics with 0 (e.g. no snow)

        # Fill missing values with 0
        df_wide = df_wide.fillna(0.0)

        # Save to cache
        df_wide.to_parquet(cache_file)

        return df_wide

    def find_station(self, latitude: float, longitude: float) -> str:
        """
        Find the nearest GHCND station to a lat/lon coordinate.
        """
        params = {
            "datasetid": "GHCND",
            "sortfield": "mindate",
            "sortorder": "desc",
            "limit": 5,
            "extent": f"{latitude - 0.2},{longitude - 0.2},{latitude + 0.2},{longitude + 0.2}",  # ~20km box
        }

        results = self._make_request("stations", params)

        if results and "results" in results:
            # Return the first one (usually nearest/best)
            return results["results"][0]["id"]

        return ""


if __name__ == "__main__":
    # Test Client
    client = NOAAClient()

    # Test 1: Find Station for JFK (40.6413, -73.7781)
    print("Finding station for JFK...")
    station_id = client.find_station(40.6413, -73.7781)
    print(f"Found Station: {station_id}")

    if station_id:
        # Test 2: Fetch Data for Jan 2024
        df = client.fetch_daily_weather(station_id, "2024-01-01", "2024-01-31")
        print("\nWeather Data Sample:")
        print(df.head())
        print(f"\nCached to: {client.cache_dir}")
