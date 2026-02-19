"""
OpenSky Client
Fetches historical flight traffic data (Arrivals/Departures) from OpenSky Network API.
Handles rate limiting, authentication, and local caching.
"""

import time
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from tqdm import tqdm

try:
    from config.api_config import OPENSKY_CREDENTIALS
    from config.data_config import EXTERNAL_DATA_DIR, REQUIRED_AIRPORTS
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parents[3]))
    from config.api_config import OPENSKY_CREDENTIALS
    from config.data_config import EXTERNAL_DATA_DIR, REQUIRED_AIRPORTS


class OpenSkyRateLimitError(Exception):
    """Raised when OpenSky API rate limit is exceeded."""

    pass


class OpenSkyClient:
    BASE_URL = "https://opensky-network.org/api"
    TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize OpenSky Client with credentials and cache directory.
        """
        # Store raw credentials for OAuth generation
        self.username = OPENSKY_CREDENTIALS["username"]
        self.password = OPENSKY_CREDENTIALS["password"]
        self.access_token = None
        self.token_expiry = 0

        if cache_dir is None:
            self.cache_dir = EXTERNAL_DATA_DIR / "traffic-real"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_access_token(self) -> Optional[str]:
        """
        Fetch or refresh OAuth2 access token.
        """
        # Check if valid token exists (buffer 60s)
        if self.access_token and time.time() < self.token_expiry - 60:
            return self.access_token

        try:
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.username,
                "client_secret": self.password,
            }
            response = requests.post(self.TOKEN_URL, data=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                # Expires in ~1800s usually
                self.token_expiry = time.time() + data.get("expires_in", 1800)
                print("Refreshed OpenSky Access Token.")
                return self.access_token
            else:
                print(f"Failed to get token: {response.text}")
                return None
        except Exception as e:
            print(f"Token Fetch Error: {e}")
            return None

    def _make_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Make a request to OpenSky API with Bearer Token.
        """
        url = self.BASE_URL + endpoint

        token = self._get_access_token()
        if not token:
            print("No valid access token available. Skipping request.")
            return []

        headers = {"Authorization": f"Bearer {token}"}

        try:
            # OpenSky limits: 1000 req/day equivalent
            time.sleep(1.0)

            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 429:
                print("Rate limit exceeded. Stopping execution to respect limits.")
                raise OpenSkyRateLimitError("Rate limit exceeded")

            # Handle Token Expiry / 401 gracefully
            if response.status_code == 401:
                print(
                    "401 Unauthorized with Token. It might be expired or scope is wrong."
                )
                return None  # Return None to indicate failure

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None  # Return None to indicate failure

    def fetch_airport_traffic_hour(
        self, icao_code: str, date_hour: datetime, mode: str = "departure"
    ) -> int:
        """
        Fetch number of flights for a specific airport in a 1-hour window.
        Uses caching to avoid redundant API calls.

        Args:
            icao_code: ICAO airport code (e.g. 'KJFK')
            date_hour: datetime object representing the start of the hour (e.g. 2024-01-01 10:00:00)
            mode: 'departure' or 'arrival'

        Returns:
            Count of flights, or -1 (or NaN logic upstream) if failed.
        """
        # Define cache key (daily cache file per airport)
        date_str = date_hour.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"{icao_code}_{date_str}_{mode}.parquet"

        # Check if we have this hour in cache
        if cache_file.exists():
            df_cache = pd.read_parquet(cache_file)
            # Ensure index is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_cache.index):
                df_cache.index = pd.to_datetime(df_cache.index)

            if date_hour in df_cache.index:
                # We can print a small debug message or just return silently
                # print(f"  [Cache Hit] {icao_code}")
                return df_cache.loc[date_hour, "count"]

        # If not cached, we need to fetch the whole day (to be efficient)
        # API allows fetching a time interval. Max 7 days for arrivals/departures.
        # We'll fetch 1 day at a time to keep cache manageable.

        start_ts = int(date_hour.replace(hour=0, minute=0, second=0).timestamp())
        end_ts = int(date_hour.replace(hour=23, minute=59, second=59).timestamp())

        endpoint = f"/flights/{mode}"
        params = {"airport": icao_code, "begin": start_ts, "end": end_ts}

        # print(f"Fetching {mode}s for {icao_code} on {date_str}...")
        flights = self._make_request(endpoint, params)

        if flights is None:
            # Request failed (404, 401, or other error), do NOT cache
            return -1

        # Process and cache the day's data
        # We need to count flights per hour
        hourly_counts = {}
        # Initialize all hours to 0
        current = date_hour.replace(hour=0, minute=0, second=0)
        for _ in range(24):
            hourly_counts[current] = 0
            current += timedelta(hours=1)

        if flights:
            for flight in flights:
                # Use 'firstSeen' for departure, 'lastSeen' for arrival?
                # Actually OpenSky returns estimated times.
                # For departure endpoint: 'firstSeen' is roughly departure time.
                # For arrival endpoint: 'lastSeen' is roughly arrival time.
                ts = (
                    flight.get("firstSeen")
                    if mode == "departure"
                    else flight.get("lastSeen")
                )
                if ts:
                    dt = datetime.fromtimestamp(ts).replace(
                        minute=0, second=0, microsecond=0
                    )
                    if dt in hourly_counts:
                        hourly_counts[dt] += 1

        # Save to parquet
        df_new = pd.DataFrame.from_dict(
            hourly_counts, orient="index", columns=["count"]
        )
        df_new.index.name = "datetime"
        df_new.to_parquet(cache_file)

        # Return specific hour
        return hourly_counts.get(date_hour, 0)

    def fetch_all_required_stations(
        self, date_hour: datetime, mode: str = "departure"
    ) -> Dict[str, int]:
        """
        Fetch traffic data for all required airports defined in config.
        """
        results = {}
        print(f"Fetching {mode} data for {len(REQUIRED_AIRPORTS)} required stations...")

        # Use tqdm for progress bar if available, else standard iterator
        iterator = tqdm(REQUIRED_AIRPORTS, desc="Fetching stations")

        for icao in iterator:
            try:
                if len(icao) != 4:
                    continue

                count = self.fetch_airport_traffic_hour(icao, date_hour, mode)
                if count >= 0:
                    results[icao] = count
            except OpenSkyRateLimitError:
                print(f"\n🛑 Rate limit reached while fetching {icao}.")
                print(
                    "Progress has been saved via cache. Run the script again later to resume."
                )
                break
            except Exception as e:
                print(f"Error fetching {icao}: {e}")
                results[icao] = -1

        return results

    def get_traffic_features(
        self, icao_code: str, flight_time: datetime
    ) -> Dict[str, float]:
        """
        Get traffic features for a flight (Departure density, Arrival density).

        Args:
            icao_code: Origin Airport ICAO
            flight_time: Scheduled Departure Time

        Returns:
            Dict with 'ORIGIN_DEPARTURES_HOUR'
        """
        # Round to nearest hour
        hour_start = flight_time.replace(minute=0, second=0, microsecond=0)

        deps = self.fetch_airport_traffic_hour(icao_code, hour_start, "departure")

        # We could also get arrivals at that airport (congestion)
        # arrs = self.fetch_airport_traffic_hour(icao_code, hour_start, "arrival")

        return {"ORIGIN_AIRPORT_TRAFFIC": float(deps) if deps >= 0 else float("nan")}


if __name__ == "__main__":
    client = OpenSkyClient()

    # Test date (5 days ago)
    test_date = datetime.now() - timedelta(days=5)
    test_date = test_date.replace(hour=12, minute=0, second=0, microsecond=0)

    # Check if we have required airports
    if REQUIRED_AIRPORTS:
        print(f"Found {len(REQUIRED_AIRPORTS)} required airports.")
        # Fetch for all
        traffic_data = client.fetch_all_required_stations(test_date)
        print(f"Fetched data for {len(traffic_data)} airports.")

        # Show stats
        successful = sum(1 for v in traffic_data.values() if v >= 0)
        print(f"Successfully fetched {successful}/{len(traffic_data)} stations.")

        # Show sample
        sample_keys = list(traffic_data.keys())[:5]
        for k in sample_keys:
            print(f"{k}: {traffic_data[k]}")
    else:
        print("No required airports found in config. Testing with KJFK.")
        count = client.fetch_airport_traffic_hour("KJFK", test_date)
        print(f"JFK traffic: {count}")
