"""
Data Collection Module
Handles all data acquisition from CSV sources and external APIs

Supports:
- BTS On-Time Performance CSV files (monthly)
- Airport metadata CSVs (OurAirports)
- NOAA Weather API (hourly observations)
- OpenSky Network API (global flight tracking)

All functions save raw data to data/raw/ for downstream processing.
No credentials are hard-coded - all read from config files.
"""

import pandas as pd
import requests
import zipfile
from pathlib import Path
from typing import Optional, List, Dict
import time
import json
from requests_cache import CachedSession
from tqdm import tqdm

# Import configuration
try:
    from config.data_config import (
        RAW_DATA_DIR,
        METADATA_DIR,
        BTS_BASE_URL,
        BTS_FILENAME_TEMPLATE,
        BTS_ESSENTIAL_COLUMNS,
        AIRPORT_METADATA_URLS,
        NOAA_API_BASE_URL,
        NOAA_RATE_LIMIT,
        OPENSKY_API_URL,
        get_api_token,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.data_config import (
        RAW_DATA_DIR,
        METADATA_DIR,
        BTS_BASE_URL,
        BTS_FILENAME_TEMPLATE,
        BTS_ESSENTIAL_COLUMNS,
        AIRPORT_METADATA_URLS,
        NOAA_API_BASE_URL,
        NOAA_RATE_LIMIT,
        OPENSKY_API_URL,
        get_api_token,
    )


class BTSDataCollector:
    """
    Collects Bureau of Transportation Statistics (BTS) On-Time Performance data

    Data Schema (21 essential columns from 109 total):
    - FL_DATE: Flight date (YYYY-MM-DD)
    - OP_CARRIER: Operating carrier code (e.g., 'AA', 'DL')
    - OP_CARRIER_FL_NUM: Flight number
    - ORIGIN: Origin airport code (IATA)
    - DEST: Destination airport code (IATA)
    - CRS_DEP_TIME: Scheduled departure time (HHMM)
    - DEP_TIME: Actual departure time (HHMM)
    - DEP_DELAY: Departure delay in minutes
    - CRS_ARR_TIME: Scheduled arrival time (HHMM)
    - ARR_TIME: Actual arrival time (HHMM)
    - ARR_DELAY: Arrival delay in minutes (TARGET VARIABLE)
    - CANCELLED: Cancellation indicator (1=cancelled)
    - DIVERTED: Diversion indicator (1=diverted)
    - DISTANCE: Flight distance in miles
    - CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY:
      Delay attribution in minutes

    Output Format: CSV files saved to data/raw/
    """

    def __init__(self):
        """Initialize BTS data collector"""
        self.base_url = BTS_BASE_URL
        self.output_dir = RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_monthly_data(
        self, year: int, month: int, timeout: int = 600
    ) -> Optional[Path]:
        """
        Download BTS on-time performance data for a specific month

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            timeout: Request timeout in seconds (default: 10 minutes)

        Returns:
            Path to downloaded CSV file, or None if download failed

        Raises:
            requests.RequestException: If download fails
        """
        filename = BTS_FILENAME_TEMPLATE.format(year=year, month=month)
        url = self.base_url + filename

        zip_path = self.output_dir / filename

        # Check if CSV already exists (search for any matching CSV)
        existing_csvs = list(self.output_dir.glob(f"*_{year}_{month}.csv"))
        if existing_csvs:
            print(f"✓ {existing_csvs[0].name} already exists, skipping download")
            return existing_csvs[0]

        print(f"Downloading BTS data for {year}-{month:02d}...")
        print(f"URL: {url}")

        try:
            # Download with streaming to handle large files
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Save ZIP file with progress bar
            total_size = int(response.headers.get("content-length", 0))
            print(f"File size: {total_size / 1024 / 1024:.1f} MB")

            with open(zip_path, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print("✓ Download complete")

            # Extract ZIP and get actual CSV filename
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Find CSV file in ZIP (don't assume exact filename)
                csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]

                if not csv_files:
                    raise ValueError(f"No CSV file found in {filename}")

                if len(csv_files) > 1:
                    print(f"⚠️  Multiple CSV files found, using: {csv_files[0]}")

                csv_filename = csv_files[0]
                zip_ref.extractall(self.output_dir)

            csv_path = self.output_dir / csv_filename

            # Clean up ZIP file
            zip_path.unlink()
            print(f"✓ Extracted to {csv_path}")

            return csv_path

        except requests.RequestException as e:
            print(f"✗ Download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None
        except (zipfile.BadZipFile, ValueError) as e:
            print(f"✗ Extraction failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None

    def load_csv(
        self, filepath: Path, use_essential_columns: bool = True
    ) -> pd.DataFrame:
        """
        Load BTS CSV file into pandas DataFrame

        Args:
            filepath: Path to CSV file
            use_essential_columns: If True, load only essential columns

        Returns:
            DataFrame with BTS data

        Schema:
            21 essential columns if use_essential_columns=True
            All 109 columns if use_essential_columns=False
        """
        print(f"Loading {filepath.name}...")

        if use_essential_columns:
            df = pd.read_csv(filepath, usecols=BTS_ESSENTIAL_COLUMNS, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)

        print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
        return df


class AirportMetadataCollector:
    """
    Collects airport metadata from OurAirports open dataset

    Data Schema:

    airports.csv:
    - id: Internal OurAirports ID
    - ident: ICAO code (e.g., 'KATL')
    - type: Airport type (large_airport, medium_airport, etc.)
    - name: Airport name
    - latitude_deg, longitude_deg: Coordinates
    - elevation_ft: Elevation in feet
    - continent, iso_country, iso_region: Location
    - municipality: City name
    - iata_code: IATA code (e.g., 'ATL') - USE THIS FOR MATCHING BTS DATA

    runways.csv:
    - airport_ident: ICAO code (links to airports.csv)
    - length_ft, width_ft: Runway dimensions
    - surface: Runway surface type

    countries.csv:
    - code: ISO country code
    - name: Country name

    Output Format: CSV files saved to data/metadata/
    """

    def __init__(self):
        """Initialize airport metadata collector"""
        self.urls = AIRPORT_METADATA_URLS
        self.output_dir = METADATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> Dict[str, Path]:
        """
        Download all airport metadata CSVs

        Returns:
            Dictionary mapping dataset name to file path
            {'airports': Path, 'runways': Path, 'countries': Path}
        """
        results = {}

        for name, url in self.urls.items():
            filepath = self.download_metadata(name, url)
            if filepath:
                results[name] = filepath

        return results

    def download_metadata(self, name: str, url: str) -> Optional[Path]:
        """
        Download a specific metadata CSV

        Args:
            name: Dataset name (e.g., 'airports')
            url: Download URL

        Returns:
            Path to downloaded CSV file, or None if failed
        """
        output_path = self.output_dir / f"{name}.csv"

        # Check if already exists
        if output_path.exists():
            print(f"✓ {name}.csv already exists, skipping download")
            return output_path

        print(f"Downloading {name} metadata from OurAirports...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"✓ Downloaded {name}.csv")
            return output_path

        except requests.RequestException as e:
            print(f"✗ Download failed for {name}: {e}")
            return None

    def load_airports(self) -> pd.DataFrame:
        """
        Load airports.csv into DataFrame

        Returns:
            DataFrame with airport metadata
        """
        filepath = self.output_dir / "airports.csv"
        if not filepath.exists():
            raise FileNotFoundError(
                f"airports.csv not found at {filepath}. Run download_all() first."
            )

        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} airports")
        return df


class NOAAWeatherCollector:
    """
    Collects weather data from NOAA Climate Data Online (CDO) API

    API Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2

    Data Schema:
    - date: Observation date/time (ISO format)
    - station: Weather station ID
    - datatype: Type of observation (TMAX, TMIN, PRCP, SNOW, etc.)
    - value: Observation value
    - attributes: Data quality flags

    Common datatypes:
    - TMAX: Maximum temperature
    - TMIN: Minimum temperature
    - PRCP: Precipitation
    - SNOW: Snowfall
    - AWND: Average wind speed
    - TAVG: Average temperature

    Rate Limit: 1000 requests per day (enforced via requests-cache)

    Output Format: JSON files saved to data/raw/weather/
    """

    def __init__(self):
        """Initialize NOAA weather collector with API token"""
        self.base_url = NOAA_API_BASE_URL
        self.output_dir = RAW_DATA_DIR / "weather"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get API token from credentials file
        try:
            self.token = get_api_token("noaa")
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️  Warning: {e}")
            print("NOAA API calls will fail without valid token")
            self.token = None

        # Set up cached session for rate limiting
        # Cache expires after 24 hours
        self.session = CachedSession(
            cache_name="noaa_cache",
            backend="sqlite",
            expire_after=86400,  # 24 hours in seconds
        )
        self.session.headers.update({"token": self.token} if self.token else {})

        self.request_count = 0
        self.max_requests_per_day = NOAA_RATE_LIMIT

    def fetch_weather_data(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        datatypes: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Fetch weather data for a specific station and time range

        Args:
            station_id: NOAA station ID (e.g., 'GHCND:USW00013874' for ATL)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            datatypes: List of data types to fetch (e.g., ['TMAX', 'PRCP'])
                      If None, fetches all available

        Returns:
            Dictionary with weather data, or None if request failed

        Rate Limiting:
            Uses requests-cache to avoid exceeding 1000 requests/day limit
            Cached responses don't count toward rate limit
        """
        if not self.token:
            print("✗ Cannot fetch weather data: No API token configured")
            return None

        # Check rate limit (cached requests don't count)
        if self.request_count >= self.max_requests_per_day:
            print(f"✗ Rate limit reached ({self.max_requests_per_day} requests/day)")
            return None

        url = f"{self.base_url}data"

        params = {
            "datasetid": "GHCND",  # Global Historical Climatology Network-Daily
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "units": "standard",  # Use standard units
            "limit": 1000,  # Max results per request
        }

        if datatypes:
            params["datatypeid"] = ",".join(datatypes)

        print(f"Fetching weather data for {station_id} ({start_date} to {end_date})...")

        try:
            response = self.session.get(url, params=params, timeout=30)

            # Check if response was from cache
            if not getattr(response, "from_cache", False):
                self.request_count += 1
                print(
                    f"  API calls today: {self.request_count}/{self.max_requests_per_day}"
                )
            else:
                print("  ✓ Retrieved from cache")

            response.raise_for_status()
            data = response.json()

            if "results" in data:
                print(f"✓ Fetched {len(data['results'])} weather observations")

                # Save to file
                filename = f"weather_{station_id.replace(':', '_')}_{start_date}_{end_date}.json"
                filepath = self.output_dir / filename

                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"✓ Saved to {filepath}")
                return data
            else:
                print("⚠️  No weather data found for this query")
                return None

        except requests.RequestException as e:
            print(f"✗ Weather data request failed: {e}")
            return None


class OpenSkyFlightCollector:
    """
    Collects flight tracking data from OpenSky Network API

    API Documentation: https://opensky-network.org/apidoc/

    Data Schema (State Vectors):
    - icao24: Unique aircraft identifier
    - callsign: Flight callsign
    - origin_country: Country of origin
    - time_position: Unix timestamp of position update
    - last_contact: Unix timestamp of last contact
    - longitude, latitude: Position coordinates
    - baro_altitude: Barometric altitude in meters
    - on_ground: Boolean, true if aircraft on ground
    - velocity: Ground speed in m/s
    - true_track: Heading in degrees
    - vertical_rate: Vertical rate in m/s

    Rate Limit:
    - Anonymous: 400 credits/day (1 credit per request)
    - Registered: 4000 credits/day

    Output Format: JSON files saved to data/raw/flights/
    """

    def __init__(self):
        """Initialize OpenSky flight collector with credentials"""
        self.base_url = OPENSKY_API_URL
        self.output_dir = RAW_DATA_DIR / "flights"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get credentials from file
        try:
            credentials = get_api_token("opensky")
            # Format: "username:password"
            if ":" in credentials:
                self.username, self.password = credentials.split(":", 1)
            else:
                print("⚠️  Warning: Invalid OpenSky credentials format")
                self.username = self.password = None
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️  Warning: {e}")
            print("OpenSky API will use anonymous access (limited)")
            self.username = self.password = None

        # Set up session
        self.session = requests.Session()
        if self.username and self.password:
            self.session.auth = (self.username, self.password)

    def fetch_flights_by_bbox(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        timestamp: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Fetch all flights within a bounding box

        Args:
            min_lat: Minimum latitude (degrees)
            max_lat: Maximum latitude (degrees)
            min_lon: Minimum longitude (degrees)
            max_lon: Maximum longitude (degrees)
            timestamp: Unix timestamp for historical data (optional)
                      If None, fetches current state

        Returns:
            Dictionary with flight state vectors, or None if failed

        Example:
            # Georgia (ATL airport area)
            fetch_flights_by_bbox(33.0, 34.5, -85.0, -83.5)
        """
        url = f"{self.base_url}/api/states/all"

        params = {
            "lamin": min_lat,
            "lamax": max_lat,
            "lomin": min_lon,
            "lomax": max_lon,
        }

        if timestamp:
            params["time"] = timestamp

        print(
            f"Fetching flights in bbox [{min_lat},{max_lat}], [{min_lon},{max_lon}]..."
        )

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data and "states" in data and data["states"]:
                num_flights = len(data["states"])
                print(f"✓ Fetched {num_flights} flights")

                # Save to file
                timestamp_str = str(timestamp) if timestamp else str(int(time.time()))
                filename = f"flights_bbox_{timestamp_str}.json"
                filepath = self.output_dir / filename

                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"✓ Saved to {filepath}")
                return data
            else:
                print("⚠️  No flights found in this area")
                return None

        except requests.RequestException as e:
            print(f"✗ Flight data request failed: {e}")
            # Check if response exists and has status code
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    print("  ⚠️  Rate limit exceeded. Wait before retrying.")
            return None

    def fetch_flights_by_airport(
        self, airport_icao: str, begin_time: int, end_time: int, arrival: bool = True
    ) -> Optional[List[Dict]]:
        """
        Fetch arrivals or departures for a specific airport

        NOTE: This endpoint requires OpenSky membership ($)

        Args:
            airport_icao: ICAO airport code (e.g., 'KATL')
            begin_time: Begin timestamp (Unix)
            end_time: End timestamp (Unix)
            arrival: If True, fetch arrivals; if False, fetch departures

        Returns:
            List of flight dictionaries, or None if failed
        """
        endpoint = "arrival" if arrival else "departure"
        url = f"{self.base_url}/api/flights/{endpoint}"

        params = {"airport": airport_icao, "begin": begin_time, "end": end_time}

        flight_type = "arrivals" if arrival else "departures"
        print(f"Fetching {flight_type} for {airport_icao}...")

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            flights = response.json()

            if flights:
                print(f"✓ Fetched {len(flights)} {flight_type}")

                # Save to file
                filename = f"{flight_type}_{airport_icao}_{begin_time}_{end_time}.json"
                filepath = self.output_dir / filename

                with open(filepath, "w") as f:
                    json.dump(flights, f, indent=2)

                print(f"✓ Saved to {filepath}")
                return flights
            else:
                print(f"⚠️  No {flight_type} found")
                return None

        except requests.RequestException as e:
            print(f"✗ {flight_type.capitalize()} request failed: {e}")
            return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def collect_bts_data(year: int, month: int) -> Optional[pd.DataFrame]:
    """
    Convenience function to download and load BTS data for a specific month

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)

    Returns:
        DataFrame with BTS data, or None if collection failed
    """
    collector = BTSDataCollector()
    filepath = collector.download_monthly_data(year, month)

    if filepath:
        return collector.load_csv(filepath)
    return None


def collect_airport_metadata() -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download and load all airport metadata

    Returns:
        Dictionary with DataFrames: {'airports': df, 'runways': df, 'countries': df}
    """
    collector = AirportMetadataCollector()
    collector.download_all()

    results = {}
    for name in ["airports", "runways", "countries"]:
        filepath = METADATA_DIR / f"{name}.csv"
        if filepath.exists():
            results[name] = pd.read_csv(filepath)
            print(f"✓ Loaded {len(results[name]):,} records from {name}.csv")

    return results


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("=" * 60)
    print("DATA COLLECTION MODULE TEST")
    print("=" * 60)

    # Test 1: Airport metadata
    print("\n1. Testing airport metadata collection...")
    airport_collector = AirportMetadataCollector()
    metadata = airport_collector.download_all()
    print(f"Downloaded {len(metadata)} metadata files")

    # Test 2: BTS data (small test - January 2024)
    print("\n2. Testing BTS data collection...")
    bts_collector = BTSDataCollector()
    # Uncomment to test download (large file!)
    # df = collect_bts_data(2024, 1)
    # if df is not None:
    #     print(f"Loaded {len(df):,} flight records")

    print("\n" + "=" * 60)
    print("Test complete. Verify files in data/raw/ and data/metadata/")
    print("=" * 60)
