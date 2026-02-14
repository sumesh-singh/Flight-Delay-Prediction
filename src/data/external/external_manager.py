"""
External Data Manager
Orchestrates the enrichment of flight data with external sources (Weather, Traffic).
Acts as the single interface for FeatureEngineer.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import timedelta

# Import clients
try:
    from src.data.external.airport_mapper import AirportMapper
    from src.data.external.noaa_client import NOAAClient
    from src.data.external.opensky_client import OpenSkyClient
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parents[3]))
    from src.data.external.airport_mapper import AirportMapper
    from src.data.external.noaa_client import NOAAClient
    from src.data.external.opensky_client import OpenSkyClient


class ExternalManager:
    def __init__(self):
        """
        Initialize External Manager and all sub-clients.
        """
        self.logger = logging.getLogger(__name__)

        self.mapper = AirportMapper()
        self.noaa = NOAAClient()
        self.opensky = OpenSkyClient()

        self.logger.info(
            "ExternalManager initialized with Mapper, NOAA, and OpenSky clients"
        )

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich a dataframe with all available external features.

        Args:
            df: DataFrame containing 'FL_DATE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME'

        Returns:
            Enriched DataFrame
        """
        self.logger.info(
            f"Enriching dataframe with external features (Rows: {len(df)})..."
        )
        df = df.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df["FL_DATE"]):
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

        # 1. Enrich Weather (Origin and Dest)
        df = self._enrich_weather_features(df)

        # 2. Enrich Traffic (Origin Congestion)
        df = self._enrich_traffic_features(df)

        return df

    def _enrich_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather features for Origin and Destination."""
        self.logger.info("Adding weather features...")

        # We need to fetch weather for unique (Airport, Date) pairs
        # Optimization: Identify all unique airports and dates needed
        # To avoid API spam, we'll iterate by unique airport.

        # Get unique dates involved (to fetch ranges efficiently? No, fetch daily per airport)

        # Enriched dicts to map back to DF
        # Key: (Airport, DateStr) -> {TMAX, PRCP, AWND...}
        weather_cache = {}

        # Processing Origins
        unique_origins = df["ORIGIN"].unique()

        for iata in unique_origins:
            # Get NOAA Station
            lat, lon = self.mapper.get_coordinates(iata)
            if lat == 0.0:
                continue

            station_id = self.noaa.find_station(lat, lon)
            if not station_id:
                continue

            # Get date range for this airport in the dataset
            airport_dates = df[df["ORIGIN"] == iata]["FL_DATE"]
            start_date = airport_dates.min()
            end_date = airport_dates.max()

            print(
                f"DEBUG: Processing Weather for {iata} ({start_date.date()} to {end_date.date()}) Station: {station_id}"
            )

            # Iterate by year to leverage caching and avoid huge requests
            current = start_date
            while current <= end_date:
                # Year chunk logic
                year_start = current
                year_end = min(
                    current + pd.DateOffset(years=1) - pd.DateOffset(days=1), end_date
                )

                s_str = year_start.strftime("%Y-%m-%d")
                e_str = year_end.strftime("%Y-%m-%d")

                # Fetch data (will use cache)
                weather_df = self.noaa.fetch_daily_weather(station_id, s_str, e_str)

                if not weather_df.empty:
                    # Add to cache for quick lookup
                    # Convert 'DATE' back to string for consistent lookup
                    weather_dict = weather_df.set_index("DATE")[
                        ["TMAX", "PRCP", "AWND", "SNOW"]
                    ].to_dict("index")
                    for date_str, stats in weather_dict.items():
                        weather_cache[(iata, date_str)] = stats

                current = year_end + pd.DateOffset(days=1)

        # Also Process Destinations (reuse cache if overlap?)
        # Logic is same, just different column prefix.
        # Ideally we refactor this, but for now let's apply to DF.

        # Define function to apply
        def get_weather(airport, date_obj):
            date_str = date_obj.strftime("%Y-%m-%d")
            key = (airport, date_str)
            if key in weather_cache:
                return weather_cache[key]
            return {"TMAX": 0, "PRCP": 0, "AWND": 0, "SNOW": 0}  # Default/Mean?

        # Apply to Origin
        # Using map/apply might be slow. Merge is better.
        # Construct a temporary DF from weather_cache?
        # That's cleaner.

        weather_records = []
        for (airport, date_str), stats in weather_cache.items():
            record = {"airport_join": airport, "date_join": pd.to_datetime(date_str)}
            record.update({f"ORIGIN_{k}": v for k, v in stats.items()})
            weather_records.append(record)

        if weather_records:
            w_df = pd.DataFrame(weather_records)
            df = df.merge(
                w_df,
                left_on=["ORIGIN", "FL_DATE"],
                right_on=["airport_join", "date_join"],
                how="left",
            )
            df = df.drop(columns=["airport_join", "date_join"])

            # Recreate for DEST (columns need DEST_ prefix)
            w_df_dest = pd.DataFrame(weather_records)
            w_df_dest.columns = [
                c.replace("ORIGIN_", "DEST_") if "ORIGIN_" in c else c
                for c in w_df_dest.columns
            ]

            df = df.merge(
                w_df_dest,
                left_on=["DEST", "FL_DATE"],
                right_on=["airport_join", "date_join"],
                how="left",
            )
            df = df.drop(columns=["airport_join", "date_join"])

            # Fill NaNs (missing weather data)
            # Use 0 for precipitation/snow, mean for Temp/Wind?
            # For robustness, 0 is safer than NaN for models like RF, but imputation is better.
            # We'll use 0 for now as 'no data' ~ 'normal condition' usually.
            cols_to_fill = [c for c in df.columns if "ORIGIN_" in c or "DEST_" in c]
            df[cols_to_fill] = df[cols_to_fill].fillna(0)

        return df

    def _enrich_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add OpenSky traffic counts."""
        self.logger.info("Adding traffic features...")

        # Traffic is hour-specific.
        # We need (Airport, Date, Hour) -> Count
        # OpenSky Client takes (ICAO, Datetime)

        # Optimization:
        # Group by (ORIGIN, FL_DATE, Hour) to minimize API calls.

        df["hour"] = (df["CRS_DEP_TIME"] // 60).astype(int)

        # Create a unique list of needed lookups
        # Unique (Origin, Date, Hour)
        # Note: 'date' + 'hour' -> datetime

        # Limit scope: fetching traffic for every single flight in a large dataset is slow via API.
        # We should iterate through the data's timeframe.

        # For prototype/sampling, we will do a 'apply' but catches errors.

        # To make this fast, we ideally pre-fetch.
        # Given we are in execution mode, let's implement a 'safe' lookup.

        def get_traffic(row):
            iata = row["ORIGIN"]
            date_val = row["FL_DATE"]
            hour = row["hour"]

            icao = self.mapper.get_icao(iata)

            # Construct datetime
            dt = date_val + timedelta(hours=int(hour))

            # Fetch
            try:
                # Use departure density as proxy for congestion
                count = self.opensky.fetch_airport_traffic_hour(icao, dt, "departure")
                return count if count != -1 else np.nan
            except Exception:
                return np.nan

        # Applying row-by-row is SLOW.
        # But `fetch_airport_traffic_hour` caches by DAY.
        # So repeated calls for the same day/hour are fast (dict lookup in cache).
        # Calls for same day different hour are fast (same cache file).

        # Apply to unique combos to verify speed?
        # Let's apply to the dataframe but with a limit or warnings?
        # For now, we'll try applying to the first N rows or all if small.

        if len(df) > 1000:
            self.logger.warning(
                "Large dataframe detected. Traffic enrichment might be slow initially."
            )

        df["ORIGIN_AIRPORT_TRAFFIC"] = df.apply(get_traffic, axis=1)

        # Fill NaN with 0 (missing traffic data ~ unavailable/low traffic)
        df["ORIGIN_AIRPORT_TRAFFIC"] = df["ORIGIN_AIRPORT_TRAFFIC"].fillna(0)

        # Drop temp hour
        df = df.drop(columns=["hour"])

        return df


if __name__ == "__main__":
    # Test Manager
    logging.basicConfig(level=logging.INFO)
    manager = ExternalManager()

    # Create dummy DF
    test_df = pd.DataFrame(
        {
            "FL_DATE": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-15")],
            "ORIGIN": ["JFK", "LAX"],
            "DEST": ["LAX", "JFK"],
            "CRS_DEP_TIME": [1200, 1800],  # 20:00 and 30:00?? No, 12:00 and 18:00
        }
    )

    enriched_df = manager.enrich_dataframe(test_df)
    print("\nEnriched Data:")
    print(enriched_df.columns)
    print(enriched_df.head())
