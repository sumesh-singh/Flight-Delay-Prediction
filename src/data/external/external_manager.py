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
                weather_df = self.noaa.fetch_daily_weather(
                    station_id, s_str, e_str, local_only=True
                )

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
        """Add OpenSky traffic counts (Vectorized Merge)."""
        self.logger.info("Adding traffic features...")

        # 1. Round CRS_DEP_TIME to nearest hour to match traffic data granularity
        # CRS_DEP_TIME is HHMM int. Convert to hours approx.
        # 1230 -> 12.5 -> 12
        # Traffic data is hourly.

        # Helper to get datetime hour from FL_DATE + CRS_DEP_TIME
        # CRS_DEP_TIME is 0-2359.
        hours = (df["CRS_DEP_TIME"] // 100).astype(int)  # 1230 // 100 = 12

        # 2. Identify unique (ICAO, Date) keys needed
        # We need to map ORIGIN (IATA) to ICAO first
        # Doing this per-row is slow. Do we have the ICAO column?
        # If not, let's map unique airports first.

        unique_origins = df["ORIGIN"].unique()
        iata_to_icao = {iata: self.mapper.get_icao(iata) for iata in unique_origins}

        # Create a temp column for ICAO
        # Use map which is faster than apply
        df["origin_icao"] = df["ORIGIN"].map(iata_to_icao)

        # We need traffic for these dates
        unique_dates = df["FL_DATE"].dt.date.unique()

        # 3. Build a local traffic cache for this chunk
        # We will load traffic data for ALL unique identifiers in this chunk into one DF
        # Then merge.

        traffic_records = []

        # Iterate unique dates (usually 1 month = 30 days) and unique airports (size N)
        # This is essentially: Load Cache Files involved in this chunk.

        # Optimization: Group by ICAO first? No, Cache is by file per Day per Airport (or per Day?)
        # ExternalManager client says: cache_dir / f"{icao_code}_{date_str}_{mode}.parquet"
        # So it is per AIRPORT per DAY.

        # This means for 300 airports * 30 days = 9000 file reads.
        # That's a lot for one chunk.
        # However, `opensky_client` might be slow if we call fetch.
        # We assume data is cached.

        # Let's try to be smart.
        # Identify unique (ICAO, Date) pairs actually present in data.
        # Group by ICAO, FL_DATE to get the list of needed files.

        needed_pairs = df[["origin_icao", "FL_DATE"]].drop_duplicates()

        # Limit checking: If we have > 5000 pairs, this might be slow loop in Python.
        # But file IO is the bottleneck.

        for _, row in needed_pairs.iterrows():
            icao = row["origin_icao"]
            date_val = row["FL_DATE"]
            date_str = date_val.strftime("%Y-%m-%d")

            # Use Client to fetch (it handles caching)
            # But the client `fetch_airport_traffic_hour` returns a single int.
            # We want the whole day if we are loading the file anyway.

            # Access client cache directly?
            # Better to use a method that returns the daily series.

            # Let's bypass the client's single-value getter and implement a bulk loader here
            # or add a bulk loader to client.
            # For now, implemented here to modify Manager only.

            cache_file = self.opensky.cache_dir / f"{icao}_{date_str}_departure.parquet"
            if cache_file.exists():
                try:
                    daily_df = pd.read_parquet(cache_file)
                    # daily_df has index 'datetime' (hourly), column 'count'

                    # We need to join on Hour.
                    # Convert index to Hour integer?
                    # Index is timestamp 2024-01-01 00:00:00, etc.

                    # Store as records: (ICAO, Date, Hour) -> Count
                    for ts, count_row in daily_df.iterrows():
                        # ts is Timestamp
                        h = ts.hour
                        traffic_records.append(
                            {
                                "origin_icao": icao,
                                "date_val": date_val,
                                "hour_val": h,
                                "traffic_count": count_row["count"],
                            }
                        )

                except Exception as e:
                    pass  # Corrupt file or other issue

            # If not exists, we skip (0 traffic).
            # We do NOT call API here to avoid 1M API calls in a loop.

        if not traffic_records:
            # excessive loop or no data
            df["ORIGIN_AIRPORT_TRAFFIC"] = 0
            df = df.drop(columns=["origin_icao"], errors="ignore")
            return df

        # 4. Create Traffic Lookup DF
        traffic_df = pd.DataFrame(traffic_records)

        # 5. Merge
        # We need to merge df with traffic_df on [origin_icao, FL_DATE, hour]
        # We created 'hours' series earlier. Add it to df temporarily.
        df["hour_temp"] = hours

        df = df.merge(
            traffic_df,
            left_on=["origin_icao", "FL_DATE", "hour_temp"],
            right_on=["origin_icao", "date_val", "hour_val"],
            how="left",
        )

        # 6. Cleanup
        df["ORIGIN_AIRPORT_TRAFFIC"] = df["traffic_count"].fillna(0).astype("float32")

        drop_cols = [
            "origin_icao",
            "hour_temp",
            "date_val",
            "hour_val",
            "traffic_count",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

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
