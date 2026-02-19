"""
Airport Mapper
Maps IATA codes (from BTS data) to ICAO codes (for OpenSky) and Lat/Lon (for NOAA).
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

# Try to import from config, fallback for standalone testing
try:
    from config.data_config import EXTERNAL_DATA_DIR
except ImportError:
    EXTERNAL_DATA_DIR = Path(__file__).parents[3] / "data" / "external"


class AirportMapper:
    def __init__(self, coordinates_path: Optional[Path] = None):
        """
        Initialize AirportMapper with path to coordinates CSV.
        """
        if coordinates_path is None:
            self.coordinates_path = EXTERNAL_DATA_DIR / "airport_coordinates.csv"
        else:
            self.coordinates_path = coordinates_path

        self.mapping_df = self._load_mapping()

        # Create lookup dictionaries for faster access
        self.iata_to_icao = self.mapping_df.set_index("IATA")["ICAO"].to_dict()
        self.iata_to_coords = self.mapping_df.set_index("IATA")[
            ["Latitude", "Longitude"]
        ].to_dict("index")

        self._warned = set()

    def _load_mapping(self) -> pd.DataFrame:
        """Load and validate airport coordinates."""
        # Priority 1: OurAirports metadata (from data/metadata/airports.csv)
        # This has accurate coordinates and ICAO/IATA codes
        metadata_path = EXTERNAL_DATA_DIR.parents[1] / "metadata" / "airports.csv"

        if metadata_path.exists():
            try:
                # OurAirports CSV columns: id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,gps_code,iata_code,local_code,home_link,wikipedia_link,keywords
                df = pd.read_csv(metadata_path)
                # Filter for airports with IATA codes
                df = df[df["iata_code"].notna()].copy()

                # Rename to standard internal format
                df = df.rename(
                    columns={
                        "iata_code": "IATA",
                        "ident": "ICAO",
                        "latitude_deg": "Latitude",
                        "longitude_deg": "Longitude",
                    }
                )
                return df[["IATA", "ICAO", "Latitude", "Longitude"]]
            except Exception as e:
                print(f"Warning: Failed to load metadata from {metadata_path}: {e}")

        # Priority 2: Fallback to local simplified CSV
        if not self.coordinates_path.exists():
            print(f"Warning: Mapping file not found at {self.coordinates_path}")
            return pd.DataFrame(columns=["IATA", "ICAO", "Latitude", "Longitude"])

        return pd.read_csv(self.coordinates_path)

    def get_icao(self, iata_code: str) -> str:
        """
        Get ICAO code for a given IATA code.
        Fallback: 'K' + IATA (standard for US CONUS), 'P' + IATA (Hawaii), etc.
        """
        if iata_code in self.iata_to_icao:
            return self.iata_to_icao[iata_code]

        # Fallback logic for US airports
        if iata_code.startswith("H") or iata_code in ["HNL", "OGG", "KOA", "LIH"]:
            return "P" + iata_code
        elif iata_code.startswith("A") or iata_code in ["ANC", "FAI"]:  # Alaska
            return "P" + iata_code
        else:
            return "K" + iata_code

    def get_coordinates(self, iata_code: str) -> Tuple[float, float]:
        """
        Get (Latitude, Longitude) for a given IATA code.
        Returns (0.0, 0.0) if not found.
        """
        if iata_code in self.iata_to_coords:
            coords = self.iata_to_coords[iata_code]
            return coords["Latitude"], coords["Longitude"]

        if iata_code not in self._warned:
            print(f"Warning: No coordinates found for {iata_code}")
            self._warned.add(iata_code)
        return 0.0, 0.0


if __name__ == "__main__":
    # Test the mapper
    mapper = AirportMapper()

    test_airports = ["JFK", "LAX", "ORD", "HNL", "XYZ"]

    print("Testing Airport Mapper:")
    print(f"loaded {len(mapper.mapping_df)} airports")

    for iata in test_airports:
        icao = mapper.get_icao(iata)
        lat, lon = mapper.get_coordinates(iata)
        print(f"{iata} -> ICAO: {icao}, Coords: ({lat}, {lon})")
