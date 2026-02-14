from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from src.data.external.opensky_client import OpenSkyClient

# Configure logging to see client output
logging.basicConfig(level=logging.INFO)


def test_integration():
    print("\n=== Testing OpenSky Integration (OAuth) ===")

    try:
        client = OpenSkyClient()

        # 1. Check Token
        print("\n[1] Fetching Access Token...")
        # Force refresh to be sure
        client.access_token = None
        token = client._get_access_token()

        if token:
            print(f"[SUCCESS] Token received! (Length: {len(token)})")
            print(f"   Preview: {token[:10]}...")
        else:
            print("[FAILED] Failed to get token.")
            return

        # 2. Fetch Traffic
        icao = "KJFK"
        # Use 2 days ago to ensure data availability (today's might be partial)
        test_date = datetime.now() - timedelta(days=2)
        test_date = test_date.replace(hour=12, minute=0, second=0, microsecond=0)

        print(f"\n[2] Fetching traffic for {icao} at {test_date}...")
        count = client.fetch_airport_traffic_hour(icao, test_date, "departure")

        print(f"   Departure Count: {count}")

        if count >= 0:
            print("\n[SUCCESS] Integration SUCCESS: API returned valid data.")
        else:
            print("\n[FAILED] Integration FAILED: API Error or No Data returned.")

    except Exception as e:
        print(f"\n[ERROR] Exception during test: {e}")


if __name__ == "__main__":
    test_integration()
