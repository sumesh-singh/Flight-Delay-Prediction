"""
Multi-Year Data Loader for 2023-2025 BTS Flight Data

Handles loading and combining data from multiple years and months.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Generator
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class MultiYearDataLoader:
    """Loads and combines flight data across multiple years."""

    def __init__(self, data_root: Path = None):
        """
        Initialize loader.

        Args:
            data_root: Path to data/raw directory
        """
        if data_root is None:
            data_root = Path(__file__).parent.parent.parent / "data" / "raw"

        self.data_root = Path(data_root)

    def get_available_files(self, year: int) -> List[Path]:
        """Get all available CSV files for a given year."""
        year_dir = self.data_root / str(year)

        if not year_dir.exists():
            return []

        # Exclude readme.html
        csv_files = sorted(
            [f for f in year_dir.glob("*.csv") if "readme" not in f.name.lower()]
        )

        return csv_files

    def load_month(
        self, year: int, month: int, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific month.

        Args:
            year: Year (2023, 2024, or 2025)
            month: Month (1-12)
            sample_size: If provided, only load this many rows

        Returns:
            DataFrame with flight data
        """
        filename = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"
        filepath = self.data_root / str(year) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if sample_size:
            df = pd.read_csv(filepath, nrows=sample_size, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)

        return df

    def load_multi_year_data(
        self,
        years: List[int],
        months: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Load data from multiple years and months.

        Args:
            years: List of years to load (e.g., [2023, 2024])
            months: List of months to load (1-12). If None, loads all available
            sample_size: If provided, samples this many rows from EACH file
            verbose: Print loading progress

        Returns:
            Concatenated DataFrame
        """
        data_frames = []
        total_rows = 0

        for year in years:
            year_dir = self.data_root / str(year)

            if not year_dir.exists():
                if verbose:
                    print(f"[WARN] Year {year} directory not found, skipping")
                continue

            # Get files to load
            if months is None:
                csv_files = self.get_available_files(year)
            else:
                csv_files = []
                for month in months:
                    filename = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"
                    filepath = year_dir / filename
                    if filepath.exists():
                        csv_files.append(filepath)

            # Load each file
            for csv_file in csv_files:
                if sample_size:
                    df = pd.read_csv(csv_file, nrows=sample_size, low_memory=False)
                else:
                    df = pd.read_csv(csv_file, low_memory=False)

                data_frames.append(df)
                total_rows += len(df)

                if verbose:
                    print(f"[OK] {csv_file.name}: {len(df):,} rows")

        if not data_frames:
            raise ValueError("No data loaded. Check years and months.")

        combined_df = pd.concat(data_frames, ignore_index=True)

        if verbose:
            print(
                f"\n[SUMMARY] Loaded {len(data_frames)} files, {total_rows:,} total rows"
            )

        return combined_df

    def load_in_chunks(
        self, years: List[int], chunk_size: int = 100000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load data in chunks to avoid memory overflow.

        Args:
            years: List of years to load
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        for year in years:
            csv_files = self.get_available_files(year)

            for csv_file in csv_files:
                print(f"[CHUNK] Loading {csv_file.name}...")

                for chunk in pd.read_csv(
                    csv_file, chunksize=chunk_size, low_memory=False
                ):
                    yield chunk


def load_train_val_test_split(
    sample_size: Optional[int] = None, verbose: bool = True
) -> tuple:
    """
    Load data with Strategy 2: Year-Based Split.

    Training:   2023 Jan-Dec + 2024 Jan-Dec (24 months)
    Test:       2025 Jan-Nov (11 months)

    Args:
        sample_size: If provided, samples from each file
        verbose: Print loading progress

    Returns:
        (train_df, test_df)
    """
    loader = MultiYearDataLoader()

    if verbose:
        print("=" * 70)
        print("LOADING MULTI-YEAR DATA - STRATEGY 2")
        print("=" * 70)
        print("\nTraining: 2023-2024 (24 months)")

    # Training: All of 2023 and 2024
    train_df = loader.load_multi_year_data(
        years=[2023, 2024],
        months=None,  # Load all available months
        sample_size=sample_size,
        verbose=verbose,
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print("Test: 2025 Jan-Nov (11 months)")

    # Test: 2025 Jan-Nov (Dec not available)
    test_df = loader.load_multi_year_data(
        years=[2025],
        months=list(range(1, 12)),  # Jan-Nov
        sample_size=sample_size,
        verbose=verbose,
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print("SPLIT SUMMARY")
        print("=" * 70)
        print(f"Training:   {len(train_df):,} flights")
        print(f"Test:       {len(test_df):,} flights")
        print(f"Total:      {len(train_df) + len(test_df):,} flights")
        print("=" * 70)

    return train_df, test_df


if __name__ == "__main__":
    # Test data loader
    print("Testing Multi-Year Data Loader\n")

    # Test with small sample
    train_df, test_df = load_train_val_test_split(sample_size=1000, verbose=True)

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Check date ranges
    if "FlightDate" in train_df.columns:
        print(
            f"\nTrain date range: {train_df['FlightDate'].min()} to {train_df['FlightDate'].max()}"
        )
        print(
            f"Test date range: {test_df['FlightDate'].min()} to {test_df['FlightDate'].max()}"
        )
