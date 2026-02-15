"""
BALANCED Multi-Year Data Loader (35 columns)

DROP-IN REPLACEMENT for the minimal (18 column) loader.
Same class name and methods, but loads 35 columns instead of 18.

Replace: src/data/multiyear_loader.py with this file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


class MultiYearDataLoader:
    """
    BALANCED: Loads 35 columns (vs 18 minimal or 109 full).
    
    Drop-in replacement - same class name and methods as original.
    """
    
    DTYPE_MAP = {
        # Identifiers (8 columns)
        'Year': 'int16',
        'Quarter': 'int8',
        'Month': 'int8',
        'DayofMonth': 'int8',
        'DayOfWeek': 'int8',
        'FlightDate': 'object',
        'Reporting_Airline': 'category',
        'Flight_Number_Reporting_Airline': 'int32',
        
        # Airports (4 columns)
        'OriginAirportID': 'int32',
        'Origin': 'category',
        'DestAirportID': 'int32',
        'Dest': 'category',
        
        # Scheduled times (2 columns)
        'CRSDepTime': 'float32',
        'CRSArrTime': 'float32',
        
        # Actual times (4 columns)
        'DepTime': 'float32',
        'ArrTime': 'float32',
        'WheelsOff': 'float32',
        'WheelsOn': 'float32',
        
        # Delays (6 columns)
        'DepDelay': 'float32',
        'DepDelayMinutes': 'float32',
        'ArrDelay': 'float32',
        'ArrDelayMinutes': 'float32',
        'DepDel15': 'float32',
        'ArrDel15': 'float32',
        
        # Duration & Distance (5 columns)
        'CRSElapsedTime': 'float32',
        'ActualElapsedTime': 'float32',
        'AirTime': 'float32',
        'TaxiOut': 'float32',
        'TaxiIn': 'float32',
        
        # Status (2 columns)
        'Cancelled': 'float32',
        'Diverted': 'float32',
        
        # Other (2 columns)
        'Distance': 'float32',
        'DistanceGroup': 'int8',
        
        # Delay causes (5 columns)
        'CarrierDelay': 'float32',
        'WeatherDelay': 'float32',
        'NASDelay': 'float32',
        'SecurityDelay': 'float32',
        'LateAircraftDelay': 'float32',
    }
    
    # BALANCED: 35 columns (vs 18 minimal)
    ESSENTIAL_COLS = [
        # Core identifiers
        'FlightDate', 'Month', 'DayOfWeek', 'DayofMonth',
        'Reporting_Airline', 'Flight_Number_Reporting_Airline',
        
        # Airports (with IDs)
        'OriginAirportID', 'Origin', 'DestAirportID', 'Dest',
        
        # Times
        'CRSDepTime', 'DepTime', 'CRSArrTime', 'ArrTime',
        'WheelsOff', 'WheelsOn',
        
        # Delays
        'DepDelay', 'DepDelayMinutes', 'DepDel15',
        'ArrDelay', 'ArrDelayMinutes', 'ArrDel15',
        
        # Duration
        'CRSElapsedTime', 'ActualElapsedTime', 'AirTime',
        'TaxiOut', 'TaxiIn',
        
        # Status
        'Cancelled', 'Diverted',
        
        # Distance
        'Distance', 'DistanceGroup',
        
        # Delay causes
        'CarrierDelay', 'WeatherDelay', 'NASDelay',
        'SecurityDelay', 'LateAircraftDelay',
    ]
    
    def __init__(self, data_root: Path = None):
        if data_root is None:
            data_root = Path(__file__).parent.parent.parent / "data" / "raw"
        self.data_root = Path(data_root)
    
    def get_available_files(self, year: int) -> List[Path]:
        """Get list of CSV files for a year."""
        year_dir = self.data_root / str(year)
        if not year_dir.exists():
            return []
        return sorted(year_dir.glob("On_Time_*.csv"))
    
    def load_month_optimized(
        self, 
        year: int, 
        month: int, 
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load single month with 35 balanced columns.
        
        Args:
            year: Year
            month: Month (1-12)
            nrows: Limit rows (for testing)
        
        Returns:
            DataFrame with 35 columns, optimized dtypes
        """
        filename = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"
        filepath = self.data_root / str(year) / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(
            filepath,
            usecols=lambda c: c in self.ESSENTIAL_COLS,
            dtype={k: v for k, v in self.DTYPE_MAP.items() if k in self.ESSENTIAL_COLS},
            nrows=nrows,
            low_memory=False
        )
        
        return df
    
    def load_year_optimized(
        self, 
        year: int, 
        months: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load full year with 35 balanced columns.
        
        Args:
            year: Year to load
            months: Specific months (default: all available)
        
        Returns:
            Combined DataFrame
        """
        year_dir = self.data_root / str(year)
        
        if not year_dir.exists():
            raise FileNotFoundError(f"Year directory not found: {year_dir}")
        
        # Get file list
        if months is None:
            csv_files = sorted(year_dir.glob("On_Time_*.csv"))
        else:
            csv_files = []
            for month in months:
                filename = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"
                filepath = year_dir / filename
                if filepath.exists():
                    csv_files.append(filepath)
        
        # Load all files
        dfs = []
        for csv_file in csv_files:
            print(f"Loading {csv_file.name}...", end=" ")
            
            try:
                df = pd.read_csv(
                    csv_file,
                    usecols=lambda c: c in self.ESSENTIAL_COLS,
                    dtype={k: v for k, v in self.DTYPE_MAP.items() if k in self.ESSENTIAL_COLS},
                    low_memory=False
                )
                dfs.append(df)
                print(f"{len(df):,} rows")
                
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No data loaded for year {year}")
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal for {year}: {len(combined):,} rows")
        print(f"Columns loaded: {len(combined.columns)}\n")
        
        return combined
    
    def load_train_test_split_optimized(self) -> tuple:
        """
        Load train (2023-2024) and test (2025) with 35 balanced columns.
        
        Returns:
            (train_df, test_df)
        """
        print("=" * 70)
        print("LOADING MULTI-YEAR DATA (BALANCED - 35 COLUMNS)")
        print("=" * 70)
        
        print("\nTraining data: 2023-2024")
        train_2023 = self.load_year_optimized(2023)
        train_2024 = self.load_year_optimized(2024)
        train_df = pd.concat([train_2023, train_2024], ignore_index=True)
        del train_2023, train_2024
        
        print("\nTest data: 2025 (Jan-Nov)")
        test_df = self.load_year_optimized(2025, months=list(range(1, 12)))
        
        print("\n" + "=" * 70)
        print("SPLIT SUMMARY")
        print("=" * 70)
        print(f"Columns: {len(train_df.columns)} (35 balanced)")
        print(f"Training:   {len(train_df):,} flights")
        print(f"Test:       {len(test_df):,} flights")
        print(f"Total:      {len(train_df) + len(test_df):,} flights")
        
        # Memory usage
        train_mem = train_df.memory_usage(deep=True).sum() / 1e6
        test_mem = test_df.memory_usage(deep=True).sum() / 1e6
        print(f"\nMemory usage:")
        print(f"Training:   {train_mem:.1f} MB")
        print(f"Test:       {test_mem:.1f} MB")
        print(f"Total:      {(train_mem + test_mem):.1f} MB")
        print("=" * 70)
        
        return train_df, test_df


# Convenience function
def load_optimized_data() -> tuple:
    """Load data with 35 balanced columns."""
    loader = MultiYearDataLoader()
    return loader.load_train_test_split_optimized()


if __name__ == "__main__":
    print("Testing Balanced Data Loader (35 columns)\n")
    
    loader = MultiYearDataLoader()
    
    # Test single month
    df = loader.load_month_optimized(2023, 1, nrows=10000)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\nColumn list:\n{list(df.columns)}")
