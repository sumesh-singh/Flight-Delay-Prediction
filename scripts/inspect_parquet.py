import pandas as pd
import glob
import os

try:
    files = glob.glob("data/external/traffic/*.parquet")
    if files:
        f = files[0]
        print(f"Reading {f}...")
        df = pd.read_parquet(f)
        print(df.head())
        print(df.describe())
    else:
        print("No parquet files found.")
except Exception as e:
    print(e)
