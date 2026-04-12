#!/usr/bin/env python3
"""
Backfill real weather data into nrl_source_data.csv for all historical games.
Uses Open-Meteo archive API — free, no API key needed.
Run once: python backfill_weather.py
"""

import os
import sys
from collections import defaultdict

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")

from u2_weather import batch_historical, _default

def main():
    df = pd.read_csv(DATA_PATH)
    for col in ["weather_temp_c", "weather_rain_mm", "weather_wind_kmh"]:
        df[col] = df[col].astype(float)

    needs_update = df[
        (df["weather_rain_mm"] == 0) &
        (df["weather_wind_kmh"] == 15) &
        (df["weather_temp_c"] == 20) &
        (df["date"].notna()) &
        (df["date"] != "")
    ]

    if len(needs_update) == 0:
        print("Weather up to date.")
        return

    venue_dates = [(v, d) for v, d in zip(needs_update["venue"].fillna(""), needs_update["date"].fillna("")) if d]

    by_venue = defaultdict(list)
    for v, d in venue_dates:
        by_venue[v].append(d)

    weather_cache = {}
    for venue, dates in sorted(by_venue.items()):
        result = batch_historical([(venue, d) for d in dates])
        weather_cache.update(result)

    updated = 0
    for idx, row in needs_update.iterrows():
        key = (row["venue"] or "", row["date"] or "")
        w = weather_cache.get(key, _default())
        df.at[idx, "weather_temp_c"]   = w["weather_temp_c"]
        df.at[idx, "weather_rain_mm"]  = w["weather_rain_mm"]
        df.at[idx, "weather_wind_kmh"] = w["weather_wind_kmh"]
        updated += 1

    df.to_csv(DATA_PATH, index=False)
    print(f"Weather updated: {updated} rows.")

if __name__ == "__main__":
    main()
