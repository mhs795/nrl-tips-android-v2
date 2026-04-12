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
    print("Loading nrl_source_data.csv...")
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    print(f"  {total} rows loaded.")
    # Ensure weather columns are float so decimal values can be stored
    for col in ["weather_temp_c", "weather_rain_mm", "weather_wind_kmh"]:
        df[col] = df[col].astype(float)

    # Find rows that still have dummy weather (rain=0, wind=15, temp=20)
    needs_update = df[
        (df["weather_rain_mm"] == 0) &
        (df["weather_wind_kmh"] == 15) &
        (df["weather_temp_c"] == 20) &
        (df["date"].notna()) &
        (df["date"] != "")
    ]
    print(f"  {len(needs_update)} rows need real weather data.")

    if len(needs_update) == 0:
        print("Nothing to update.")
        return

    # Build list of (venue, date) pairs
    venue_dates = list(zip(needs_update["venue"].fillna(""), needs_update["date"].fillna("")))
    venue_dates = [(v, d) for v, d in venue_dates if d]

    print(f"\nFetching weather for {len(set(venue_dates))} unique venue/date combos...")
    print("(Grouped by venue to minimise API calls — may take a minute...)\n")

    # Group by venue for progress display
    by_venue = defaultdict(list)
    for v, d in venue_dates:
        by_venue[v].append(d)

    weather_cache = {}
    done = 0
    for venue, dates in sorted(by_venue.items()):
        vdates = [(venue, d) for d in dates]
        result = batch_historical(vdates)
        weather_cache.update(result)
        done += len(dates)
        label = venue[:40] if venue else "(no venue)"
        print(f"  {label:<42} {len(dates):>3} games  [{done}/{len(venue_dates)}]")

    # Apply back to dataframe
    updated = 0
    for idx, row in needs_update.iterrows():
        key = (row["venue"] or "", row["date"] or "")
        w = weather_cache.get(key, _default())
        df.at[idx, "weather_temp_c"]   = w["weather_temp_c"]
        df.at[idx, "weather_rain_mm"]  = w["weather_rain_mm"]
        df.at[idx, "weather_wind_kmh"] = w["weather_wind_kmh"]
        updated += 1

    df.to_csv(DATA_PATH, index=False)
    print(f"\nDone — {updated} rows updated with real weather data.")
    print("Now retrain: python nrl_model.py --train")

if __name__ == "__main__":
    main()
