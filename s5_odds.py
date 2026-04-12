#!/usr/bin/env python3
"""
Fetches historical NRL odds from aussportsbetting.com and merges them into
nrl_source_data.csv.

Source: https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/
Data includes: open/close h2h odds and handicap line from 2013 onward.

Maps to columns:
  market_home_win_odds   ← Home Odds Close
  market_away_win_odds   ← Away Odds Close
  market_home_handicap   ← Home Line Close
  market_open_home_odds  ← Home Odds Open
  market_open_away_odds  ← Away Odds Open

Usage:
  python s5_odds.py            # download fresh and backfill all rows
  python s5_odds.py --no-download  # use cached nrl_odds_raw.xlsx
"""

import argparse
import os
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
ODDS_PATH  = os.path.join(SCRIPT_DIR, "nrl_odds_raw.xlsx")
ODDS_URL   = "https://www.aussportsbetting.com/historical_data/nrl.xlsx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/",
}

# Normalise team names to match nrl_source_data.csv
TEAM_NORM = {
    "Canterbury-Bankstown Bulldogs": "Canterbury Bulldogs",
    "Cronulla-Sutherland Sharks":    "Cronulla Sharks",
    "Manly-Warringah Sea Eagles":    "Manly Sea Eagles",
    "North QLD Cowboys":             "North Queensland Cowboys",
    "St George Dragons":             "St George Illawarra Dragons",
    "St. George Illawarra Dragons":  "St George Illawarra Dragons",
}


def normalise(name: str) -> str:
    return TEAM_NORM.get(str(name).strip(), str(name).strip())


def download_odds():
    print(f"Downloading odds data from aussportsbetting.com ...")
    r = requests.get(ODDS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    with open(ODDS_PATH, "wb") as f:
        f.write(r.content)
    print(f"  Saved {len(r.content):,} bytes to {ODDS_PATH}")


def load_odds() -> pd.DataFrame:
    df = pd.read_excel(ODDS_PATH, header=1)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df["home_norm"] = df["Home Team"].apply(normalise)
    df["away_norm"] = df["Away Team"].apply(normalise)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true",
                        help="Use cached nrl_odds_raw.xlsx instead of downloading")
    args = parser.parse_args()

    if not args.no_download or not os.path.exists(ODDS_PATH):
        download_odds()

    print("Loading odds data ...")
    odds = load_odds()
    print(f"  {len(odds)} odds rows, {odds['Date'].min().date()} to {odds['Date'].max().date()}")

    print("Loading nrl_source_data.csv ...")
    src = pd.read_csv(DATA_PATH)
    src["date_dt"] = pd.to_datetime(src["date"], errors="coerce")

    # Ensure columns exist and are float
    for col in ["market_home_win_odds", "market_away_win_odds", "market_home_handicap",
                "market_open_home_odds", "market_open_away_odds"]:
        if col not in src.columns:
            src[col] = 0.0
        src[col] = src[col].astype(float)

    # Build a lookup: (date, home_norm, away_norm) → odds row
    # Also build a team-only lookup for ±1 day fallback
    odds_lookup: dict[tuple, pd.Series] = {}
    odds_team_lookup: dict[tuple, list] = {}  # (home, away) → list of (date, row)
    for _, row in odds.iterrows():
        key = (row["Date"].date(), row["home_norm"], row["away_norm"])
        odds_lookup[key] = row
        team_key = (row["home_norm"], row["away_norm"])
        odds_team_lookup.setdefault(team_key, []).append((row["Date"].date(), row))

    updated = 0
    unmatched = 0

    for idx, row in src.iterrows():
        if pd.isna(row.get("winner")):
            continue  # skip unplayed games
        date = row["date_dt"].date() if pd.notna(row.get("date_dt")) else None
        if date is None:
            continue

        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        key  = (date, home, away)

        orow = odds_lookup.get(key)
        if orow is None:
            # Try ±1 day fallback (timezone differences between sources)
            from datetime import timedelta
            for candidate_date, candidate_row in odds_team_lookup.get((home, away), []):
                if abs((candidate_date - date).days) <= 1:
                    orow = candidate_row
                    break
        if orow is None:
            unmatched += 1
            continue

        def val(*cols):
            """Return first non-null/non-zero value across fallback columns."""
            for col in cols:
                v = orow.get(col)
                try:
                    f = float(v)
                    if pd.notna(f) and f != 0.0:
                        return f
                except (TypeError, ValueError):
                    pass
            return 0.0

        # Use closing odds; fall back to opening odds when closing is unavailable
        src.at[idx, "market_home_win_odds"]  = val("Home Odds Close", "Home Odds Open")
        src.at[idx, "market_away_win_odds"]  = val("Away Odds Close", "Away Odds Open")
        src.at[idx, "market_home_handicap"]  = val("Home Line Close", "Home Line Open")
        src.at[idx, "market_open_home_odds"] = val("Home Odds Open")
        src.at[idx, "market_open_away_odds"] = val("Away Odds Open")
        updated += 1

    src.drop(columns=["date_dt"], inplace=True)
    src.to_csv(DATA_PATH, index=False)

    print(f"\nUpdated : {updated} rows with odds")
    print(f"Unmatched: {unmatched} rows (no odds found for date+teams)")
    print(f"Saved to {DATA_PATH}")

    # Summary of coverage
    filled = (src["market_home_win_odds"] != 0).sum()
    print(f"\nOdds coverage: {filled}/{len(src)} rows ({filled/len(src)*100:.1f}%)")


if __name__ == "__main__":
    main()
