#!/usr/bin/env python3
"""
Backfill home_key_players_out / away_key_players_out into nrl_source_data.csv
by fetching actual player lists from the NRL match centre for every historical game.

Uses SquadTracker to compute "key players out vs rolling baseline" for each game,
processing games in strict chronological order.

Player data is cached locally in squad_player_cache.json to avoid re-fetching
historical data on every run (prevents nrl.com rate-limiting).

Usage:  python backfill_squads.py
        python backfill_squads.py --season 2025 --round 5   # specific slice only
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
CACHE_PATH  = os.path.join(SCRIPT_DIR, "squad_player_cache.json")
NRL_BASE    = "https://www.nrl.com"
DRAW_API    = f"{NRL_BASE}/draw/data"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.nrl.com/",
}

sys.path.insert(0, SCRIPT_DIR)
from u3_squad import SquadTracker, fetch_players


def load_player_cache() -> dict:
    """Load cached player data from disk. Keys: 'season/round/home/away'."""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_player_cache(cache: dict):
    """Persist player cache to disk."""
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def cache_key(season: int, round_num: int, home: str, away: str) -> str:
    return f"{season}/{round_num}/{home}/{away}"


def build_round_url_cache(season: int, round_num: int) -> dict[tuple, str]:
    """Fetch all matchCentreUrls for a season+round in one API call."""
    cache = {}
    try:
        r = requests.get(
            DRAW_API,
            params={"competition": 111, "season": season, "round": round_num},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            for f in r.json().get("fixtures", []):
                ht  = f.get("homeTeam", {}).get("nickName", "")
                at  = f.get("awayTeam", {}).get("nickName", "")
                url = f.get("matchCentreUrl")
                if ht and at and url:
                    cache[(ht.lower(), at.lower())] = url
    except Exception:
        pass
    return cache


def find_mc_url(cache: dict, home: str, away: str) -> str | None:
    """Fuzzy-match a home/away pair to a cached matchCentreUrl."""
    hl, al = home.lower(), away.lower()
    for (ht, at), url in cache.items():
        if (ht in hl or hl in ht) and (at in al or al in at):
            return url
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int,  default=None)
    parser.add_argument("--round",    type=int,  default=None)
    parser.add_argument("--new-only", action="store_true",
                        help="Only process rows where key_players_out is not yet computed")
    args = parser.parse_args()

    print("Loading nrl_source_data.csv...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Ensure columns exist — NaN means "not yet computed"; 0 means "computed, none out"
    for col in ("home_key_players_out", "away_key_players_out"):
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows to write: completed games matching any filter
    mask = df["winner"].notna()
    if args.season:
        mask &= df["season"] == args.season
    if args.round:
        mask &= df["round"] == args.round
    if args.new_only:
        mask &= df["home_key_players_out"].isna()

    # Process ALL completed games in chronological order for the tracker,
    # but only write values for the filtered slice.
    all_completed = df[df["winner"].notna()].sort_values(["date", "season", "round"])
    write_mask    = set(df[mask].index)

    print(f"  Processing {len(all_completed)} completed games chronologically...")
    print(f"  Will write squad info for {len(write_mask)} rows.")

    # Load local player cache — avoids re-fetching historical data every run
    player_cache = load_player_cache()
    cache_dirty  = False
    cached_hits  = 0

    tracker = SquadTracker()
    url_cache: dict[tuple, dict] = {}   # (season, round) -> {(ht,at): url}
    last_season = None

    updated = 0
    skipped = 0

    for idx, row in all_completed.iterrows():
        season    = int(row["season"])
        round_num = int(row["round"])
        home      = str(row["home_team"])
        away      = str(row["away_team"])

        # Reset tracker at season boundaries
        if season != last_season:
            tracker.reset_season()
            last_season = season

        ck = cache_key(season, round_num, home, away)

        if ck in player_cache:
            # Use cached player data — no network request needed
            players = player_cache[ck]
            cached_hits += 1
        else:
            # Fetch matchCentreUrl cache for this round if not cached
            key = (season, round_num)
            if key not in url_cache:
                print(f"\n  Season {season} Round {round_num}...", end=" ", flush=True)
                url_cache[key] = build_round_url_cache(season, round_num)
                time.sleep(0.3)

            mc_url = find_mc_url(url_cache[key], home, away)
            if not mc_url:
                print("?", end="", flush=True)
                skipped += 1
                continue

            players = fetch_players(mc_url)
            # Save to cache (even if empty — avoids re-fetching on next run)
            player_cache[ck] = players
            cache_dirty = True
            time.sleep(0.15)   # polite ~6 req/sec

        home_players = players.get("home", [])
        away_players = players.get("away", [])

        # Compute key_players_out BEFORE recording this game in the tracker
        home_out = tracker.key_players_out(home, home_players)
        away_out = tracker.key_players_out(away, away_players)

        # Write to dataframe if this row is in the write set
        if idx in write_mask:
            df.at[idx, "home_key_players_out"] = home_out
            df.at[idx, "away_key_players_out"] = away_out
            updated += 1
            print(".", end="", flush=True)

        # Update tracker with actual starters for future games
        tracker.update(home, home_players)
        tracker.update(away, away_players)

    # Persist cache updates
    if cache_dirty:
        save_player_cache(player_cache)
        print(f"\n  Player cache updated ({len(player_cache)} games stored).")

    if cached_hits:
        print(f"  {cached_hits} games loaded from local cache (no network requests).")

    print(f"\nUpdated {updated} rows | Skipped {skipped} (no matchCentreUrl found)")
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved → {DATA_PATH}")
    print("Now retrain: python nrl_model.py --train")


if __name__ == "__main__":
    main()
