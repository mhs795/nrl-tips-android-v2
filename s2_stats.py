#!/usr/bin/env python3
"""
Fetches per-game advanced stats from the NRL match centre API and backfills
them into nrl_source_data.csv.

The endpoint:  https://www.nrl.com/draw/{matchCentreUrl}/data
returns completion rate, errors, penalties, tackle efficiency, line breaks,
post-contact metres, kick metres etc. per game — no browser needed.

Usage:  python fetch_stats.py          # backfill all missing stats
        python fetch_stats.py --season 2024 --round 5   # specific round only
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
NRL_BASE   = "https://www.nrl.com"
DRAW_API   = f"{NRL_BASE}/draw/data"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.nrl.com/",
}

# Map from NRL match centre stat title → our column names
STAT_MAP = {
    "Completion Rate":       ("completion_rate",        "completion_rate"),
    "Errors":                ("errors",                 "errors_pg"),
    "Penalties Conceded":    ("penalties",              "penalties_pg"),
    "Tackles Made":          ("tackles",                "tackles"),
    "Missed Tackles":        ("missed_tackles",         "missed_tackles"),
    "Line Breaks":           ("line_breaks",            "line_breaks_pg"),
    "Post Contact Metres":   ("post_contact_metres",    "post_contact_metres_pg"),
    "Kicking Metres":        ("kick_metres",            "kick_metres_pg"),
    "Handling Errors":       ("handling_errors",        "handling_errors"),
}


def fetch_match_stats(match_centre_url: str) -> dict | None:
    """
    Fetch stats for a single game from its match centre URL.
    Returns {home: {stat: val, ...}, away: {stat: val, ...}} or None on failure.
    """
    url = f"{NRL_BASE}{match_centre_url}data"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        groups = data.get("stats", {}).get("groups", [])

        home_stats, away_stats = {}, {}
        for group in groups:
            for stat in group.get("stats", []):
                title = stat.get("title", "")
                if title not in STAT_MAP:
                    continue
                h = stat.get("homeValue", {}).get("value")
                a = stat.get("awayValue", {}).get("value")
                _, col = STAT_MAP[title]
                if h is not None:
                    home_stats[col] = float(h)
                if a is not None:
                    away_stats[col] = float(a)

        # Compute tackle efficiency from tackles + missed tackles
        for s in (home_stats, away_stats):
            t  = s.pop("tackles",        None)
            mt = s.pop("missed_tackles", None)
            s.pop("handling_errors", None)
            if t is not None and mt is not None and (t + mt) > 0:
                s["tackle_eff"] = round(t / (t + mt) * 100, 2)

        # Tries from scoring data (not in stats groups)
        home_tries = data.get("homeTeam", {}).get("scoring", {}).get("tries", {}).get("made")
        away_tries = data.get("awayTeam", {}).get("scoring", {}).get("tries", {}).get("made")
        if home_tries is not None:
            home_stats["tries_pg"] = float(home_tries)
        if away_tries is not None:
            away_stats["tries_pg"] = float(away_tries)

        return {"home": home_stats, "away": away_stats}
    except Exception:
        return None


def get_match_centre_url(season: int, round_num: int,
                          home_team: str, away_team: str) -> str | None:
    """Look up the matchCentreUrl from the NRL draw API."""
    try:
        r = requests.get(
            DRAW_API,
            params={"competition": 111, "season": season, "round": round_num},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            return None
        for f in r.json().get("fixtures", []):
            ht = f.get("homeTeam", {}).get("nickName", "")
            at = f.get("awayTeam", {}).get("nickName", "")
            # Match by team nicknames (loose comparison)
            if (ht.lower() in home_team.lower() or home_team.lower() in ht.lower()) and \
               (at.lower() in away_team.lower() or away_team.lower() in at.lower()):
                return f.get("matchCentreUrl")
    except Exception:
        pass
    return None


def build_round_url_cache(season: int, round_num: int) -> dict[tuple, str]:
    """Fetch all matchCentreUrls for a round at once."""
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
                ht = f.get("homeTeam", {}).get("nickName", "")
                at = f.get("awayTeam", {}).get("nickName", "")
                url = f.get("matchCentreUrl")
                if ht and at and url:
                    cache[(ht, at)] = url
    except Exception:
        pass
    return cache


STAT_COLS = [
    "home_completion_rate", "home_errors_pg", "home_penalties_pg",
    "home_tackle_eff", "home_line_breaks_pg",
    "home_post_contact_metres_pg", "home_kick_metres_pg", "home_tries_pg",
    "away_completion_rate", "away_errors_pg", "away_penalties_pg",
    "away_tackle_eff", "away_line_breaks_pg",
    "away_post_contact_metres_pg", "away_kick_metres_pg", "away_tries_pg",
]


def needs_stats(row) -> bool:
    """True if this row is missing any advanced stats (including tries)."""
    return (
        all(row.get(c, 0) == 0 for c in STAT_COLS)
        or row.get("home_kick_metres_pg", 0) == 0
        or row.get("home_tries_pg", 0) == 0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round",  type=int, default=None)
    args = parser.parse_args()

    print("Loading nrl_source_data.csv...")
    df = pd.read_csv(DATA_PATH)
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].astype(float)

    # Filter to rows needing stats
    mask = df.apply(needs_stats, axis=1)
    if args.season:
        mask &= df["season"] == args.season
    if args.round:
        mask &= df["round"] == args.round
    # Only historical (completed) games
    mask &= df["winner"].notna()

    todo = df[mask]
    print(f"  {len(todo)} rows need advanced stats.")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    # Group by season+round for efficiency
    updated = 0
    url_cache: dict[tuple, str] = {}
    last_season_round = (None, None)

    for idx, row in todo.iterrows():
        season    = int(row["season"])
        round_num = int(row["round"])
        home      = str(row["home_team"])
        away      = str(row["away_team"])

        # Rebuild URL cache when season/round changes
        if (season, round_num) != last_season_round:
            print(f"\n  Season {season} Round {round_num}...", end=" ", flush=True)
            url_cache = build_round_url_cache(season, round_num)
            last_season_round = (season, round_num)
            time.sleep(0.3)

        # Find the matchCentreUrl
        mc_url = None
        for (ht, at), url in url_cache.items():
            if (ht.lower() in home.lower() or home.lower() in ht.lower()) and \
               (at.lower() in away.lower() or away.lower() in at.lower()):
                mc_url = url
                break

        if not mc_url:
            print("?", end="", flush=True)
            continue

        stats = fetch_match_stats(mc_url)
        if not stats:
            print("x", end="", flush=True)
            time.sleep(0.2)
            continue

        # Write home stats
        for col, val in stats["home"].items():
            df.at[idx, f"home_{col}"] = val
        # Write away stats
        for col, val in stats["away"].items():
            df.at[idx, f"away_{col}"] = val

        updated += 1
        print(".", end="", flush=True)
        time.sleep(0.25)   # polite — ~4 req/sec

    print(f"\n\nUpdated {updated} rows with real advanced stats.")
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved to {DATA_PATH}")
    print("Now retrain: python nrl_model.py --train")


if __name__ == "__main__":
    main()
