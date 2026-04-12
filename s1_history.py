#!/usr/bin/env python3
"""
Fetch historical NRL results (2020-2025) from the nrl.com draw API and
compute all pre-game form stats for each fixture, then write to nrl_source_data.csv.

Usage:  python fetch_history.py
"""

import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from u1_travel import travel_km as _travel_km

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_PATH    = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
NRL_API     = "https://www.nrl.com/draw/data"
COMPETITION = 111
SEASONS     = list(range(2020, datetime.now().year + 1))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.nrl.com/draw/",
}

NICK_MAP = {
    "Broncos":   "Brisbane Broncos",
    "Raiders":   "Canberra Raiders",
    "Bulldogs":  "Canterbury Bulldogs",
    "Sharks":    "Cronulla Sharks",
    "Titans":    "Gold Coast Titans",
    "Sea Eagles":"Manly Sea Eagles",
    "Storm":     "Melbourne Storm",
    "Knights":   "Newcastle Knights",
    "Warriors":  "New Zealand Warriors",
    "Cowboys":   "North Queensland Cowboys",
    "Eels":      "Parramatta Eels",
    "Panthers":  "Penrith Panthers",
    "Rabbitohs": "South Sydney Rabbitohs",
    "Dragons":   "St George Illawarra Dragons",
    "Roosters":  "Sydney Roosters",
    "Tigers":    "Wests Tigers",
    "Dolphins":  "Dolphins",
}

def canonical(nick: str) -> str:
    return NICK_MAP.get(nick.strip(), nick.strip())


# ─── STEP 1: FETCH RAW RESULTS ────────────────────────────────────────────────

def fetch_round(season: int, round_num: int) -> list[dict]:
    try:
        r = requests.get(
            NRL_API,
            params={"competition": COMPETITION, "season": season, "round": round_num},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for f in data.get("fixtures", []):
            if f.get("matchState") not in ("FullTime", "Post"):
                continue
            hs = f.get("homeTeam", {}).get("score")
            as_ = f.get("awayTeam", {}).get("score")
            if hs is None or as_ is None:
                continue
            dt_str = (f.get("clock") or {}).get("kickOffTimeLong", "")
            date = dt_str[:10] if dt_str else ""
            results.append({
                "season":    season,
                "round":     round_num,
                "date":      date,
                "venue":     f.get("venue", ""),
                "home_team": canonical(f.get("homeTeam", {}).get("nickName", "")),
                "away_team": canonical(f.get("awayTeam", {}).get("nickName", "")),
                "home_score":int(hs),
                "away_score":int(as_),
                "winner":    "home" if int(hs) > int(as_) else "away",
            })
        return results
    except Exception as e:
        print(f"    error: {e}")
        return []


def fetch_all_seasons() -> list[dict]:
    all_games = []
    for season in SEASONS:
        print(f"\nSeason {season}")
        # First call to get round list
        try:
            r = requests.get(
                NRL_API,
                params={"competition": COMPETITION, "season": season, "round": 1},
                headers=HEADERS,
                timeout=15,
            )
            rounds = [x["value"] for x in r.json().get("filterRounds", [])]
        except Exception:
            rounds = list(range(1, 28))

        for rnd in rounds:
            games = fetch_round(season, rnd)
            if games:
                print(f"  R{rnd}: {len(games)} games", end="  ", flush=True)
                all_games.extend(games)
            time.sleep(0.3)   # be polite
        print()

    # Sort chronologically
    all_games.sort(key=lambda g: (g["date"] or "9999", g["season"], g["round"]))
    print(f"\nTotal games fetched: {len(all_games)}")
    return all_games


# ─── STEP 2: COMPUTE PRE-GAME STATS ──────────────────────────────────────────

class TeamState:
    """Tracks a team's running stats as the season progresses."""
    def __init__(self):
        self.reset_season()
        self.all_results: list[dict] = []   # lifetime history for H2H and form

    def reset_season(self):
        self.wins = self.losses = self.draws = 0
        self.pts_for:   list[int] = []
        self.pts_ag:    list[int] = []
        self.home_w = self.home_p = self.away_w = self.away_p = 0
        self.streak = 0
        self.last_date = None

    def snapshot(self, is_home: bool, before_date: str) -> dict:
        total = self.wins + self.losses + self.draws
        pf_avg = np.mean(self.pts_for)  if self.pts_for  else 0.0
        pa_avg = np.mean(self.pts_ag)   if self.pts_ag   else 0.0
        last5_pf = self.pts_for[-5:]
        last5_pa = self.pts_ag[-5:]
        last5_w  = sum(
            1 for i, (pf, pa) in enumerate(zip(reversed(self.pts_for), reversed(self.pts_ag)))
            if i < 5 and pf > pa
        )
        days_rest = 7
        if self.last_date:
            try:
                d = (datetime.fromisoformat(before_date) -
                     datetime.fromisoformat(self.last_date)).days
                days_rest = max(1, d)
            except Exception:
                pass
        return {
            "ladder_pos":             0,   # filled in from table later
            "season_wins":            self.wins,
            "season_losses":          self.losses,
            "season_draws":           self.draws,
            "season_pts_for_avg":     round(pf_avg, 2),
            "season_pts_against_avg": round(pa_avg, 2),
            "last5_wins":             last5_w,
            "last5_pts_for_avg":      round(np.mean(last5_pf) if last5_pf else 0, 2),
            "last5_pts_against_avg":  round(np.mean(last5_pa) if last5_pa else 0, 2),
            "last5_pts_diff_avg":     round(
                np.mean(np.array(last5_pf) - np.array(last5_pa)) if last5_pf else 0, 2
            ),
            "home_record_wins":       self.home_w if is_home else self.away_w,
            "home_record_played":     self.home_p if is_home else self.away_p,
            "win_streak":             self.streak,
            "days_rest":              days_rest,
            "travel_km":              0,
            # Advanced stats (not available from NRL API — model uses what it can)
            "completion_rate":        0,
            "errors_pg":              0,
            "penalties_pg":           0,
            "tackle_eff":             0,
            "line_breaks_pg":         0,
            "tries_pg":               0,
            "post_contact_metres_pg": 0,
            "kick_metres_pg":         0,
            "key_players_out":        0,
            "origin_players_out":     0,
        }

    def update(self, pf: int, pa: int, is_home: bool, date: str):
        won = pf > pa
        drew = pf == pa
        self.pts_for.append(pf)
        self.pts_ag.append(pa)
        if won:
            self.wins += 1
            self.streak = self.streak + 1 if self.streak >= 0 else 1
        elif drew:
            self.draws += 1
            self.streak = 0
        else:
            self.losses += 1
            self.streak = self.streak - 1 if self.streak <= 0 else -1
        if is_home:
            self.home_p += 1
            if won: self.home_w += 1
        else:
            self.away_p += 1
            if won: self.away_w += 1
        self.last_date = date
        self.all_results.append({"date": date, "pf": pf, "pa": pa, "is_home": is_home})


def compute_h2h(team_states: dict, home: str, away: str, before_date: str) -> dict:
    """Compute H2H from all_results accumulated in TeamState."""
    home_results = {r["date"]: r for r in team_states[home].all_results}
    away_results = {r["date"]: r for r in team_states[away].all_results}
    common_dates = sorted(set(home_results) & set(away_results))
    # Filter to only dates before this game
    meetings = [d for d in common_dates if d < before_date][-10:]

    if not meetings:
        return {
            "h2h_home_wins":        5, "h2h_total_meetings":   10,
            "h2h_home_pts_for_avg": 20, "h2h_away_pts_for_avg": 20,
            "h2h_last3yr_home_wins":2, "h2h_last3yr_meetings": 4,
        }

    home_wins = 0
    pf_h, pf_a = [], []
    for d in meetings:
        hr = home_results[d]
        if hr["pf"] > hr["pa"]:
            home_wins += 1
        pf_h.append(hr["pf"]); pf_a.append(hr["pa"])

    try:
        cutoff = (datetime.fromisoformat(before_date) - timedelta(days=3*365)).isoformat()[:10]
    except Exception:
        cutoff = "2000-01-01"
    recent = [d for d in meetings if d >= cutoff]
    r_hw = sum(1 for d in recent if home_results[d]["pf"] > home_results[d]["pa"])

    return {
        "h2h_home_wins":        home_wins,
        "h2h_total_meetings":   len(meetings),
        "h2h_home_pts_for_avg": round(np.mean(pf_h), 2),
        "h2h_away_pts_for_avg": round(np.mean(pf_a), 2),
        "h2h_last3yr_home_wins":r_hw,
        "h2h_last3yr_meetings": len(recent),
    }


def build_ladder(team_states: dict) -> dict:
    """Rank teams by pts (win=2, draw=1), then points diff."""
    rows = []
    for team, state in team_states.items():
        total = state.wins + state.losses + state.draws
        if total == 0:
            continue
        pts = state.wins * 2 + state.draws
        diff = sum(state.pts_for) - sum(state.pts_ag)
        rows.append((team, pts, diff))
    rows.sort(key=lambda x: (-x[1], -x[2]))
    return {team: pos + 1 for pos, (team, _, _) in enumerate(rows)}


def compute_all_features(raw_games: list[dict]) -> list[dict]:
    """Walk through games chronologically and build feature rows."""
    rows = []
    team_states: dict[str, TeamState] = defaultdict(TeamState)
    current_season = None

    for game in raw_games:
        season = game["season"]
        home   = game["home_team"]
        away   = game["away_team"]
        date   = game["date"]
        hsc    = game["home_score"]
        asc    = game["away_score"]

        # Reset season stats when season changes
        if season != current_season:
            for state in team_states.values():
                state.reset_season()
            current_season = season

        if not home or not away:
            continue

        # Snapshot stats BEFORE this game
        h_snap = team_states[home].snapshot(is_home=True,  before_date=date)
        a_snap = team_states[away].snapshot(is_home=False, before_date=date)
        h2h    = compute_h2h(team_states, home, away, date)

        # Build ladder BEFORE this game
        ladder = build_ladder(team_states)
        h_snap["ladder_pos"] = ladder.get(home, 0)
        a_snap["ladder_pos"] = ladder.get(away, 0)

        row = {
            "season":      season,
            "round":       game["round"],
            "date":        date,
            "venue":       game["venue"],
            "home_team":   home,
            "away_team":   away,
            "home_score":  hsc,
            "away_score":  asc,
            "winner":      game["winner"],
            # Home
            "home_ladder_pos":             h_snap["ladder_pos"],
            "home_season_wins":            h_snap["season_wins"],
            "home_season_losses":          h_snap["season_losses"],
            "home_season_draws":           h_snap["season_draws"],
            "home_season_pts_for_avg":     h_snap["season_pts_for_avg"],
            "home_season_pts_against_avg": h_snap["season_pts_against_avg"],
            "home_last5_wins":             h_snap["last5_wins"],
            "home_last5_pts_for_avg":      h_snap["last5_pts_for_avg"],
            "home_last5_pts_against_avg":  h_snap["last5_pts_against_avg"],
            "home_last5_pts_diff_avg":     h_snap["last5_pts_diff_avg"],
            "home_home_record_wins":       h_snap["home_record_wins"],
            "home_home_record_played":     h_snap["home_record_played"],
            "home_win_streak":             h_snap["win_streak"],
            "home_days_rest":              h_snap["days_rest"],
            "home_travel_km":              0.0,
            "home_completion_rate":        0,
            "home_errors_pg":              0,
            "home_penalties_pg":           0,
            "home_tackle_eff":             0,
            "home_line_breaks_pg":         0,
            "home_tries_pg":               0,
            "home_post_contact_metres_pg": 0,
            "home_kick_metres_pg":         0,
            "home_key_players_out":        float("nan"),
            "home_origin_players_out":     0,
            # Away
            "away_ladder_pos":             a_snap["ladder_pos"],
            "away_ladder_pos2":            a_snap["ladder_pos"],
            "away_season_wins":            a_snap["season_wins"],
            "away_season_losses":          a_snap["season_losses"],
            "away_season_draws":           a_snap["season_draws"],
            "away_season_pts_for_avg":     a_snap["season_pts_for_avg"],
            "away_season_pts_against_avg": a_snap["season_pts_against_avg"],
            "away_last5_wins":             a_snap["last5_wins"],
            "away_last5_pts_for_avg":      a_snap["last5_pts_for_avg"],
            "away_last5_pts_against_avg":  a_snap["last5_pts_against_avg"],
            "away_last5_pts_diff_avg":     a_snap["last5_pts_diff_avg"],
            "away_away_record_wins":       a_snap["home_record_wins"],
            "away_away_record_played":     a_snap["home_record_played"],
            "away_win_streak":             a_snap["win_streak"],
            "away_days_rest":              a_snap["days_rest"],
            "away_travel_km":              _travel_km(away, game["venue"]),
            "away_completion_rate":        0,
            "away_errors_pg":              0,
            "away_penalties_pg":           0,
            "away_tackle_eff":             0,
            "away_line_breaks_pg":         0,
            "away_tries_pg":               0,
            "away_post_contact_metres_pg": 0,
            "away_kick_metres_pg":         0,
            "away_key_players_out":        float("nan"),
            "away_origin_players_out":     0,
            # H2H
            **h2h,
            # Market (blank — not available historically)
            "market_home_win_odds":  0,
            "market_away_win_odds":  0,
            "market_home_handicap":  0,
            "market_open_home_odds": 0,
            "market_open_away_odds": 0,
            # Weather
            "weather_temp_c":   20,
            "weather_rain_mm":  0,
            "weather_wind_kmh": 15,
            # Flags
            "is_finals":            1 if (isinstance(game["round"], int) and game["round"] > 27)
                                        or (isinstance(game["round"], str) and not game["round"].isdigit())
                                        else 0,
            "is_elimination_final": 0,
            "is_neutral_venue":     0,
            "crowd":                0,
        }
        rows.append(row)

        # Now update state AFTER recording the pre-game snapshot
        team_states[home].update(hsc, asc, is_home=True,  date=date)
        team_states[away].update(asc, hsc, is_home=False, date=date)

    return rows


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def _rebuild_team_states(existing_df: pd.DataFrame) -> dict:
    """Reconstruct TeamState for every team from an existing CSV DataFrame."""
    states = defaultdict(TeamState)
    current_season = None
    rows = existing_df[existing_df["winner"].notna()].sort_values(
        ["date", "season", "round"]
    )
    for _, row in rows.iterrows():
        season = row["season"]
        if season != current_season:
            for st in states.values():
                st.reset_season()
            current_season = season
        home = row["home_team"]
        away = row["away_team"]
        hsc  = int(row["home_score"])
        asc  = int(row["away_score"])
        date = str(row["date"])[:10]
        states[home].update(hsc, asc, is_home=True,  date=date)
        states[away].update(asc, hsc, is_home=False, date=date)
    return states


def _fetch_new_rounds(last_season: int, last_round: int) -> list[dict]:
    """Fetch only rounds we don't have yet."""
    current_year = datetime.now().year
    new_raw = []
    for season in range(last_season, current_year + 1):
        print(f"\nSeason {season}")
        try:
            r = requests.get(
                NRL_API,
                params={"competition": COMPETITION, "season": season, "round": 1},
                headers=HEADERS,
                timeout=15,
            )
            rounds = [x["value"] for x in r.json().get("filterRounds", [])]
        except Exception:
            rounds = list(range(1, 28))

        for rnd in rounds:
            games = fetch_round(season, rnd)
            if games:
                print(f"  R{rnd}: {len(games)} games", end="  ", flush=True)
                new_raw.extend(games)
            time.sleep(0.3)
        print()
    new_raw.sort(key=lambda g: (g["date"] or "9999", g["season"], g["round"]))
    return new_raw


def main_new_only():
    """Append only rounds not yet in the CSV, preserving existing data."""
    print("=" * 60)
    print("NRL Historical Data Fetcher — new rounds only")
    print("=" * 60)

    if not os.path.exists(OUT_PATH):
        print("No existing CSV — running full fetch instead.")
        main_full()
        return

    existing = pd.read_csv(OUT_PATH, parse_dates=["date"])
    completed = existing[existing["winner"].notna()]
    if len(completed) == 0:
        print("CSV has no completed games — running full fetch instead.")
        main_full()
        return

    last_season = int(completed["season"].max())
    last_round  = int(completed[completed["season"] == last_season]["round"].max())
    print(f"Existing data ends at Season {last_season} Round {last_round}.")

    # Deduplicate key so we don't add rows we already have
    existing_keys = set(
        zip(existing["season"], existing["round"],
            existing["home_team"], existing["away_team"])
    )

    print("\n[1/3] Fetching new rounds from nrl.com...")
    raw_new = _fetch_new_rounds(last_season, last_round)
    raw_new = [g for g in raw_new
               if (g["season"], g["round"], g["home_team"], g["away_team"])
               not in existing_keys]

    if not raw_new:
        print("No new completed games found. Already up to date.")
        return

    print(f"\nFound {len(raw_new)} new game(s).")

    # For correct form stats we replay ALL history then compute only new rows.
    print("\n[2/3] Rebuilding team state from existing CSV for accurate form stats...")
    team_states = _rebuild_team_states(existing)
    current_season = int(completed["season"].max())  # state left at end of existing data

    # compute_all_features processes games from scratch; we adapt it inline
    new_rows = []
    for game in raw_new:
        season = game["season"]
        home   = game["home_team"]
        away   = game["away_team"]
        date   = game["date"]
        hsc    = game["home_score"]
        asc    = game["away_score"]

        if season != current_season:
            for st in team_states.values():
                st.reset_season()
            current_season = season

        if not home or not away:
            continue

        h_snap = team_states[home].snapshot(is_home=True,  before_date=date)
        a_snap = team_states[away].snapshot(is_home=False, before_date=date)
        h2h    = compute_h2h(team_states, home, away, date)
        ladder = build_ladder(team_states)
        h_snap["ladder_pos"] = ladder.get(home, 0)
        a_snap["ladder_pos"] = ladder.get(away, 0)

        row = {
            "season": season, "round": game["round"], "date": date,
            "venue": game["venue"], "home_team": home, "away_team": away,
            "home_score": hsc, "away_score": asc, "winner": game["winner"],
            "home_ladder_pos":             h_snap["ladder_pos"],
            "home_season_wins":            h_snap["season_wins"],
            "home_season_losses":          h_snap["season_losses"],
            "home_season_draws":           h_snap["season_draws"],
            "home_season_pts_for_avg":     h_snap["season_pts_for_avg"],
            "home_season_pts_against_avg": h_snap["season_pts_against_avg"],
            "home_last5_wins":             h_snap["last5_wins"],
            "home_last5_pts_for_avg":      h_snap["last5_pts_for_avg"],
            "home_last5_pts_against_avg":  h_snap["last5_pts_against_avg"],
            "home_last5_pts_diff_avg":     h_snap["last5_pts_diff_avg"],
            "home_home_record_wins":       h_snap["home_record_wins"],
            "home_home_record_played":     h_snap["home_record_played"],
            "home_win_streak":             h_snap["win_streak"],
            "home_days_rest":              h_snap["days_rest"],
            "home_travel_km":              0.0,
            "home_completion_rate": 0, "home_errors_pg": 0,
            "home_penalties_pg": 0, "home_tackle_eff": 0,
            "home_line_breaks_pg": 0, "home_tries_pg": 0,
            "home_post_contact_metres_pg": 0, "home_kick_metres_pg": 0,
            "home_key_players_out": float("nan"),
            "home_origin_players_out": 0,
            "away_ladder_pos":             a_snap["ladder_pos"],
            "away_ladder_pos2":            a_snap["ladder_pos"],
            "away_season_wins":            a_snap["season_wins"],
            "away_season_losses":          a_snap["season_losses"],
            "away_season_draws":           a_snap["season_draws"],
            "away_season_pts_for_avg":     a_snap["season_pts_for_avg"],
            "away_season_pts_against_avg": a_snap["season_pts_against_avg"],
            "away_last5_wins":             a_snap["last5_wins"],
            "away_last5_pts_for_avg":      a_snap["last5_pts_for_avg"],
            "away_last5_pts_against_avg":  a_snap["last5_pts_against_avg"],
            "away_last5_pts_diff_avg":     a_snap["last5_pts_diff_avg"],
            "away_away_record_wins":       a_snap["home_record_wins"],
            "away_away_record_played":     a_snap["home_record_played"],
            "away_win_streak":             a_snap["win_streak"],
            "away_days_rest":              a_snap["days_rest"],
            "away_travel_km":              _travel_km(away, game["venue"]),
            "away_completion_rate": 0, "away_errors_pg": 0,
            "away_penalties_pg": 0, "away_tackle_eff": 0,
            "away_line_breaks_pg": 0, "away_tries_pg": 0,
            "away_post_contact_metres_pg": 0, "away_kick_metres_pg": 0,
            "away_key_players_out": float("nan"),
            "away_origin_players_out": 0,
            **h2h,
            "market_home_win_odds": 0, "market_away_win_odds": 0,
            "market_home_handicap": 0, "market_open_home_odds": 0,
            "market_open_away_odds": 0,
            "weather_temp_c": 20, "weather_rain_mm": 0, "weather_wind_kmh": 15,
            "is_finals": 1 if (isinstance(game["round"], int) and game["round"] > 27)
                             or (isinstance(game["round"], str)
                                 and not str(game["round"]).isdigit()) else 0,
            "is_elimination_final": 0, "is_neutral_venue": 0, "crowd": 0,
        }
        new_rows.append(row)
        team_states[home].update(hsc, asc, is_home=True,  date=date)
        team_states[away].update(asc, hsc, is_home=False, date=date)

    print(f"\n[3/3] Appending {len(new_rows)} row(s) to {OUT_PATH}...")
    new_df = pd.DataFrame(new_rows)
    # Align columns with existing CSV
    new_df = new_df.reindex(columns=existing.columns)
    new_df.to_csv(OUT_PATH, mode="a", header=False, index=False)
    print(f"  Done — {len(new_rows)} new rows appended.")
    print("\nNow run fetch_stats.py and backfill_squads.py --new-only, then retrain.")


def main_full():
    print("=" * 60)
    print("NRL Historical Data Fetcher — 2020 to 2025")
    print("=" * 60)

    print("\n[1/3] Fetching raw results from nrl.com...")
    raw_games = fetch_all_seasons()

    print("\n[2/3] Computing pre-game form stats...")
    feature_rows = compute_all_features(raw_games)
    print(f"  Built {len(feature_rows)} feature rows.")

    print(f"\n[3/3] Writing to {OUT_PATH}...")
    df = pd.DataFrame(feature_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"  Done — {len(df)} rows written.")

    print("\nGames per season:")
    for s, grp in df.groupby("season"):
        hw = (grp["winner"] == "home").sum()
        aw = (grp["winner"] == "away").sum()
        print(f"  {s}: {len(grp)} games  (home wins: {hw}, away wins: {aw})")

    print("\nNow run:  python nrl_model.py --train")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only", action="store_true",
                        help="Only fetch rounds not already in the CSV (much faster)")
    args = parser.parse_args()
    if args.new_only:
        main_new_only()
    else:
        main_full()


if __name__ == "__main__":
    main()
