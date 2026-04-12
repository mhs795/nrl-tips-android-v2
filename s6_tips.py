#!/usr/bin/env python3
"""
NRL Auto Tipper
Fetches current round fixtures from nrl.com, computes features from local history,
runs the XGBoost model and outputs formatted tips.

Usage:
  python auto_tip.py                  # auto-detect round
  python auto_tip.py --round 12       # specific round
  python auto_tip.py --season 2025 --round 12
"""

import argparse
import json
import os
import pickle
import re
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH          = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
MODEL_PATH         = os.path.join(SCRIPT_DIR, "nrl_model.pkl")
MODEL_PATH_NO_ODDS = os.path.join(SCRIPT_DIR, "nrl_model_no_odds.pkl")

def _load_odds_key() -> str:
    """Read ODDS_API_KEY from env, falling back to ~/.claude/settings.json."""
    key = os.environ.get("ODDS_API_KEY", "")
    if key:
        return key
    try:
        import json as _json
        cfg = os.path.expanduser("~/.claude/settings.json")
        with open(cfg) as _f:
            return _json.load(_f).get("env", {}).get("ODDS_API_KEY", "")
    except Exception:
        return ""

ODDS_API_KEY = _load_odds_key()

CURRENT_SEASON = datetime.now().year

def _travel_km(team: str, venue: str) -> float:
    try:
        from u1_travel import travel_km
        return travel_km(team, venue)
    except Exception:
        return 0.0


def _fetch_squad_info(games: list[dict], season: int, round_num: int) -> dict[tuple, tuple]:
    """
    Fetch team lists for the upcoming round from the NRL match centre.
    Returns {(home_team, away_team): (home_key_players_out, away_key_players_out)}.

    Builds a SquadTracker from the last KEY_WINDOW rounds to establish baselines,
    then computes key_players_out for the upcoming round.
    Returns empty dict on any failure (callers fall back to 0s).
    """
    try:
        from u3_squad import SquadTracker, fetch_players, KEY_WINDOW, KEY_NUMBERS
    except Exception:
        return {}

    def _round_url_map(s: int, rnd: int) -> dict[tuple, str]:
        """Fetch (home, away) -> matchCentreUrl for one round."""
        try:
            r = requests.get(
                "https://www.nrl.com/draw/data",
                params={"competition": 111, "season": s, "round": rnd},
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code != 200:
                return {}
            result = {}
            for f in r.json().get("fixtures", []):
                ht  = canonical(f.get("homeTeam", {}).get("nickName", ""))
                at  = canonical(f.get("awayTeam", {}).get("nickName", ""))
                url = f.get("matchCentreUrl")
                if ht and at and url:
                    result[(ht, at)] = url
            return result
        except Exception:
            return {}

    tracker = SquadTracker()
    name_lookup: dict[int, str] = {}  # playerId -> "F. Lastname"

    def _collect_names(player_list):
        for p in player_list:
            pid = p.get("playerId")
            if pid:
                name_lookup[pid] = f"{p.get('firstName', '?')[0]}. {p.get('lastName', '?')}"

    NUM_TO_POS = {1: "Fullback", 6: "Five-Eighth", 7: "Halfback", 9: "Hooker"}

    def _absent_names(team, players):
        starter_map = {p["number"]: p for p in players if p.get("number") in KEY_NUMBERS}
        absent = []
        for num in sorted(KEY_NUMBERS):
            regular_id = tracker.regular_starter(team, num)
            if regular_id is None:
                continue
            actual = starter_map.get(num)
            if actual is None or actual.get("playerId") != regular_id:
                pos = NUM_TO_POS.get(num, f"#{num}")
                reg_name = name_lookup.get(regular_id, "unknown")
                absent.append(f"{reg_name} ({pos})")
        return ", ".join(absent)

    # Seed tracker with last KEY_WINDOW completed rounds for baselines
    seed_start = max(1, round_num - KEY_WINDOW)
    for past_rnd in range(seed_start, round_num):
        url_map = _round_url_map(season, past_rnd)
        if not url_map:
            continue
        for (ht, at), url in url_map.items():
            players = fetch_players(url)
            if players.get("home") or players.get("away"):
                _collect_names(players.get("home", []))
                _collect_names(players.get("away", []))
                tracker.update(ht, players.get("home", []))
                tracker.update(at, players.get("away", []))
        import time as _time
        _time.sleep(0.3)

    # Now fetch this round's player lists and compute key_players_out
    url_map = _round_url_map(season, round_num)
    if not url_map:
        return {}

    result = {}
    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        mc_url = url_map.get((home, away)) or url_map.get((away, home))
        if not mc_url:
            result[(home, away)] = (0, 0, "", "")
            continue
        players = fetch_players(mc_url)
        home_players = players.get("home", [])
        away_players = players.get("away", [])
        _collect_names(home_players)
        _collect_names(away_players)
        home_out = tracker.key_players_out(home, home_players)
        away_out = tracker.key_players_out(away, away_players)
        home_names = _absent_names(home, home_players)
        away_names = _absent_names(away, away_players)
        result[(home, away)] = (home_out, away_out, home_names, away_names)

    return result


def _get_weather(venue: str, game_date: str) -> dict:
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from u2_weather import get_weather
        return get_weather(venue, game_date)
    except Exception:
        return {"weather_temp_c": 20.0, "weather_rain_mm": 0.0, "weather_wind_kmh": 15.0}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html,*/*",
}

# Canonical team names used in nrl_source_data.csv
TEAM_ALIASES = {
    "brisbane broncos":          "Brisbane Broncos",
    "broncos":                   "Brisbane Broncos",
    "canberra raiders":          "Canberra Raiders",
    "raiders":                   "Canberra Raiders",
    "canterbury bulldogs":       "Canterbury Bulldogs",
    "bulldogs":                  "Canterbury Bulldogs",
    "cronulla-sutherland sharks":"Cronulla Sharks",
    "cronulla sharks":           "Cronulla Sharks",
    "sharks":                    "Cronulla Sharks",
    "gold coast titans":         "Gold Coast Titans",
    "titans":                    "Gold Coast Titans",
    "manly sea eagles":          "Manly Sea Eagles",
    "sea eagles":                "Manly Sea Eagles",
    "manly-warringah sea eagles":"Manly Sea Eagles",
    "manly warringah sea eagles":"Manly Sea Eagles",
    "melbourne storm":           "Melbourne Storm",
    "storm":                     "Melbourne Storm",
    "newcastle knights":         "Newcastle Knights",
    "knights":                   "Newcastle Knights",
    "new zealand warriors":      "New Zealand Warriors",
    "warriors":                  "New Zealand Warriors",
    "nz warriors":               "New Zealand Warriors",
    "north queensland cowboys":  "North Queensland Cowboys",
    "cowboys":                   "North Queensland Cowboys",
    "parramatta eels":           "Parramatta Eels",
    "eels":                      "Parramatta Eels",
    "penrith panthers":          "Penrith Panthers",
    "panthers":                  "Penrith Panthers",
    "south sydney rabbitohs":    "South Sydney Rabbitohs",
    "rabbitohs":                 "South Sydney Rabbitohs",
    "st. george illawarra dragons": "St George Illawarra Dragons",
    "st george illawarra dragons":  "St George Illawarra Dragons",
    "dragons":                   "St George Illawarra Dragons",
    "sydney roosters":           "Sydney Roosters",
    "roosters":                  "Sydney Roosters",
    "wests tigers":              "Wests Tigers",
    "tigers":                    "Wests Tigers",
    "dolphins":                  "Dolphins",
}

def canonical(name: str) -> str:
    return TEAM_ALIASES.get(name.lower().strip(), name.strip())


# ─── ROUND DETECTION ──────────────────────────────────────────────────────────

def estimate_current_round(season: int) -> int:
    """
    Find the next upcoming NRL round by querying the NRL.com draw API.
    Scans rounds until we find one that has at least one future game.
    Falls back to date-based estimate if the API is unreachable.
    """
    today = datetime.now().date()

    # Date-based fallback
    season_start = datetime(season, 3, 6)
    days = (datetime.now() - season_start).days
    if days < 0:
        fallback = 1
    else:
        fallback = days // 7 + 1
        if datetime.now().weekday() >= 5:
            fallback += 1
    fallback = min(max(1, fallback), 27)

    # Search from a few rounds before the fallback estimate to catch cases
    # where the API-based season started earlier than the date formula assumes.
    search_start = max(1, fallback - 4)
    for rnd in range(search_start, 28):
        try:
            games = fetch_draw_nrlcom(season, rnd)
        except Exception:
            break
        if not games:
            continue
        dates = [g.get("date", "") for g in games if g.get("date")]
        if not dates:
            continue
        # If any game in this round is today or in the future, this is our round
        future = [d for d in dates if d >= str(today)]
        if future:
            return rnd

    return fallback


def best_round_for_odds(season: int, odds_map: dict) -> int:
    """Find the NRL draw round whose teams best overlap with live odds fixtures."""
    if not odds_map:
        return estimate_current_round(season)
    odds_teams = set(t for pair in odds_map for t in pair)
    base = estimate_current_round(season)
    best_round, best_score = base, -1
    for rnd in [base, base - 1]:
        if rnd < 1 or rnd > 27:
            continue
        games = fetch_draw_nrlcom(season, rnd)
        draw_teams = set(t for g in games for t in (g["home_team"], g["away_team"]))
        score = len(odds_teams & draw_teams)
        if score > best_score:
            best_score, best_round = score, rnd
    return best_round


# ─── NRL.COM DRAW FETCHER ─────────────────────────────────────────────────────

def fetch_draw_nrlcom(season: int, round_num: int) -> list[dict]:
    """
    Try to fetch the draw from nrl.com.
    Returns list of dicts: {home_team, away_team, venue, date, round}
    """
    # NRL.com Nuxt-based widget endpoint (used by the web app internally)
    api_url = (
        f"https://www.nrl.com/draw/data"
        f"?competition=111&season={season}&round={round_num}"
    )
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return _parse_nrl_api_draw(data, round_num, season)
    except Exception:
        pass

    # Fallback: scrape the HTML draw page
    try:
        html_url = f"https://www.nrl.com/draw/nrl/{season}/{round_num}/"
        r = requests.get(html_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return _parse_nrl_html_draw(r.text, round_num, season)
    except Exception:
        pass

    return []


def _parse_nrl_api_draw(data: dict, round_num: int, season: int) -> list[dict]:
    """Parse the JSON structure returned by nrl.com/draw/data."""
    games = []
    try:
        # The API may nest fixtures under various keys
        fixtures = (
            data.get("fixtures") or
            data.get("draw", {}).get("fixtures") or
            data.get("data", {}).get("fixtures") or
            []
        )
        for f in fixtures:
            home = (
                f.get("homeTeam", {}).get("nickName") or
                f.get("homeTeam", {}).get("name", "")
            )
            away = (
                f.get("awayTeam", {}).get("nickName") or
                f.get("awayTeam", {}).get("name", "")
            )
            venue = f.get("venue", {}).get("name", "") if isinstance(f.get("venue"), dict) else f.get("venue", "")
            date_str = f.get("clock", {}).get("kickOffTimeLong") or f.get("matchDate", "")
            games.append({
                "season":  season,
                "round":   round_num,
                "home_team": canonical(home),
                "away_team": canonical(away),
                "venue":   venue,
                "date":    date_str[:10] if date_str else "",
                "kickoff": date_str or "",   # full ISO datetime for kick-off time checks
            })
    except Exception:
        pass
    return games


def _parse_nrl_html_draw(html: str, round_num: int, season: int) -> list[dict]:
    """Extract game data from embedded JSON in the NRL.com Nuxt page."""
    games = []
    try:
        # Nuxt stores state in window.__NUXT__ or __nuxt_data__
        for pattern in [
            r'window\.__NUXT__\s*=\s*(\{[\s\S]*?\});\s*</script>',
            r'"fixtures"\s*:\s*(\[[\s\S]*?\])',
        ]:
            m = re.search(pattern, html)
            if m:
                raw = m.group(1)
                # Try to extract fixture-like objects
                home_matches = re.findall(r'"homeTeam"[^}]*?"(?:nickName|name)"\s*:\s*"([^"]+)"', raw)
                away_matches = re.findall(r'"awayTeam"[^}]*?"(?:nickName|name)"\s*:\s*"([^"]+)"', raw)
                venues       = re.findall(r'"venue"[^}]*?"name"\s*:\s*"([^"]+)"', raw)
                dates        = re.findall(r'"matchDate"\s*:\s*"([^"]{10})', raw)
                for i, (h, a) in enumerate(zip(home_matches, away_matches)):
                    games.append({
                        "season": season,
                        "round": round_num,
                        "home_team": canonical(h),
                        "away_team": canonical(a),
                        "venue": venues[i] if i < len(venues) else "",
                        "date": dates[i] if i < len(dates) else "",
                    })
                if games:
                    break
    except Exception:
        pass
    return games


def _fetch_draw_from_csv(season: int, round_num: int) -> list[dict]:
    """Read fixtures for a round from nrl_source_data.csv (fallback when API unavailable)."""
    if not os.path.exists(DATA_PATH):
        return []
    try:
        df   = pd.read_csv(DATA_PATH, usecols=['season', 'round', 'home_team', 'away_team', 'venue', 'date'])
        mask = (df['season'].astype(int) == season) & (df['round'].astype(int) == round_num)
        rows = df[mask]
        if rows.empty:
            return []
        print(f"  [fallback] Loaded {len(rows)} fixtures from local CSV.")
        return [
            {
                'season':    season,
                'round':     round_num,
                'home_team': canonical(str(row['home_team'])),
                'away_team': canonical(str(row['away_team'])),
                'venue':     str(row.get('venue', '')),
                'date':      str(row.get('date', '')),
            }
            for _, row in rows.iterrows()
        ]
    except Exception as e:
        print(f"  [fallback] CSV read failed: {e}")
        return []


# ─── LADDER FETCHER ───────────────────────────────────────────────────────────

def fetch_ladder_nrlcom(season: int) -> dict[str, dict]:
    """
    Fetch NRL ladder. Returns dict keyed by canonical team name:
    {pos, wins, losses, draws, pts_for, pts_against}
    """
    api_url = f"https://www.nrl.com/ladder/data?competition=111&season={season}"
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return _parse_nrl_ladder(r.json())
    except Exception:
        pass

    try:
        html_url = f"https://www.nrl.com/ladder/nrl/{season}/"
        r = requests.get(html_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return _parse_nrl_ladder_html(r.text)
    except Exception:
        pass

    return {}


def _parse_nrl_ladder(data: dict) -> dict:
    ladder = {}
    try:
        # New API format (2025+): positions[].stats + teamNickname
        if "positions" in data:
            for i, row in enumerate(data["positions"]):
                team = canonical(row.get("teamNickname") or "")
                stats = row.get("stats", {})
                if team:
                    ladder[team] = {
                        "pos":         i + 1,
                        "wins":        int(stats.get("wins", 0)),
                        "losses":      int(stats.get("lost", 0)),
                        "draws":       int(stats.get("drawn", 0)),
                        "pts_for":     float(stats.get("points for", 0)),
                        "pts_against": float(stats.get("points against", 0)),
                    }
            return ladder

        # Legacy API format: ladderStandings / standings
        rows = (
            data.get("ladderStandings") or
            data.get("ladder", {}).get("standings") or
            data.get("standings") or
            []
        )
        for i, row in enumerate(rows):
            team = canonical(
                row.get("teamNickName") or row.get("teamName") or ""
            )
            if team:
                ladder[team] = {
                    "pos":          i + 1,
                    "wins":         int(row.get("wins", 0)),
                    "losses":       int(row.get("losses", 0)),
                    "draws":        int(row.get("draws", 0)),
                    "pts_for":      float(row.get("pointsFor", 0)),
                    "pts_against":  float(row.get("pointsAgainst", 0)),
                }
    except Exception:
        pass
    return ladder


def _parse_nrl_ladder_html(html: str) -> dict:
    """Extract ladder from HTML page."""
    ladder = {}
    try:
        rows = re.findall(
            r'<tr[^>]*>[\s\S]*?<td[^>]*>(\d+)</td>[\s\S]*?<td[^>]*>([^<]+)</td>[\s\S]*?</tr>',
            html
        )
        for pos, name in rows:
            team = canonical(name.strip())
            if team:
                ladder[team] = {"pos": int(pos), "wins": 0, "losses": 0,
                                "draws": 0, "pts_for": 0, "pts_against": 0}
    except Exception:
        pass
    return ladder


def _fetch_odds_from_csv(season: int, round_num: int) -> dict[tuple, dict]:
    """Read odds for a specific round from nrl_source_data.csv (fallback for past rounds)."""
    if not os.path.exists(DATA_PATH):
        return {}
    try:
        cols = ['season', 'round', 'home_team', 'away_team', 
                'market_home_win_odds', 'market_away_win_odds', 'market_home_handicap']
        df   = pd.read_csv(DATA_PATH, usecols=cols)
        mask = (df['season'].astype(int) == season) & (df['round'].astype(int) == round_num)
        rows = df[mask]
        if rows.empty:
            return {}
        
        print(f"  [fallback] Loaded odds for {len(rows)} games from local CSV.")
        odds_map = {}
        for _, row in rows.iterrows():
            ht = canonical(str(row['home_team']))
            at = canonical(str(row['away_team']))
            h_odds = float(row.get('market_home_win_odds', 0))
            a_odds = float(row.get('market_away_win_odds', 0))
            handicap = float(row.get('market_home_handicap', 0))
            if h_odds > 1.0: # Valid odds are > 1.0
                odds_map[(ht, at)] = {
                    "home_odds": h_odds,
                    "away_odds": a_odds,
                    "handicap":  handicap
                }
        return odds_map
    except Exception as e:
        print(f"  [fallback] CSV odds read failed: {e}")
        return {}


# ─── ODDS FETCHER (optional) ──────────────────────────────────────────────────

def fetch_odds() -> dict[tuple, dict]:
    """
    Fetch NRL odds from The Odds API (https://the-odds-api.com, free tier).
    Set ODDS_API_KEY env var to enable. Returns dict keyed by (home, away) tuple.
    """
    if not ODDS_API_KEY:
        return {}
    try:
        url = (
            "https://api.the-odds-api.com/v4/sports/rugbyleague_nrl/odds/"
            f"?apiKey={ODDS_API_KEY}&regions=au&markets=h2h,spreads&oddsFormat=decimal"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        odds_map = {}
        for game in r.json():
            ht = canonical(game.get("home_team", ""))
            at = canonical(game.get("away_team", ""))
            home_odds = away_odds = handicap = None
            for book in game.get("bookmakers", []):
                for mkt in book.get("markets", []):
                    if mkt["key"] == "h2h":
                        for out in mkt["outcomes"]:
                            if canonical(out["name"]) == ht:
                                home_odds = out["price"]
                            elif canonical(out["name"]) == at:
                                away_odds = out["price"]
                    if mkt["key"] == "spreads":
                        for out in mkt["outcomes"]:
                            if canonical(out["name"]) == ht:
                                handicap = out.get("point", 0)
                if home_odds and away_odds:
                    break
            if home_odds and away_odds:
                odds_map[(ht, at)] = {
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "handicap":  handicap or 0,
                }
        return odds_map
    except Exception:
        return {}


# ─── FEATURE COMPUTATION FROM LOCAL HISTORY ───────────────────────────────────

def load_history() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df[df["winner"].notna()].copy()


def team_form(hist: pd.DataFrame, team: str, before_date, season: int) -> dict:
    """Compute all form stats for a team from historical data."""
    team_games = hist[
        ((hist["home_team"] == team) | (hist["away_team"] == team)) &
        (hist["date"] < before_date)
    ].sort_values("date")

    season_games = team_games[team_games["season"] == season]
    last5        = team_games.tail(5)

    def game_result(row, t):
        is_home = row["home_team"] == t
        pts_for  = row["home_score"] if is_home else row["away_score"]
        pts_ag   = row["away_score"] if is_home else row["home_score"]
        won      = (row["winner"] == "home") == is_home
        return pts_for, pts_ag, won

    def aggregate(games):
        if len(games) == 0:
            return {"wins": 0, "pf": 0, "pa": 0, "pd": 0}
        pf_list, pa_list, wins = [], [], 0
        for _, row in games.iterrows():
            pf, pa, w = game_result(row, team)
            pf_list.append(pf); pa_list.append(pa); wins += int(w)
        return {
            "wins": wins,
            "pf":   np.mean(pf_list),
            "pa":   np.mean(pa_list),
            "pd":   np.mean(np.array(pf_list) - np.array(pa_list)),
        }

    seas = aggregate(season_games)
    l5   = aggregate(last5)

    # Home/away record this season
    home_games = season_games[season_games["home_team"] == team]
    away_games = season_games[season_games["away_team"] == team]
    home_w = sum(home_games["winner"] == "home")
    away_w = sum(away_games["winner"] == "away")

    # Win streak
    streak = 0
    if len(team_games) > 0:
        for _, row in team_games.iloc[::-1].iterrows():
            _, _, won = game_result(row, team)
            if streak == 0:
                streak = 1 if won else -1
            elif (streak > 0) == won:
                streak += (1 if won else -1) * 0 + (1 if won else -1)
                streak = abs(streak) * (1 if won else -1)
            else:
                break

    # Days rest
    last_game_date = team_games["date"].max() if len(team_games) > 0 else None
    days_rest = (before_date - last_game_date).days if last_game_date else 7

    # Travel km — rough lookup from team home city to venue (filled best-effort)
    # (Left as 0 here; populated from data if available)

    total_played = len(season_games)
    return {
        "ladder_pos":           0,        # filled from live ladder
        "season_wins":          seas["wins"],
        "season_losses":        total_played - seas["wins"],
        "season_draws":         0,
        "season_pts_for_avg":   seas["pf"],
        "season_pts_against_avg": seas["pa"],
        "last5_wins":           l5["wins"],
        "last5_pts_for_avg":    l5["pf"],
        "last5_pts_against_avg":l5["pa"],
        "last5_pts_diff_avg":   l5["pd"],
        "home_record_wins":     home_w,
        "home_record_played":   len(home_games),
        "away_record_wins":     away_w,
        "away_record_played":   len(away_games),
        "win_streak":           streak,
        "days_rest":            days_rest,
        "travel_km":            0.0,
        # Advanced stats — averaged from last 5 in source data if columns exist
        "completion_rate":      0,
        "errors_pg":            0,
        "penalties_pg":         0,
        "tackle_eff":           0,
        "line_breaks_pg":       0,
        "tries_pg":             0,
        "post_contact_metres_pg": 0,
        "kick_metres_pg":       0,
        "key_players_out":      0,
        "origin_players_out":   0,
    }


def _fill_advanced_stats(form: dict, hist: pd.DataFrame, team: str, before_date, prefix: str) -> dict:
    """Pull advanced stats from last 5 historical rows for a team if columns exist."""
    stat_cols = [
        "completion_rate", "errors_pg", "penalties_pg", "tackle_eff",
        "line_breaks_pg", "tries_pg", "post_contact_metres_pg", "kick_metres_pg",
    ]
    team_rows = hist[
        ((hist["home_team"] == team) | (hist["away_team"] == team)) &
        (hist["date"] < before_date)
    ].tail(5)

    for col in stat_cols:
        vals = []
        for _, r in team_rows.iterrows():
            # Use the correct home/away prefix for each historical row
            p = "home" if r["home_team"] == team else "away"
            pcol = f"{p}_{col}"
            if pcol in hist.columns and pd.notna(r[pcol]) and r[pcol] != 0:
                vals.append(r[pcol])
        if vals:
            form[col] = np.mean(vals)
    return form


def h2h_stats(hist: pd.DataFrame, home: str, away: str, before_date) -> dict:
    """Compute head-to-head stats between two teams."""
    meetings = hist[
        (
            ((hist["home_team"] == home) & (hist["away_team"] == away)) |
            ((hist["home_team"] == away) & (hist["away_team"] == home))
        ) &
        (hist["date"] < before_date)
    ].sort_values("date").tail(10)

    if len(meetings) == 0:
        return {
            "h2h_home_wins": 5, "h2h_total_meetings": 10,
            "h2h_home_pts_for_avg": 20, "h2h_away_pts_for_avg": 20,
            "h2h_last3yr_home_wins": 2, "h2h_last3yr_meetings": 4,
        }

    home_wins = 0
    pf_home, pf_away = [], []
    for _, row in meetings.iterrows():
        if row["home_team"] == home:
            home_wins += int(row["winner"] == "home")
            pf_home.append(row["home_score"]); pf_away.append(row["away_score"])
        else:
            home_wins += int(row["winner"] == "away")
            pf_home.append(row["away_score"]); pf_away.append(row["home_score"])

    cutoff_3yr = before_date - timedelta(days=3 * 365)
    recent = meetings[meetings["date"] >= cutoff_3yr]
    r_home_wins = sum(
        (r["home_team"] == home and r["winner"] == "home") or
        (r["home_team"] == away and r["winner"] == "away")
        for _, r in recent.iterrows()
    )

    return {
        "h2h_home_wins":        home_wins,
        "h2h_total_meetings":   len(meetings),
        "h2h_home_pts_for_avg": np.mean(pf_home) if pf_home else 20,
        "h2h_away_pts_for_avg": np.mean(pf_away) if pf_away else 20,
        "h2h_last3yr_home_wins":r_home_wins,
        "h2h_last3yr_meetings": len(recent),
    }


# ─── BUILD FEATURE ROWS ───────────────────────────────────────────────────────

def build_game_row(
    game: dict,
    hist: pd.DataFrame,
    ladder: dict,
    odds_map: dict,
) -> dict:
    home  = game["home_team"]
    away  = game["away_team"]
    gdate = pd.to_datetime(game["date"]) if game.get("date") else datetime.now()

    hf = team_form(hist, home, gdate, game["season"])
    af = team_form(hist, away, gdate, game["season"])

    hf = _fill_advanced_stats(hf, hist, home, gdate, "home")
    af = _fill_advanced_stats(af, hist, away, gdate, "away")

    # Travel — home team travels 0, away team travels from their home city
    venue = game.get("venue", "")
    hf["travel_km"] = 0.0
    af["travel_km"] = _travel_km(away, venue)

    # Inject live ladder position
    if home in ladder:
        hf["ladder_pos"] = ladder[home]["pos"]
        hf["season_wins"] = ladder[home]["wins"]
        hf["season_losses"] = ladder[home]["losses"]
        hf["season_draws"] = ladder[home]["draws"]
        g = max(ladder[home]["wins"] + ladder[home]["losses"] + ladder[home]["draws"], 1)
        hf["season_pts_for_avg"]     = ladder[home]["pts_for"] / g
        hf["season_pts_against_avg"] = ladder[home]["pts_against"] / g
    if away in ladder:
        af["ladder_pos"] = ladder[away]["pos"]
        af["season_wins"] = ladder[away]["wins"]
        af["season_losses"] = ladder[away]["losses"]
        af["season_draws"] = ladder[away]["draws"]
        g = max(ladder[away]["wins"] + ladder[away]["losses"] + ladder[away]["draws"], 1)
        af["season_pts_for_avg"]     = ladder[away]["pts_for"] / g
        af["season_pts_against_avg"] = ladder[away]["pts_against"] / g

    h2h = h2h_stats(hist, home, away, gdate)

    odds = odds_map.get((home, away), odds_map.get((away, home), {}))
    if odds and (away, home) in odds_map and (home, away) not in odds_map:
        # odds were stored away-as-home — flip them
        home_odds = odds.get("away_odds", 0)
        away_odds = odds.get("home_odds", 0)
        handicap  = -odds.get("handicap", 0)
    else:
        home_odds = odds.get("home_odds", 0)
        away_odds = odds.get("away_odds", 0)
        handicap  = odds.get("handicap", 0)

    row = {
        "season": game["season"],
        "round":  game["round"],
        "date":   game.get("date", ""),
        "venue":  game.get("venue", ""),
        "home_team": home,
        "away_team": away,
        "home_score": None, "away_score": None, "winner": None,
        # Home stats
        "home_ladder_pos":             hf["ladder_pos"],
        "home_season_wins":            hf["season_wins"],
        "home_season_losses":          hf["season_losses"],
        "home_season_draws":           hf["season_draws"],
        "home_season_pts_for_avg":     hf["season_pts_for_avg"],
        "home_season_pts_against_avg": hf["season_pts_against_avg"],
        "home_last5_wins":             hf["last5_wins"],
        "home_last5_pts_for_avg":      hf["last5_pts_for_avg"],
        "home_last5_pts_against_avg":  hf["last5_pts_against_avg"],
        "home_last5_pts_diff_avg":     hf["last5_pts_diff_avg"],
        "home_home_record_wins":       hf["home_record_wins"],
        "home_home_record_played":     hf["home_record_played"],
        "home_win_streak":             hf["win_streak"],
        "home_days_rest":              hf["days_rest"],
        "home_travel_km":              hf["travel_km"],
        "home_completion_rate":        hf["completion_rate"],
        "home_errors_pg":              hf["errors_pg"],
        "home_penalties_pg":           hf["penalties_pg"],
        "home_tackle_eff":             hf["tackle_eff"],
        "home_line_breaks_pg":         hf["line_breaks_pg"],
        "home_tries_pg":               hf["tries_pg"],
        "home_post_contact_metres_pg": hf["post_contact_metres_pg"],
        "home_kick_metres_pg":         hf["kick_metres_pg"],
        "home_key_players_out":        hf["key_players_out"],
        "home_origin_players_out":     hf["origin_players_out"],
        # Away stats
        "away_ladder_pos":             af["ladder_pos"],
        "away_ladder_pos2":            af["ladder_pos"],
        "away_season_wins":            af["season_wins"],
        "away_season_losses":          af["season_losses"],
        "away_season_draws":           af["season_draws"],
        "away_season_pts_for_avg":     af["season_pts_for_avg"],
        "away_season_pts_against_avg": af["season_pts_against_avg"],
        "away_last5_wins":             af["last5_wins"],
        "away_last5_pts_for_avg":      af["last5_pts_for_avg"],
        "away_last5_pts_against_avg":  af["last5_pts_against_avg"],
        "away_last5_pts_diff_avg":     af["last5_pts_diff_avg"],
        "away_away_record_wins":       af["away_record_wins"],
        "away_away_record_played":     af["away_record_played"],
        "away_win_streak":             af["win_streak"],
        "away_days_rest":              af["days_rest"],
        "away_travel_km":              af["travel_km"],
        "away_completion_rate":        af["completion_rate"],
        "away_errors_pg":              af["errors_pg"],
        "away_penalties_pg":           af["penalties_pg"],
        "away_tackle_eff":             af["tackle_eff"],
        "away_line_breaks_pg":         af["line_breaks_pg"],
        "away_tries_pg":               af["tries_pg"],
        "away_post_contact_metres_pg": af["post_contact_metres_pg"],
        "away_kick_metres_pg":         af["kick_metres_pg"],
        "away_key_players_out":        af["key_players_out"],
        "away_origin_players_out":     af["origin_players_out"],
        # H2H
        **h2h,
        # Market
        "market_home_win_odds":  home_odds,
        "market_away_win_odds":  away_odds,
        "market_home_handicap":  handicap,
        "market_open_home_odds": home_odds,
        "market_open_away_odds": away_odds,
        # Weather — fetched from Open-Meteo (historical or forecast)
        **_get_weather(game.get("venue", ""), game.get("date", "")),
        # Flags
        "is_finals":           0,
        "is_elimination_final":0,
        "is_neutral_venue":    0,
        "crowd":               0,
    }
    return row


# ─── PREDICT ──────────────────────────────────────────────────────────────────

def run_predictions(games_df: pd.DataFrame, use_odds: bool = True) -> pd.DataFrame:
    """Load model, engineer features, output predictions."""
    # Import here to avoid circular dep
    sys.path.insert(0, SCRIPT_DIR)
    from m5_nrl import engineer_features, FEATURE_COLS

    model_path = MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS

    if not os.path.exists(model_path):
        if use_odds:
            # Fall back to market implied probability from odds
            def odds_prob(row):
                h, a = row["market_home_win_odds"], row["market_away_win_odds"]
                if h > 1 and a > 1:
                    hi = 1 / h; ai = 1 / a; total = hi + ai
                    return round(hi / total * 100, 1), round(ai / total * 100, 1)
                return 50.0, 50.0
            games_df = games_df.copy()
            probs = games_df.apply(odds_prob, axis=1, result_type="expand")
            games_df["home_win_prob"] = probs[0]
            games_df["away_win_prob"] = probs[1]
            games_df["predicted_winner"] = np.where(
                games_df["home_win_prob"] >= games_df["away_win_prob"],
                games_df["home_team"], games_df["away_team"]
            )
            games_df["confidence"] = np.maximum(games_df["home_win_prob"], games_df["away_win_prob"])
            return games_df
        else:
            raise FileNotFoundError(
                f"No no-odds model found at {model_path}. "
                "Use 'Retrain (No Odds)' to train it first."
            )

    with open(model_path, "rb") as f:
        model, feature_cols = pickle.load(f)

    df_feat = engineer_features(games_df)
    X = df_feat.reindex(columns=feature_cols, fill_value=0).fillna(0)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    games_df = games_df.copy()
    games_df["home_win_prob"]    = (probs * 100).round(1)
    games_df["away_win_prob"]    = ((1 - probs) * 100).round(1)
    games_df["predicted_winner"] = np.where(preds, games_df["home_team"], games_df["away_team"])
    games_df["confidence"]       = np.maximum(games_df["home_win_prob"], games_df["away_win_prob"])
    return games_df


# ─── OUTPUT ───────────────────────────────────────────────────────────────────

def print_tips(results: pd.DataFrame, round_num: int, season: int, data_source: str, use_odds: bool = True):
    results = results.sort_values("date")
    model_tag = "" if use_odds else " [no-odds model]"
    print(f"\nNRL {season} — ROUND {round_num} TIPS{model_tag}  [{data_source}]")
    print("=" * 70)

    no_model = not os.path.exists(MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS)
    if no_model:
        print("  [odds-only mode — train the model with more historical data for better picks]\n")

    for _, row in results.iterrows():
        tip  = row["predicted_winner"]
        conf = row["confidence"]
        h    = f"{row['home_team']} ({row['home_win_prob']:.0f}%)"
        a    = f"{row['away_team']} ({row['away_win_prob']:.0f}%)"
        bar  = "█" * int((conf - 50) / 5)
        flag = " ⚡" if conf >= 70 else ""
        print(f"  {h:<32}  v  {a:<32}")
        print(f"  TIP: {tip:<40} {conf:.0f}% {bar}{flag}")

        # Squad notes
        h_out = int(row.get("home_key_players_out", 0))
        a_out = int(row.get("away_key_players_out", 0))
        notes = []
        if h_out > 0:
            h_names = str(row.get("home_players_out_names", "") or "")
            detail = f" ({h_names})" if h_names else ""
            notes.append(f"{row['home_team']}: {h_out} key player{'s' if h_out>1 else ''} out{detail}")
        if a_out > 0:
            a_names = str(row.get("away_players_out_names", "") or "")
            detail = f" ({a_names})" if a_names else ""
            notes.append(f"{row['away_team']}: {a_out} key player{'s' if a_out>1 else ''} out{detail}")
        if notes:
            print(f"  ⚠  {' | '.join(notes)}")
        print()

    print("=" * 70)
    tips_list = results[["date", "home_team", "away_team", "predicted_winner", "home_win_prob", "away_win_prob", "confidence"]]
    suffix = "" if use_odds else "_no_odds"
    out_path = os.path.join(SCRIPT_DIR, f"tips_{season}_r{round_num}{suffix}.csv")

    if not os.path.exists(out_path):
        tips_list.to_csv(out_path, index=False)
        print(f"Saved → {out_path}\n")



# ─── AUTO-UPDATE LAST ROUND RESULTS ──────────────────────────────────────────

def fetch_completed_round(season: int, round_num: int) -> list[dict]:
    """Fetch a completed round's results (scores) from nrl.com."""
    games = fetch_draw_nrlcom(season, round_num)
    completed = []
    for g in games:
        # Only include games that have been played (both scores present via the API)
        raw = _fetch_raw_fixture(season, round_num, g["home_team"], g["away_team"])
        if raw:
            completed.append(raw)
    return completed


def _fetch_raw_fixture(season: int, round_num: int, home: str, away: str) -> dict | None:
    """Re-fetch a specific round from nrl.com and return completed fixture if scored."""
    try:
        r = requests.get(
            "https://www.nrl.com/draw/data",
            params={"competition": 111, "season": season, "round": round_num},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            return None
        for f in r.json().get("fixtures", []):
            ht = canonical(f.get("homeTeam", {}).get("nickName", ""))
            at = canonical(f.get("awayTeam", {}).get("nickName", ""))
            if ht != home or at != away:
                continue
            hs = f.get("homeTeam", {}).get("score")
            as_ = f.get("awayTeam", {}).get("score")
            state = f.get("matchState", "")
            if hs is None or as_ is None or state not in ("FullTime", "Post"):
                return None
            dt_str = (f.get("clock") or {}).get("kickOffTimeLong", "")
            return {
                "season":     season,
                "round":      round_num,
                "date":       dt_str[:10] if dt_str else "",
                "venue":      f.get("venue", ""),
                "home_team":  ht,
                "away_team":  at,
                "home_score": int(hs),
                "away_score": int(as_),
                "winner":     "home" if int(hs) > int(as_) else "away",
            }
    except Exception:
        pass
    return None


def auto_save_last_round(season: int, tips_round: int) -> bool:
    """
    Fetch the previous round's results, compute feature rows, append to CSV.
    Returns True if new rows were added.
    """
    last_round = tips_round - 1
    if last_round < 1:
        return False

    # Check which games from last round are already in the CSV
    existing = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
    already_saved = set()
    if len(existing):
        prev = existing[(existing["season"] == season) & (existing["round"] == last_round)]
        already_saved = set(zip(prev["home_team"], prev["away_team"]))

    print(f"Checking Round {last_round} results...")
    raw_games = fetch_draw_nrlcom(season, last_round)
    if not raw_games:
        print(f"  Could not fetch Round {last_round} from nrl.com.")
        return False

    # Filter to completed games not yet saved
    new_games = []
    for g in raw_games:
        if (g["home_team"], g["away_team"]) in already_saved:
            continue
        result = _fetch_raw_fixture(season, last_round, g["home_team"], g["away_team"])
        if result:
            new_games.append(result)

    if not new_games:
        print(f"  Round {last_round} already up to date ({len(already_saved)} games in DB).")
        return False

    print(f"  Found {len(new_games)} new result(s) — computing features and saving...")

    # Compute feature rows using current history
    hist = load_history()
    ladder = {}   # no live ladder needed for historical rows
    new_rows = []
    for g in new_games:
        row = build_game_row(g, hist, ladder, {})
        row["home_score"] = g["home_score"]
        row["away_score"] = g["away_score"]
        row["winner"]     = g["winner"]
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(DATA_PATH, mode="a", header=not os.path.exists(DATA_PATH), index=False)
    print(f"  Saved {len(new_rows)} result(s) to nrl_source_data.csv.")
    return True


def retrain(use_odds: bool = True):
    """Re-train the model in-process."""
    # Ensure we only train, do not regenerate past tips
    sys.path.insert(0, SCRIPT_DIR)
    from m5_nrl import load_data, train
    label = "with odds" if use_odds else "no odds"
    print(f"Retraining model ({label})...")
    df = load_data(DATA_PATH)
    train(df, use_odds=use_odds)
    print(f"Retraining complete ({label}).")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, default=CURRENT_SEASON)
    parser.add_argument("--round",    type=int, default=None)
    parser.add_argument("--no-odds",  action="store_true",
                        help="Use the no-odds model; skip fetching live odds")
    args = parser.parse_args()

    season   = args.season
    use_odds = not args.no_odds
    round_num = args.round or estimate_current_round(season)

    # 1. Check for existing snapshot (historical record)
    suffix = "" if use_odds else "_no_odds"
    snapshot_path = os.path.join(SCRIPT_DIR, f"tips_{season}_r{round_num}{suffix}.csv")
    if os.path.exists(snapshot_path):
        # Lock to snapshot once the first game of the round has kicked off
        games = fetch_draw_nrlcom(season, round_num)
        started = False
        if games:
            now = datetime.now().astimezone()
            for g in sorted(games, key=lambda x: x.get("kickoff") or x.get("date", "")):
                kickoff_str = g.get("kickoff") or g.get("date", "")
                if not kickoff_str:
                    continue
                try:
                    # Full ISO datetime with timezone (from API)
                    from datetime import timezone
                    ko = datetime.fromisoformat(kickoff_str)
                    if ko.tzinfo is None:
                        ko = ko.replace(tzinfo=timezone.utc)
                    if now >= ko:
                        started = True
                except ValueError:
                    # Date-only fallback (YYYY-MM-DD)
                    if kickoff_str[:10] < str(now.date()):
                        started = True
                break  # only check the first (earliest) game
        
        if started:
            print(f"\n[Snapshot found: {os.path.basename(snapshot_path)} (round has started)]")
            try:
                results = pd.read_csv(snapshot_path)
                # Add placeholders for squad info if missing in CSV
                for col in ["home_key_players_out", "away_key_players_out", "home_players_out_names", "away_players_out_names"]:
                    if col not in results.columns:
                        results[col] = 0 if "out" in col and "names" not in col else ""
                print_tips(results, round_num, season, "historical snapshot", use_odds=use_odds)
                return
            except Exception as e:
                print(f"  [warning] Could not read snapshot: {e} — regenerating...")
        else:
            print(f"\n[Snapshot found: {os.path.basename(snapshot_path)}, but round has not started. Regenerating...]")
            os.remove(snapshot_path)  # allow fresh tips to be saved

    # 2. Fetch odds (skipped in no-odds mode)
    odds_map = {}
    if use_odds:
        if round_num < estimate_current_round(season):
            # For past rounds, always prioritize historical odds from CSV
            odds_map = _fetch_odds_from_csv(season, round_num)
        
        if not odds_map and ODDS_API_KEY:
            # Try live API if no CSV odds found or if current/future round
            print("Fetching live odds...")
            odds_map = fetch_odds()
            if odds_map:
                print(f"  Live odds loaded for {len(odds_map)} games.")
        
        if not odds_map:
            # Final fallback to CSV if API failed
            odds_map = _fetch_odds_from_csv(season, round_num)
    else:
        print("  [no-odds mode — skipping live odds fetch]")

    # 3. Auto-save last round's results and retrain if anything new
    new_data = auto_save_last_round(season, round_num)
    if new_data:
        print("New results detected. Retraining models...")
        retrain(use_odds=use_odds)
    else:
        print("No new results to save. Skipping retrain.")

    print(f"\nFetching NRL {season} Round {round_num} draw...")

    # 3. Fetch draw — fall back to CSV for past rounds when API is unavailable
    games = fetch_draw_nrlcom(season, round_num)
    data_source = "nrl.com"
    if not games:
        games = _fetch_draw_from_csv(season, round_num)
        data_source = "local CSV"
    if not games:
        print("  [!] Could not fetch draw from nrl.com or local CSV")
        print(f"      You can manually create:  round{round_num}.csv  using round_template.csv")
        sys.exit(1)

    print(f"  Found {len(games)} games.")

    # 4. Fetch ladder
    print("Fetching ladder...")
    ladder = fetch_ladder_nrlcom(season)
    if ladder:
        print(f"  Ladder loaded ({len(ladder)} teams).")
    else:
        print("  Could not fetch live ladder — using local history for form stats.")

    # 5. Load local history
    hist = load_history()
    print(f"Historical games in local DB: {len(hist)}")

    # 6. Fetch squad info (key players out vs rolling baseline)
    print("Fetching squad lists for key players out...")
    squad_info = _fetch_squad_info(games, season, round_num)
    if squad_info:
        announced = sum(1 for v in squad_info.values() if v != (0, 0))
        print(f"  Squad data: {announced}/{len(squad_info)} games have announced teams.")
    else:
        print("  Squad data not yet available — key_players_out will be 0.")

    # 7. Build feature rows
    rows = []
    for g in games:
        row = build_game_row(g, hist, ladder, odds_map)
        # Inject squad data if available
        sd = squad_info.get((g["home_team"], g["away_team"]), (0, 0, "", ""))
        row["home_key_players_out"] = sd[0]
        row["away_key_players_out"] = sd[1]
        row["home_players_out_names"] = sd[2] if len(sd) > 2 else ""
        row["away_players_out_names"] = sd[3] if len(sd) > 3 else ""
        rows.append(row)
    games_df = pd.DataFrame(rows)

    # 7. Predict and print
    results = run_predictions(games_df, use_odds=use_odds)
    print_tips(results, round_num, season, data_source, use_odds=use_odds)


if __name__ == "__main__":
    main()
