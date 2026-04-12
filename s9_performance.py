#!/usr/bin/env python3
"""
NRL Tips — performance checker.
Finds the most recent tips CSV that has actual results available in
nrl_source_data.csv and prints a match-by-match breakdown plus summary.

Usage:
  python s9_performance.py              # auto-detect last completed round
  python s9_performance.py --round 4   # specific round
  python s9_performance.py --season 2025 --round 4
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")


def _find_tips_files(no_odds: bool = False):
    """Return list of (season, round, path) sorted newest-first."""
    pattern = os.path.join(SCRIPT_DIR, "tips_*_r*.csv")
    results = []
    for path in glob.glob(pattern):
        name = os.path.basename(path).lower()
        if no_odds:
            m = re.search(r"tips_(\d{4})_r(\d+)_no_odds\.csv$", name)
        else:
            m = re.search(r"tips_(\d{4})_r(\d+)\.csv$", name)
            # Exclude no_odds files when looking for standard tips
            if m and name.endswith("_no_odds.csv"):
                m = None
        if m:
            results.append((int(m.group(1)), int(m.group(2)), path))
    results.sort(reverse=True)
    return results


def _load_actual(season: int, round_num: int) -> pd.DataFrame:
    """Return rows from source data for the given season/round."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    mask = (df["season"] == season) & (df["round"] == round_num)
    return df[mask][["home_team", "away_team", "home_score", "away_score", "winner", "date"]].copy()


def check_performance(season: int | None = None, round_num: int | None = None, no_odds: bool = False):
    tips_files = _find_tips_files(no_odds=no_odds)
    if not tips_files:
        print("No tips CSV files found.")
        sys.exit(1)

    # Auto-detect: walk newest-first, pick first one with actual results
    candidates = tips_files
    if season is not None:
        candidates = [(s, r, p) for s, r, p in candidates if s == season]
    if round_num is not None:
        candidates = [(s, r, p) for s, r, p in candidates if r == round_num]

    chosen = None
    for s, r, path in candidates:
        actual = _load_actual(s, r)
        if not actual.empty:
            chosen = (s, r, path, actual)
            break

    if chosen is None:
        if season or round_num:
            print(f"No results found in source data for the requested round.")
        else:
            print("No completed rounds found in source data yet — check back after the round is done.")
        sys.exit(0)

    s, r, path, actual = chosen
    tips = pd.read_csv(path)

    # Sort tips by game date (schedule order) using dates from source data
    tips = tips.merge(
        actual[["home_team", "away_team", "date"]],
        on=["home_team", "away_team"],
        how="left",
    ).sort_values("date").drop(columns=["date"])

    model_label = "no odds" if no_odds else "with odds"
    print(f"\nRound {r} · {s}  [{model_label}]\n")

    correct = 0
    total   = 0

    for _, tip in tips.iterrows():
        home = tip["home_team"]
        away = tip["away_team"]
        pred = tip["predicted_winner"]
        conf = tip.get("confidence", tip.get("home_win_prob", "?"))

        row = actual[
            (actual["home_team"] == home) & (actual["away_team"] == away)
        ]
        if row.empty:
            print(f"{home} vs {away}\n  Tipped: {pred}  —  result not found\n")
            continue

        row = row.iloc[0]
        actual_winner = home if row["winner"] == "home" else away
        home_score    = int(row["home_score"])
        away_score    = int(row["away_score"])

        hit    = pred == actual_winner
        correct += int(hit)
        total  += 1
        marker  = "✔" if hit else "✘"

        print(f"{marker} {home} {home_score} – {away_score} {away}")
        print(f"  Tipped: {pred}  ({conf:.0f}%)  →  {actual_winner} won\n")

    if total:
        pct = correct / total * 100
        print(f"RESULT: {correct}/{total} correct  ({pct:.0f}%)\n")
    else:
        print("No matched games found.\n")


def _current_round(season: int) -> int:
    """Find the next upcoming NRL round by querying the NRL.com draw API.
    Scans rounds until we find one that has at least one future game.
    Falls back to CSV-based estimate if the API is unreachable.
    """
    import datetime as _dt
    import requests as _req
    today = _dt.datetime.now().date()

    # NRL.com draw endpoint headers
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nrl.com/"
    }

    # Search rounds to find the first one that hasn't completed yet
    for rnd in range(1, 28):
        try:
            r = _req.get(
                "https://www.nrl.com/draw/data",
                params={"competition": 111, "season": season, "round": rnd},
                headers=headers, timeout=6,
            )
            if r.status_code != 200: continue
            fixtures = r.json().get("fixtures", [])
            if not fixtures: continue
            
            # If ANY game in this round is today or in the future, this is our round
            dates = []
            for f in fixtures:
                d = (f.get("clock", {}).get("kickOffTimeLong") or f.get("matchDate", ""))[:10]
                if d: dates.append(d)
            
            if any(d >= str(today) for d in dates):
                return rnd
        except Exception:
            break

    # Fallback: max completed round + 1 from CSV
    if not os.path.exists(DATA_PATH):
        return 1
    try:
        df = pd.read_csv(DATA_PATH, usecols=['season', 'round', 'winner'])
        completed = df[(df['season'].astype(int) == season) & df['winner'].notna()]
        return int(completed['round'].max()) + 1 if not completed.empty else 1
    except Exception:
        return 1


def compare_models(season: int | None = None, round_num: int | None = None):
    """Side-by-side comparison of the odds and no-odds models for the same round."""
    import datetime as _dt
    cur_season = season or _dt.datetime.now().year

    odds_files    = _find_tips_files(no_odds=False)
    no_odds_files = _find_tips_files(no_odds=True)

    # Build lookup: (season, round) → path
    odds_map    = {(s, r): p for s, r, p in odds_files}
    no_odds_map = {(s, r): p for s, r, p in no_odds_files}

    # Determine which round to compare — prefer the requested/current round,
    # then fall back to the newest round that has both CSVs.
    target_round = round_num or _current_round(cur_season)
    key = (cur_season, target_round)

    has_odds    = key in odds_map
    has_no_odds = key in no_odds_map

    # Fall back only if we don't have BOTH for the target round
    if not (has_odds and has_no_odds):
        # Fall back to the newest round with both CSVs
        found = False
        for s, r, odds_path in sorted(odds_files, reverse=True):
            if (s, r) in no_odds_map:
                key = (s, r)
                has_odds = has_no_odds = True
                cur_season, target_round = s, r
                found = True
                break
        
        if not found:
            print("No tips CSV files found for either model.")
            print("Generate tips using 'Tips (with Odds)' and 'Tips (No Odds)' first.")
            sys.exit(0)

    chosen = (cur_season, target_round, odds_map[key], no_odds_map[key], _load_actual(cur_season, target_round))

    s, r, odds_path, no_odds_path, actual = chosen
    tips_odds    = pd.read_csv(odds_path)
    tips_no_odds = pd.read_csv(no_odds_path)

    # Sort by game date if we have actual data and date column exists
    if not actual.empty and "date" in tips_odds.columns and "date" in actual.columns:
        # Perform merge
        tips_odds = tips_odds.merge(
            actual[["home_team", "away_team", "date"]], on=["home_team", "away_team"], how="left"
        )
        # Verify date exists after merge
        if "date" in tips_odds.columns:
            tips_odds = tips_odds.sort_values("date").drop(columns=["date"])


    partial = actual.empty or actual["winner"].isna().any()
    header_note = "  (partial)" if partial else ""
    print(f"\nRound {r} · {s}  — Model Comparison{header_note}\n")

    odds_correct = odds_total = no_odds_correct = no_odds_total = 0
    agree_correct = agree_total = disagree_odds_correct = disagree_total = 0

    for _, tip in tips_odds.iterrows():
        home = tip["home_team"]
        away = tip["away_team"]

        pred_odds = tip["predicted_winner"]
        conf_odds = tip.get("confidence", tip.get("home_win_prob", 0))

        no_odds_row = tips_no_odds[
            (tips_no_odds["home_team"] == home) & (tips_no_odds["away_team"] == away)
        ]
        if no_odds_row.empty:
            continue
        no_odds_row  = no_odds_row.iloc[0]
        pred_no_odds = no_odds_row["predicted_winner"]
        conf_no_odds = no_odds_row.get("confidence", no_odds_row.get("home_win_prob", 0))

        agree     = pred_odds == pred_no_odds
        agree_str = "  [agree]" if agree else "  [DIFFER]"

        actual_row = actual[(actual["home_team"] == home) & (actual["away_team"] == away)] if not actual.empty else pd.DataFrame()
        has_result = not actual_row.empty and pd.notna(actual_row.iloc[0].get("winner"))

        print(f"{home} vs {away}")
        print(f"  Odds:    {pred_odds} ({conf_odds:.0f}%)")
        print(f"  No Odds: {pred_no_odds} ({conf_no_odds:.0f}%)  {agree_str.strip()}")

        if has_result:
            actual_row    = actual_row.iloc[0]
            actual_winner = home if actual_row["winner"] == "home" else away
            home_score    = int(actual_row["home_score"])
            away_score    = int(actual_row["away_score"])

            hit_odds    = pred_odds    == actual_winner
            hit_no_odds = pred_no_odds == actual_winner

            odds_correct    += int(hit_odds);    odds_total    += 1
            no_odds_correct += int(hit_no_odds); no_odds_total += 1

            mk_odds    = "✔" if hit_odds    else "✘"
            mk_no_odds = "✔" if hit_no_odds else "✘"

            if agree:
                agree_total   += 1
                agree_correct += int(hit_odds)
            else:
                disagree_total        += 1
                disagree_odds_correct += int(hit_odds)

            print(f"  Result: {mk_odds}/{mk_no_odds}  {home} {home_score} – {away_score} {away}  →  {actual_winner}\n")
        else:
            print(f"  Result: not yet played\n")

    if odds_total:
        o_pct  = odds_correct    / odds_total    * 100
        n_pct  = no_odds_correct / no_odds_total * 100
        winner = "with odds" if odds_correct > no_odds_correct else ("no odds" if no_odds_correct > odds_correct else "tie")
        print(f"Odds:    {odds_correct}/{odds_total} correct  ({o_pct:.0f}%)")
        print(f"No Odds: {no_odds_correct}/{no_odds_total} correct  ({n_pct:.0f}%)")
        print(f"Winner:  {winner}")
        if agree_total:
            a_pct = agree_correct / agree_total * 100
            print(f"Agreed {agree_total} games → {agree_correct}/{agree_total} correct ({a_pct:.0f}%)")
        if disagree_total:
            d_pct = disagree_odds_correct / disagree_total * 100
            print(f"Differed {disagree_total} games → odds got {disagree_odds_correct}/{disagree_total} ({d_pct:.0f}%)")
    elif odds_total == 0 and no_odds_total == 0:
        print("No results yet — check back after games are played.\n")


def main():
    parser = argparse.ArgumentParser(description="Check NRL tips performance")
    parser.add_argument("--round",    type=int, default=None)
    parser.add_argument("--season",   type=int, default=None)
    parser.add_argument("--no-odds",  action="store_true", help="Check no-odds model tips")
    parser.add_argument("--compare",  action="store_true", help="Compare odds vs no-odds model")
    args = parser.parse_args()
    if args.compare:
        compare_models(season=args.season, round_num=args.round)
    else:
        check_performance(season=args.season, round_num=args.round, no_odds=args.no_odds)


if __name__ == "__main__":
    main()
