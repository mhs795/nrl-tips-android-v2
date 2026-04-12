#!/usr/bin/env python3
"""
Combined data collection script for Android.
Runs history update, then fetches stats, squads, and weather for new rounds.
"""
import os
import sys
import runpy
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")

def run_step(script, args):
    print(f"\n--- Running {script} {' '.join(args)} ---")
    path = os.path.join(SCRIPT_DIR, script)
    old_argv = sys.argv
    sys.argv = [path] + args
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error in {script}: {e}")
    finally:
        sys.argv = old_argv

def main():
    # 1. Update history (get new winners/scores)
    print("Checking for new game results...")
    
    def get_existing_rounds():
        if not os.path.exists(DATA_PATH): return set()
        df = pd.read_csv(DATA_PATH)
        c = df[df['winner'].notna()]
        return set(zip(c['season'].astype(int), c['round'].astype(int)))

    before = get_existing_rounds()
    run_step("s1_history.py", ["--new-only"])
    after = get_existing_rounds()
    
    new_rounds = sorted(list(after - before))
    
    if not new_rounds:
        print("No new rounds found. Checking for missing stats/squads in recent rounds...")
        # Still check last couple of rounds just in case stats weren't ready
        import datetime
        curr_year = datetime.datetime.now().year
        # Get last 2 rounds from CSV
        df = pd.read_csv(DATA_PATH)
        last_rounds = df[df['season'] == curr_year]['round'].unique()
        new_rounds = sorted([ (curr_year, int(r)) for r in last_rounds ])[-2:]

    # 2. Update stats and squads for new/recent rounds
    for season, rnd in new_rounds:
        print(f"\nUpdating Round {rnd}, {season}...")
        run_step("s2_stats.py", ["--season", str(season), "--round", str(rnd)])
        run_step("s4_squads.py", ["--season", str(season), "--round", str(rnd)])
    
    # 3. Update weather (always fast)
    run_step("s3_weather.py", [])
    
    # 4. Update odds if key is present
    if os.environ.get("ODDS_API_KEY"):
        run_step("s5_odds.py", [])
        
    print("\nData collection complete.")

if __name__ == "__main__":
    main()
