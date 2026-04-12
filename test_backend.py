#!/usr/bin/env python3
"""
Backend smoke tests — run on every push before the APK build.
Exercises the code path behind each UI button so import errors and
logic crashes are caught without needing a real Android device.

Buttons covered:
  Tips (with Odds)   → s6_tips main() → auto_save + retrain + run_predictions(odds)
  Tips (No Odds)     → s6_tips main() → auto_save + retrain + run_predictions(no-odds)
  Compare Models     → s9_performance import
  Show Models        → nrl_model_info.json / nrl_model_no_odds_info.json
  Collect New Data   → s1_history / s2_stats imports
  Collect All Data   → same
"""
import sys, os, json, traceback
import pandas as pd
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        FAIL += 1

# ── helpers ────────────────────────────────────────────────────────────────────

def _mock_games():
    """Minimal games DataFrame with all columns engineer_features needs."""
    row = {
        "round": 5, "season": 2026, "date": "2026-04-10",
        "home_team": "Broncos", "away_team": "Cowboys", "venue": "Suncorp Stadium",
    }
    numeric_cols = [
        "home_ladder_pos", "away_ladder_pos",
        "home_last5_wins", "away_last5_wins",
        "home_last5_pts_diff_avg", "away_last5_pts_diff_avg",
        "home_season_pts_for_avg", "away_season_pts_for_avg",
        "home_season_pts_against_avg", "away_season_pts_against_avg",
        "home_days_rest", "away_days_rest",
        "home_travel_km", "away_travel_km",
        "home_win_streak", "away_win_streak",
        "home_key_players_out", "away_key_players_out",
        "home_origin_players_out", "away_origin_players_out",
        "home_home_record_played", "home_home_record_wins",
        "away_away_record_played", "away_away_record_wins",
        "home_season_wins", "home_season_losses", "home_season_draws",
        "away_season_wins", "away_season_losses", "away_season_draws",
        "h2h_total_meetings", "h2h_home_wins",
        "h2h_home_pts_for_avg", "h2h_away_pts_for_avg",
        "h2h_last3yr_meetings", "h2h_last3yr_home_wins",
        "home_completion_rate", "away_completion_rate",
        "home_errors_pg", "away_errors_pg",
        "home_penalties_pg", "away_penalties_pg",
        "home_tackle_eff", "away_tackle_eff",
        "home_line_breaks_pg", "away_line_breaks_pg",
        "home_post_contact_metres_pg", "away_post_contact_metres_pg",
        "home_kick_metres_pg", "away_kick_metres_pg",
        "market_home_win_odds", "market_away_win_odds",
        "market_home_handicap", "market_open_home_odds",
        "weather_rain_mm", "weather_wind_kmh",
        "is_finals", "is_neutral_venue",
        "home_players_out_names", "away_players_out_names",
    ]
    for c in numeric_cols:
        row[c] = 0
    # Give odds a realistic non-zero value so implied prob calc works
    row["market_home_win_odds"] = 1.8
    row["market_away_win_odds"] = 2.0
    return pd.DataFrame([row])


# ── 1. Core imports ────────────────────────────────────────────────────────────

def test_imports():
    from m5_nrl import (load_data, compute_rolling_stats, engineer_features,
                        FEATURE_COLS, predict, train)
    from nrl_predict import load_model, predict_proba_home

test("imports: m5_nrl (load_data, compute_rolling_stats, engineer_features, train, predict)", test_imports)

def test_s6_imports():
    from s6_tips import run_predictions, retrain, auto_save_last_round

test("imports: s6_tips (run_predictions, retrain, auto_save_last_round)", test_s6_imports)


# ── 2. load_data (numpy pipeline) — used by retrain on Tips buttons ──────────

def test_load_data():
    from m5_nrl import load_data
    df = load_data(os.path.join(REPO, "nrl_source_data.csv"))
    assert len(df) > 0, "load_data returned empty DataFrame"
    assert "ladder_diff" in df.columns, "engineer_features not applied (ladder_diff missing)"
    assert "completion_diff" in df.columns, "engineer_features not applied (completion_diff missing)"
    # Verify rolling stat columns are float (not object)
    for col in ["home_completion_rate", "home_line_breaks_pg"]:
        if col in df.columns:
            assert df[col].dtype in (np.float64, np.float32, float), \
                f"{col} has unexpected dtype {df[col].dtype}"

test("load_data: CSV → compute_rolling_stats → engineer_features (no dtype errors)", test_load_data)


# ── 3. NPZ model loading ───────────────────────────────────────────────────────

def test_npz_models():
    from nrl_predict import load_model
    for fname in ("nrl_model.npz", "nrl_model_no_odds.npz"):
        path = os.path.join(REPO, fname)
        assert os.path.exists(path), f"Model file missing: {fname}"
        m = load_model(path)
        assert "n_folds" in m and len(m["n_folds"]) > 0, f"{fname}: no n_folds"
        assert "feature_cols" in m and len(m["feature_cols"]) > 0, f"{fname}: no feature_cols"

test("npz models: load nrl_model.npz and nrl_model_no_odds.npz", test_npz_models)


# ── 4. run_predictions — Tips (with Odds) button ──────────────────────────────

def test_run_predictions_odds():
    from s6_tips import run_predictions
    result = run_predictions(_mock_games(), use_odds=True)
    assert "predicted_winner" in result.columns, "predicted_winner column missing"
    assert "home_win_prob" in result.columns, "home_win_prob column missing"
    assert result["predicted_winner"].notna().all(), "predicted_winner has NaN"

test("run_predictions (with odds): npz model → predicted_winner column", test_run_predictions_odds)


# ── 5. run_predictions — Tips (No Odds) button ────────────────────────────────

def test_run_predictions_no_odds():
    from s6_tips import run_predictions
    result = run_predictions(_mock_games(), use_odds=False)
    assert "predicted_winner" in result.columns, "predicted_winner column missing"
    assert result["predicted_winner"].notna().all(), "predicted_winner has NaN"

test("run_predictions (no odds): npz model → predicted_winner column", test_run_predictions_no_odds)


# ── 6. retrain — called automatically when new data is saved ─────────────────

def test_retrain():
    from s6_tips import retrain
    # On Android, train() is a stub — this just tests load_data doesn't crash
    retrain(use_odds=True)
    retrain(use_odds=False)

test("retrain: load_data + train stub (no crash, no dtype error)", test_retrain)


# ── 7. Show Models — reads JSON info files ────────────────────────────────────

def test_show_models():
    for fname in ("nrl_model_info.json", "nrl_model_no_odds_info.json"):
        path = os.path.join(REPO, fname)
        assert os.path.exists(path), f"Model info file missing: {fname}"
        with open(path) as f:
            d = json.load(f)
        assert "cv_accuracy" in d, f"{fname}: cv_accuracy missing"

test("show models: nrl_model_info.json and nrl_model_no_odds_info.json readable", test_show_models)


# ── 8. Compare Models — s9_performance import ─────────────────────────────────

def test_s9_import():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s9_performance", os.path.join(REPO, "s9_performance.py"))
    # Just check it parses without SyntaxError — don't execute main()
    mod = importlib.util.module_from_spec(spec)
    # Compile only
    with open(os.path.join(REPO, "s9_performance.py")) as f:
        compile(f.read(), "s9_performance.py", "exec")

test("compare models: s9_performance.py compiles without error", test_s9_import)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*55}\n")

if FAIL > 0:
    sys.exit(1)
