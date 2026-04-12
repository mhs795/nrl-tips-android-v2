"""
NRL Tipping Model — Android version.
Prediction uses pre-exported .npz model files (no sklearn required).
Training is not supported on Android — use the desktop app to retrain,
then regenerate the .npz files and rebuild the APK.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_PATH         = "nrl_model.npz"
MODEL_PATH_NO_ODDS = "nrl_model_no_odds.npz"

ODDS_COLS = ["market_home_implied_prob", "market_handicap", "odds_move"]

# ─── ROLLING STATS ────────────────────────────────────────────────────────────

_ROLLING_STAT_COLS = [
    "completion_rate", "errors_pg", "penalties_pg", "tackle_eff",
    "line_breaks_pg", "tries_pg", "post_contact_metres_pg", "kick_metres_pg",
]


def compute_rolling_stats(df, window=5):
    """Replace raw per-game stats with rolling pre-game averages (training alignment)."""
    from collections import defaultdict, deque

    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    team_history = defaultdict(lambda: deque(maxlen=window))

    # Pre-build output lists to avoid per-cell df.at[] assignment
    # (cell-by-cell assignment triggers numpy dtype inference bugs on Android)
    new_cols = {}
    for prefix in ("home", "away"):
        for col in _ROLLING_STAT_COLS:
            full_col = f"{prefix}_{col}"
            if full_col in df.columns:
                new_cols[full_col] = list(df[full_col].astype(float))

    for idx in range(n):
        row = df.iloc[idx]
        home = row["home_team"]
        away = row["away_team"]

        for prefix, team in [("home", home), ("away", away)]:
            hist = list(team_history[team])
            for col in _ROLLING_STAT_COLS:
                full_col = f"{prefix}_{col}"
                if full_col not in new_cols:
                    continue
                vals = [h[col] for h in hist if h.get(col, 0) != 0]
                new_cols[full_col][idx] = float(np.mean(vals)) if vals else 0.0

        for prefix, team in [("home", home), ("away", away)]:
            game_stats = {}
            for col in _ROLLING_STAT_COLS:
                full_col = f"{prefix}_{col}"
                val = row[full_col] if full_col in df.columns else 0.0
                game_stats[col] = float(val) if pd.notna(val) else 0.0
            team_history[team].append(game_stats)

    # Assign entire columns at once — no per-cell dtype inference
    for full_col, vals_list in new_cols.items():
        df[full_col] = vals_list

    return df


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = compute_rolling_stats(df)
    df = engineer_features(df)
    return df


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    df["ladder_diff"]      = df["away_ladder_pos"] - df["home_ladder_pos"]
    df["form_diff"]        = df["home_last5_wins"] - df["away_last5_wins"]
    df["pts_diff_diff"]    = df["home_last5_pts_diff_avg"] - df["away_last5_pts_diff_avg"]
    df["pts_for_diff"]     = df["home_season_pts_for_avg"] - df["away_season_pts_for_avg"]
    df["pts_against_diff"] = df["away_season_pts_against_avg"] - df["home_season_pts_against_avg"]
    df["rest_advantage"]   = df["home_days_rest"] - df["away_days_rest"]
    df["travel_diff"]      = df["away_travel_km"] - df["home_travel_km"]
    df["streak_diff"]      = df["home_win_streak"] - df["away_win_streak"]
    df["key_players_diff"] = df["away_key_players_out"] - df["home_key_players_out"]
    df["origin_diff"]      = df["away_origin_players_out"] - df["home_origin_players_out"]

    df["home_venue_winrate"] = np.where(
        df["home_home_record_played"] > 0,
        df["home_home_record_wins"] / df["home_home_record_played"].clip(lower=1), 0.5).astype(float)
    df["away_venue_winrate"] = np.where(
        df["away_away_record_played"] > 0,
        df["away_away_record_wins"] / df["away_away_record_played"].clip(lower=1), 0.5).astype(float)
    df["venue_winrate_diff"] = df["home_venue_winrate"] - df["away_venue_winrate"]

    df["home_season_winrate"] = (
        df["home_season_wins"] /
        (df["home_season_wins"] + df["home_season_losses"] + df["home_season_draws"]).clip(lower=1))
    df["away_season_winrate"] = (
        df["away_season_wins"] /
        (df["away_season_wins"] + df["away_season_losses"] + df["away_season_draws"]).clip(lower=1))
    df["season_winrate_diff"] = df["home_season_winrate"] - df["away_season_winrate"]

    df["h2h_home_winrate"] = np.where(
        df["h2h_total_meetings"] > 0,
        df["h2h_home_wins"] / df["h2h_total_meetings"].clip(lower=1), 0.5).astype(float)
    df["h2h_pts_diff"]      = df["h2h_home_pts_for_avg"] - df["h2h_away_pts_for_avg"]
    df["h2h_recent_winrate"] = np.where(
        df["h2h_last3yr_meetings"] > 0,
        df["h2h_last3yr_home_wins"] / df["h2h_last3yr_meetings"].clip(lower=1), 0.5).astype(float)

    df["home_form_momentum"] = (
        df["home_last5_pts_diff_avg"] -
        (df["home_season_pts_for_avg"] - df["home_season_pts_against_avg"])
    ).clip(-15, 15)
    df["away_form_momentum"] = (
        df["away_last5_pts_diff_avg"] -
        (df["away_season_pts_for_avg"] - df["away_season_pts_against_avg"])
    ).clip(-15, 15)
    df["momentum_diff"] = df["home_form_momentum"] - df["away_form_momentum"]

    df["completion_diff"]   = df["home_completion_rate"] - df["away_completion_rate"]
    df["errors_diff"]       = df["away_errors_pg"] - df["home_errors_pg"]
    df["penalties_diff"]    = df["away_penalties_pg"] - df["home_penalties_pg"]
    df["tackle_eff_diff"]   = df["home_tackle_eff"] - df["away_tackle_eff"]
    df["line_breaks_diff"]  = df["home_line_breaks_pg"] - df["away_line_breaks_pg"]
    df["post_contact_diff"] = df["home_post_contact_metres_pg"] - df["away_post_contact_metres_pg"]
    df["kick_metres_diff"]  = df["home_kick_metres_pg"] - df["away_kick_metres_pg"]

    if "market_home_win_odds" in df.columns and "market_away_win_odds" in df.columns:
        home_impl  = 1 / df["market_home_win_odds"].replace(0, np.nan)
        away_impl  = 1 / df["market_away_win_odds"].replace(0, np.nan)
        total_impl = home_impl + away_impl
        df["market_home_implied_prob"] = (home_impl / total_impl).fillna(0.5)
        df["market_handicap"]          = df["market_home_handicap"].fillna(0)
        if "market_open_home_odds" in df.columns:
            df["odds_move"] = df["market_open_home_odds"] - df["market_home_win_odds"]
        else:
            df["odds_move"] = 0.0

    if "weather_rain_mm" in df.columns:
        df["wet_weather"] = (df["weather_rain_mm"] > 2).astype(int)
    if "weather_wind_kmh" in df.columns:
        df["strong_wind"] = (df["weather_wind_kmh"] > 30).astype(int)

    df["is_finals"]        = df["is_finals"].fillna(0).astype(int) if "is_finals" in df.columns else 0
    df["is_neutral_venue"] = df["is_neutral_venue"].fillna(0).astype(int) if "is_neutral_venue" in df.columns else 0

    return df


FEATURE_COLS = [
    "ladder_diff", "form_diff", "pts_diff_diff", "pts_for_diff", "pts_against_diff",
    "rest_advantage", "travel_diff", "streak_diff", "key_players_diff", "origin_diff",
    "venue_winrate_diff", "season_winrate_diff", "momentum_diff",
    "h2h_home_winrate", "h2h_pts_diff", "h2h_recent_winrate",
    "completion_diff", "errors_diff", "penalties_diff", "tackle_eff_diff",
    "line_breaks_diff", "post_contact_diff", "kick_metres_diff",
    "market_home_implied_prob", "market_handicap", "odds_move",
    "wet_weather", "strong_wind", "is_finals", "is_neutral_venue",
    "home_last5_wins", "away_last5_wins",
    "home_form_momentum", "away_form_momentum",
    "home_season_winrate", "away_season_winrate",
    "home_win_streak", "away_win_streak",
    "home_days_rest", "away_days_rest",
    "home_ladder_pos", "away_ladder_pos",
    "home_key_players_out", "away_key_players_out",
    "home_completion_rate", "away_completion_rate",
    "home_errors_pg", "away_errors_pg",
    "home_tackle_eff", "away_tackle_eff",
    "home_line_breaks_pg", "away_line_breaks_pg",
    "home_venue_winrate", "away_venue_winrate",
]

FEATURE_COLS_NO_ODDS = [c for c in FEATURE_COLS if c not in ODDS_COLS]

# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict(predict_path, model=None, feature_cols=None, use_odds=True):
    from nrl_predict import load_model, predict_proba_home

    npz_path = MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No model found at {npz_path}.")

    npz_model    = load_model(npz_path)
    feature_cols = npz_model['feature_cols']

    df_pred   = engineer_features(pd.read_csv(predict_path, parse_dates=["date"]))
    available = [c for c in feature_cols if c in df_pred.columns]
    missing   = [c for c in feature_cols if c not in df_pred.columns]
    if missing:
        print(f"  [!] Missing columns (will use 0): {missing}")

    X_pred = df_pred.reindex(columns=feature_cols, fill_value=0).fillna(0).values
    probs  = predict_proba_home(npz_model, X_pred)
    preds  = (probs >= 0.5).astype(int)

    results = df_pred[["round", "date", "home_team", "away_team"]].copy()
    results["home_win_prob"]     = (probs * 100).round(1)
    results["away_win_prob"]     = ((1 - probs) * 100).round(1)
    results["predicted_winner"]  = np.where(preds == 1, df_pred["home_team"], df_pred["away_team"])
    results["confidence"]        = np.maximum(results["home_win_prob"], results["away_win_prob"])
    results = results.sort_values("confidence", ascending=False)

    print("\n" + "=" * 72)
    print(f"  NRL TIPPING PREDICTIONS — Round {df_pred['round'].iloc[0]}")
    print("=" * 72)
    print(f"  {'HOME':<28} {'AWAY':<28} {'TIP':<28} {'CONF':>5}")
    print("  " + "─" * 68)
    for _, row in results.iterrows():
        home_str = f"{row['home_team']} ({row['home_win_prob']}%)"
        away_str = f"{row['away_team']} ({row['away_win_prob']}%)"
        print(f"  {home_str:<28} {away_str:<28} {row['predicted_winner']:<28} {row['confidence']:>5.1f}%")
    print("=" * 72)

    out_path = predict_path.replace(".csv", "_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Predictions saved → {out_path}\n")
    return results


# ─── TRAINING (not supported on Android) ──────────────────────────────────────

def train(*args, **kwargs):
    print("\n  [!] Model training is not supported on Android.")
    print("      Use the desktop app to retrain, then rebuild the APK.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NRL Tipping Model (Android)")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", metavar="FILE")
    parser.add_argument("--no-odds", action="store_true")
    parser.add_argument("--show",    action="store_true")
    args = parser.parse_args()

    use_odds = not args.no_odds

    if args.train:
        train()

    if args.predict:
        predict(args.predict, use_odds=use_odds)

    if args.show:
        import json
        info_path = (MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS).replace(".npz", "_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print(f"  [!] No model info at {info_path}.")

    if not args.train and not args.predict and not args.show:
        parser.print_help()


if __name__ == "__main__":
    main()
