"""
NRL Tipping Model — HistGradientBoosting-based winner predictor
Usage:
  python nrl_model.py --train                   # train on historical data
  python nrl_model.py --predict round.csv       # predict upcoming round
  python nrl_model.py --train --predict round.csv
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_PATH         = "nrl_source_data.csv"
MODEL_PATH        = "nrl_model.pkl"
MODEL_PATH_NO_ODDS = "nrl_model_no_odds.pkl"

# Odds-derived features excluded from the no-odds model
ODDS_COLS = ["market_home_implied_prob", "market_handicap", "odds_move"]

NRL_TEAMS = [
    "Brisbane Broncos", "Canberra Raiders", "Canterbury Bulldogs",
    "Cronulla Sharks", "Gold Coast Titans", "Manly Sea Eagles",
    "Melbourne Storm", "Newcastle Knights", "New Zealand Warriors",
    "North Queensland Cowboys", "Parramatta Eels", "Penrith Panthers",
    "South Sydney Rabbitohs", "St George Illawarra Dragons",
    "Sydney Roosters", "Wests Tigers", "Dolphins", "Dolphins FC",
]

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw data columns."""
    df = df.copy()

    # --- Differential features (home minus away) ---
    df["ladder_diff"]          = df["away_ladder_pos"] - df["home_ladder_pos"]   # positive = home higher
    df["form_diff"]            = df["home_last5_wins"] - df["away_last5_wins"]
    df["pts_diff_diff"]        = df["home_last5_pts_diff_avg"] - df["away_last5_pts_diff_avg"]
    df["pts_for_diff"]         = df["home_season_pts_for_avg"] - df["away_season_pts_for_avg"]
    df["pts_against_diff"]     = df["away_season_pts_against_avg"] - df["home_season_pts_against_avg"]
    df["rest_advantage"]       = df["home_days_rest"] - df["away_days_rest"]
    df["travel_diff"]          = df["away_travel_km"] - df["home_travel_km"]
    df["streak_diff"]          = df["home_win_streak"] - df["away_win_streak"]
    df["key_players_diff"]     = df["away_key_players_out"] - df["home_key_players_out"]
    df["origin_diff"]          = df["away_origin_players_out"] - df["home_origin_players_out"]

    # --- Home ground win rate ---
    df["home_venue_winrate"]   = np.where(
        df["home_home_record_played"] > 0,
        df["home_home_record_wins"] / df["home_home_record_played"].clip(lower=1),
        0.5
    ).astype(float)
    df["away_venue_winrate"]   = np.where(
        df["away_away_record_played"] > 0,
        df["away_away_record_wins"] / df["away_away_record_played"].clip(lower=1),
        0.5
    ).astype(float)
    df["venue_winrate_diff"]   = df["home_venue_winrate"] - df["away_venue_winrate"]

    # --- Season win rate ---
    df["home_season_winrate"]  = (
        df["home_season_wins"] /
        (df["home_season_wins"] + df["home_season_losses"] + df["home_season_draws"]).clip(lower=1)
    )
    df["away_season_winrate"]  = (
        df["away_season_wins"] /
        (df["away_season_wins"] + df["away_season_losses"] + df["away_season_draws"]).clip(lower=1)
    )
    df["season_winrate_diff"]  = df["home_season_winrate"] - df["away_season_winrate"]

    # --- Head-to-head home win rate ---
    df["h2h_home_winrate"]     = np.where(
        df["h2h_total_meetings"] > 0,
        df["h2h_home_wins"] / df["h2h_total_meetings"].clip(lower=1),
        0.5
    ).astype(float)
    df["h2h_pts_diff"]         = df["h2h_home_pts_for_avg"] - df["h2h_away_pts_for_avg"]
    df["h2h_recent_winrate"]   = np.where(
        df["h2h_last3yr_meetings"] > 0,
        df["h2h_last3yr_home_wins"] / df["h2h_last3yr_meetings"].clip(lower=1),
        0.5
    ).astype(float)

    # --- Form momentum (last 5 pts diff vs season baseline, capped to avoid outlier blowouts) ---
    df["home_form_momentum"] = (
        df["home_last5_pts_diff_avg"] -
        (df["home_season_pts_for_avg"] - df["home_season_pts_against_avg"])
    ).clip(-15, 15)
    df["away_form_momentum"] = (
        df["away_last5_pts_diff_avg"] -
        (df["away_season_pts_for_avg"] - df["away_season_pts_against_avg"])
    ).clip(-15, 15)
    df["momentum_diff"] = df["home_form_momentum"] - df["away_form_momentum"]

    # --- Advanced stats differentials ---
    df["completion_diff"]      = df["home_completion_rate"] - df["away_completion_rate"]
    df["errors_diff"]          = df["away_errors_pg"] - df["home_errors_pg"]           # lower errors = better
    df["penalties_diff"]       = df["away_penalties_pg"] - df["home_penalties_pg"]
    df["tackle_eff_diff"]      = df["home_tackle_eff"] - df["away_tackle_eff"]
    df["line_breaks_diff"]     = df["home_line_breaks_pg"] - df["away_line_breaks_pg"]
    df["post_contact_diff"]    = df["home_post_contact_metres_pg"] - df["away_post_contact_metres_pg"]
    df["kick_metres_diff"]     = df["home_kick_metres_pg"] - df["away_kick_metres_pg"]

    # --- Market implied probability ---
    if "market_home_win_odds" in df.columns and "market_away_win_odds" in df.columns:
        home_impl = 1 / df["market_home_win_odds"].replace(0, np.nan)
        away_impl = 1 / df["market_away_win_odds"].replace(0, np.nan)
        total_impl = home_impl + away_impl
        df["market_home_implied_prob"] = (home_impl / total_impl).fillna(0.5)
        df["market_handicap"]          = df["market_home_handicap"].fillna(0)
        # line movement signal
        if "market_open_home_odds" in df.columns:
            df["odds_move"] = df["market_open_home_odds"] - df["market_home_win_odds"]
        else:
            df["odds_move"] = 0.0

    # --- Weather impact ---
    if "weather_rain_mm" in df.columns:
        df["wet_weather"] = (df["weather_rain_mm"] > 2).astype(int)
    if "weather_wind_kmh" in df.columns:
        df["strong_wind"] = (df["weather_wind_kmh"] > 30).astype(int)

    # --- Context flags ---
    df["is_finals"]            = df["is_finals"].fillna(0).astype(int) if "is_finals" in df.columns else 0
    df["is_neutral_venue"]     = df["is_neutral_venue"].fillna(0).astype(int) if "is_neutral_venue" in df.columns else 0

    return df


FEATURE_COLS = [
    # Differential / relative
    "ladder_diff", "form_diff", "pts_diff_diff", "pts_for_diff", "pts_against_diff",
    "rest_advantage", "travel_diff", "streak_diff", "key_players_diff", "origin_diff",
    "venue_winrate_diff", "season_winrate_diff", "momentum_diff",
    "h2h_home_winrate", "h2h_pts_diff", "h2h_recent_winrate",
    # Advanced stats
    "completion_diff", "errors_diff", "penalties_diff", "tackle_eff_diff",
    "line_breaks_diff", "post_contact_diff", "kick_metres_diff",
    # Market
    "market_home_implied_prob", "market_handicap", "odds_move",
    # Context
    "wet_weather", "strong_wind", "is_finals", "is_neutral_venue",
    # Raw individual (model learns interactions)
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

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

_ROLLING_STAT_COLS = [
    "completion_rate", "errors_pg", "penalties_pg", "tackle_eff",
    "line_breaks_pg", "tries_pg", "post_contact_metres_pg", "kick_metres_pg",
]

def compute_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Replace raw per-game advanced stats with rolling pre-game averages.

    s2_stats.py stores the actual stats achieved *in* each game.  The model
    must instead see what the team averaged in the N games *before* this one,
    because that is what s6_tips.py supplies at prediction time.

    For each game (sorted by date) we:
      1. Write the rolling mean of the team's last `window` games into the row.
      2. Append this game's original actual stats to the team's history buffer.
    """
    from collections import defaultdict, deque

    df = df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    team_history: dict = defaultdict(lambda: deque(maxlen=window))

    # Pre-build output lists to avoid per-cell df.at[] assignment
    new_cols: dict = {}
    for prefix in ("home", "away"):
        for col in _ROLLING_STAT_COLS:
            full_col = f"{prefix}_{col}"
            if full_col in df.columns:
                new_cols[full_col] = list(df[full_col].astype(float))

    for idx in range(n):
        row = df.iloc[idx]
        home = row["home_team"]
        away = row["away_team"]

        # 1. Overwrite with pre-game rolling averages
        for prefix, team in [("home", home), ("away", away)]:
            hist = list(team_history[team])
            for col in _ROLLING_STAT_COLS:
                full_col = f"{prefix}_{col}"
                if full_col not in new_cols:
                    continue
                vals = [h[col] for h in hist if h.get(col, 0) != 0]
                new_cols[full_col][idx] = float(np.mean(vals)) if vals else 0.0

        # 2. Add this game's actual stats to each team's history buffer
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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = compute_rolling_stats(df)   # align training with prediction-time features
    df = engineer_features(df)
    return df


def prepare_training(df: pd.DataFrame):
    """Return X, y for labelled games (winner column populated)."""
    labelled = df[df["winner"].notna()].copy()
    labelled["target"] = (labelled["winner"] == "home").astype(int)

    # Keep only columns that exist in FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in labelled.columns]
    X = labelled[available].fillna(0)
    y = labelled["target"]
    return X, y, available


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────

def build_model():
    base = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=3,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )
    # Isotonic calibration gives better probability estimates
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    return model


def train(df: pd.DataFrame, use_odds: bool = True):
    feat_cols = FEATURE_COLS if use_odds else FEATURE_COLS_NO_ODDS
    save_path = MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS
    label     = "with odds" if use_odds else "no odds"

    labelled = df[df["winner"].notna()].copy()
    labelled["target"] = (labelled["winner"] == "home").astype(int)
    available = [c for c in feat_cols if c in labelled.columns]
    X = labelled[available].fillna(0)
    y = labelled["target"]

    if len(X) < 20:
        print(f"  [!] Only {len(X)} labelled games — model will have low confidence. Add more historical data.")

    model = build_model()

    # Cross-validate first
    raw_model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42,
    )
    cv_scores = cross_val_score(raw_model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring="accuracy")
    print(f"\n  [{label}] Cross-validated accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit calibrated model on full data
    model.fit(X, y)

    # In-sample metrics
    preds     = model.predict(X)
    probs     = model.predict_proba(X)[:, 1]
    acc       = accuracy_score(y, preds)
    ll        = log_loss(y, probs)
    brier     = brier_score_loss(y, probs)
    print(f"  Training accuracy:        {acc:.3f}")
    print(f"  Log loss:                 {ll:.4f}")
    print(f"  Brier score:              {brier:.4f}  (lower = better; random ≈ 0.25)")

    # Feature importance from the underlying estimators
    importances = np.zeros(len(available))
    for est in model.calibrated_classifiers_:
        importances += est.estimator.feature_importances_
    importances /= len(model.calibrated_classifiers_)
    imp_df = pd.DataFrame({"feature": available, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
    print("\n  Top 15 features:")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<35} {bar} ({row['importance']:.4f})")

    with open(save_path, "wb") as f:
        pickle.dump((model, available), f)

    import json, datetime
    info_path = save_path.replace(".pkl", "_info.json")
    info = {
        "trained_at":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cv_accuracy":     round(float(cv_scores.mean()), 4),
        "cv_std":          round(float(cv_scores.std()), 4),
        "train_accuracy":  round(float(acc), 4),
        "brier_score":     round(float(brier), 4),
        "features": [
            {"name": row["feature"], "importance": round(float(row["importance"]), 4)}
            for _, row in imp_df.iterrows()
        ],
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model saved → {save_path}")
    return model, available


# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict(predict_path: str, model=None, feature_cols=None, use_odds: bool = True):
    model_path = MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}. Run --train first.")
        with open(model_path, "rb") as f:
            model, feature_cols = pickle.load(f)

    # Prediction CSVs already have rolling averages computed by s6_tips.py,
    # so skip compute_rolling_stats (which needs full history to work correctly).
    df_pred = engineer_features(pd.read_csv(predict_path, parse_dates=["date"]))
    available = [c for c in feature_cols if c in df_pred.columns]
    missing   = [c for c in feature_cols if c not in df_pred.columns]
    if missing:
        print(f"  [!] Missing columns (will use 0): {missing}")

    X_pred = df_pred[available].reindex(columns=feature_cols, fill_value=0).fillna(0)

    probs  = model.predict_proba(X_pred)[:, 1]   # P(home wins)
    preds  = (probs >= 0.5).astype(int)

    results = df_pred[["round", "date", "home_team", "away_team"]].copy()
    results["home_win_prob"]  = (probs * 100).round(1)
    results["away_win_prob"]  = ((1 - probs) * 100).round(1)
    results["predicted_winner"] = np.where(preds == 1, df_pred["home_team"], df_pred["away_team"])
    results["confidence"]     = np.maximum(results["home_win_prob"], results["away_win_prob"])
    results = results.sort_values("confidence", ascending=False)

    print("\n" + "=" * 72)
    print(f"  NRL TIPPING PREDICTIONS — Round {df_pred['round'].iloc[0]}")
    print("=" * 72)
    header = f"  {'HOME':<28} {'AWAY':<28} {'TIP':<28} {'CONF':>5}"
    print(header)
    print("  " + "─" * 68)
    for _, row in results.iterrows():
        home_str = f"{row['home_team']} ({row['home_win_prob']}%)"
        away_str = f"{row['away_team']} ({row['away_win_prob']}%)"
        conf_str = f"{row['confidence']:.1f}%"
        tip      = row["predicted_winner"]
        marker   = "◀" if tip == row["home_team"] else "  ▶"
        print(f"  {home_str:<28} {away_str:<28} {tip:<28} {conf_str:>6}")
    print("=" * 72)

    out_path = predict_path.replace(".csv", "_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Predictions saved → {out_path}\n")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def show_model(use_odds: bool = True):
    model_path = MODEL_PATH if use_odds else MODEL_PATH_NO_ODDS
    label      = "with odds" if use_odds else "no odds"

    if not os.path.exists(model_path):
        print(f"  [!] No model found at {model_path}. Run Retrain first.")
        return

    mtime = os.path.getmtime(model_path)
    import datetime
    trained_at = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    size_kb    = os.path.getsize(model_path) / 1024

    with open(model_path, "rb") as f:
        model, feature_cols = pickle.load(f)

    print(f"\n=== Model Info ({label}) ===")
    print(f"  File:           {model_path}")
    print(f"  Last trained:   {trained_at}")
    print(f"  File size:      {size_kb:.1f} KB")
    print(f"  Features used:  {len(feature_cols)}")

    # Feature importances from calibrated classifiers
    importances = np.zeros(len(feature_cols))
    for est in model.calibrated_classifiers_:
        importances += est.estimator.feature_importances_
    importances /= len(model.calibrated_classifiers_)
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
    print("\n  Top 15 features:")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<35} {bar} ({row['importance']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="NRL Tipping Model")
    parser.add_argument("--train",    action="store_true", help="Train model on nrl_source_data.csv")
    parser.add_argument("--predict",  metavar="FILE",      help="Predict winners for an upcoming round CSV")
    parser.add_argument("--no-odds",  action="store_true", help="Train/predict without market odds features")
    parser.add_argument("--show",     action="store_true", help="Show info about saved model without retraining")
    args = parser.parse_args()

    use_odds = not args.no_odds
    model, feature_cols = None, None

    if args.train:
        print("\n[TRAIN]")
        df = load_data(DATA_PATH)
        print(f"  Loaded {len(df)} games from {DATA_PATH}")
        model, feature_cols = train(df, use_odds=use_odds)

    if args.predict:
        print("\n[PREDICT]")
        predict(args.predict, model, feature_cols, use_odds=use_odds)

    if args.show:
        show_model(use_odds=use_odds)

    if not args.train and not args.predict and not args.show:
        parser.print_help()


if __name__ == "__main__":
    main()
