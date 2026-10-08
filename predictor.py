"""
Utility: load the trained model and expose a clean predict() interface.
"""

import os
import pickle
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_metadata.pkl")


def load_model():
    """Load the best trained pipeline from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def load_metadata() -> dict:
    """Load saved training metadata (accuracy, AUC, feature importance, …)."""
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return meta


def predict(
    venue: str,
    batting_team: str,
    bowling_team: str,
    toss_winner: str,
    toss_decision: str,
    first_innings_score: int,
    first_innings_wickets: int,
    target: int,
    balls_bowled: int,
    runs_scored: int,
    wickets_fallen: int,
) -> dict:
    """
    Run inference and return a dict with:
      - win_probability  : float [0,1] – probability batting_team wins
      - batting_team_wins: int {0,1}   – predicted label
      - lose_probability : float [0,1]
    """
    model = load_model()

    overs_completed  = balls_bowled // 6
    balls_in_over    = balls_bowled % 6
    runs_needed      = max(0, target - runs_scored)
    balls_left       = max(0, 120 - balls_bowled)

    crr = round(runs_scored / (balls_bowled / 6), 2) if balls_bowled > 0 else 0.0
    rrr = round(runs_needed / (balls_left / 6), 2)   if balls_left  > 0 else 999.0

    row = pd.DataFrame([{
        "venue":                 venue,
        "batting_team":          batting_team,
        "bowling_team":          bowling_team,
        "toss_winner":           toss_winner,
        "toss_decision":         toss_decision,
        "first_innings_score":   first_innings_score,
        "first_innings_wickets": first_innings_wickets,
        "target":                target,
        "balls_bowled":          balls_bowled,
        "overs_completed":       overs_completed,
        "balls_in_over":         balls_in_over,
        "runs_scored":           runs_scored,
        "wickets_fallen":        wickets_fallen,
        "runs_needed":           runs_needed,
        "balls_left":            balls_left,
        "crr":                   crr,
        "rrr":                   rrr,
    }])

    win_prob  = float(model.predict_proba(row)[0][1])
    label     = int(model.predict(row)[0])

    return {
        "win_probability":  round(win_prob, 4),
        "lose_probability": round(1 - win_prob, 4),
        "batting_team_wins": label,
        "crr": crr,
        "rrr": rrr,
        "runs_needed": runs_needed,
        "balls_left":  balls_left,
    }
