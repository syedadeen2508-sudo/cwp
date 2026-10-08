"""
Cricket Win Prediction - Dataset Generator
Generates realistic synthetic IPL/T20 cricket match data for training ML models.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Teams and Venues ─────────────────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans",
    "Lucknow Super Giants"
]

VENUES = [
    "Wankhede Stadium", "MA Chidambaram Stadium", "M. Chinnaswamy Stadium",
    "Eden Gardens", "Arun Jaitley Stadium", "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium", "Rajiv Gandhi Stadium", "Narendra Modi Stadium",
    "BRSABV Ekana Cricket Stadium"
]

TOSS_DECISIONS = ["bat", "field"]

def team_strength(team: str) -> float:
    """Return a normalized strength score for each team (0.4 – 1.0)."""
    strengths = {
        "Mumbai Indians": 0.92,
        "Chennai Super Kings": 0.89,
        "Royal Challengers Bangalore": 0.78,
        "Kolkata Knight Riders": 0.82,
        "Delhi Capitals": 0.75,
        "Punjab Kings": 0.70,
        "Rajasthan Royals": 0.77,
        "Sunrisers Hyderabad": 0.74,
        "Gujarat Titans": 0.80,
        "Lucknow Super Giants": 0.72,
    }
    return strengths.get(team, 0.70)


def generate_innings_data(batting_team: str, bowling_team: str,
                          target: int = None, is_second_innings: bool = False) -> dict:
    """Simulate an innings and return relevant features."""
    bt_str = team_strength(batting_team)
    bw_str = team_strength(bowling_team)
    
    # Base score influenced by team strengths
    base_score = int(150 + (bt_str - bw_str) * 40 + np.random.normal(0, 18))
    base_score = max(90, min(250, base_score))
    
    wickets = max(0, min(10, int(np.random.normal(6, 2.5))))
    
    # Economy / run rate fluctuations
    run_rate = round(base_score / 20, 2)
    
    return {
        "score": base_score,
        "wickets": wickets,
        "run_rate": run_rate,
    }


def generate_live_match_state(batting_team: str, bowling_team: str,
                               target: int, balls_bowled: int) -> dict:
    """
    Simulate a second-innings snapshot at a given ball number (1–120).
    Returns features useful for live win prediction.
    """
    bt_str = team_strength(batting_team)
    bw_str = team_strength(bowling_team)

    overs_completed = balls_bowled // 6
    balls_in_over = balls_bowled % 6

    # Runs scored so far – influenced by strength and slight randomness
    expected_rate = (target / 120) * (1 + (bt_str - bw_str) * 0.3)
    runs_scored = int(expected_rate * balls_bowled + np.random.normal(0, 5))
    runs_scored = max(0, min(target, runs_scored))

    wickets_fallen = min(10, max(0, int(np.random.normal(
        overs_completed * 0.35 * (1 + bw_str - bt_str), 1.2))))

    runs_needed = target - runs_scored
    balls_left = 120 - balls_bowled

    crr = round(runs_scored / (balls_bowled / 6), 2) if balls_bowled > 0 else 0.0
    rrr = round(runs_needed / (balls_left / 6), 2) if balls_left > 0 else 999.0

    return {
        "runs_scored": runs_scored,
        "wickets_fallen": wickets_fallen,
        "runs_needed": runs_needed,
        "balls_left": balls_left,
        "crr": crr,
        "rrr": rrr,
    }


def calculate_win_probability(row: pd.Series) -> float:
    """
    Heuristic win probability for the batting team (chasing side).
    Returns P(batting_team wins).
    """
    bt_str = team_strength(row["batting_team"])
    bw_str = team_strength(row["bowling_team"])

    # Wickets resource remaining (10 = all in, 0 = none lost)
    wicket_resource = (10 - row["wickets_fallen"]) / 10

    # Ball resource
    ball_resource = row["balls_left"] / 120

    # Run ratio (how comfortable the chase is)
    if row["runs_needed"] <= 0:
        return 1.0
    if row["balls_left"] <= 0:
        return 0.0

    rrr = row["rrr"]
    rrr_factor = max(0, 1 - (rrr - 6) / 14)  # 6 runs/over is par; 20+ is impossible

    # Composite score
    p_win = (
        0.30 * wicket_resource
        + 0.30 * ball_resource
        + 0.25 * rrr_factor
        + 0.10 * bt_str
        - 0.05 * bw_str
    )
    p_win = float(np.clip(p_win + np.random.normal(0, 0.05), 0.02, 0.98))
    return round(p_win, 4)


def generate_dataset(n_matches: int = 5000) -> pd.DataFrame:
    """Generate a dataset of live match snapshots."""
    records = []

    for _ in range(n_matches):
        # Pick teams
        team1, team2 = np.random.choice(TEAMS, size=2, replace=False)
        venue = np.random.choice(VENUES)
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(TOSS_DECISIONS)

        # Determine batting order
        if toss_decision == "bat":
            batting_first = toss_winner
            bowling_first = team2 if toss_winner == team1 else team1
        else:
            batting_first = team2 if toss_winner == team1 else team1
            bowling_first = toss_winner

        # First innings
        first_inn = generate_innings_data(batting_first, bowling_first)
        target = first_inn["score"] + 1  # +1 for D/L or standard win rule

        # Random snapshot in the second innings (balls 6 → 119)
        balls_bowled = np.random.randint(6, 120)

        # Batting team in second innings
        batting_team = bowling_first   # roles reversed in 2nd innings
        bowling_team = batting_first

        snap = generate_live_match_state(batting_team, bowling_team, target, balls_bowled)

        row = {
            # Match metadata
            "venue": venue,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            # First innings summary
            "first_innings_score": first_inn["score"],
            "first_innings_wickets": first_inn["wickets"],
            "target": target,
            # Live state features
            "balls_bowled": balls_bowled,
            "overs_completed": balls_bowled // 6,
            "balls_in_over": balls_bowled % 6,
            "runs_scored": snap["runs_scored"],
            "wickets_fallen": snap["wickets_fallen"],
            "runs_needed": snap["runs_needed"],
            "balls_left": snap["balls_left"],
            "crr": snap["crr"],            # Current Run Rate
            "rrr": snap["rrr"],            # Required Run Rate
        }

        # Derive win probability
        p_win = calculate_win_probability(pd.Series(row))
        row["win_probability"] = p_win
        row["batting_team_wins"] = int(p_win >= 0.50)

        records.append(row)

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating cricket dataset ...")
    df = generate_dataset(n_matches=8000)
    out_path = os.path.join(os.path.dirname(__file__), "cricket_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Dataset saved -> {out_path}")
    print(f"   Shape : {df.shape}")
    print(f"\nSample rows:\n{df.head(3).to_string()}")
    print(f"\nClass distribution:\n{df['batting_team_wins'].value_counts()}")
