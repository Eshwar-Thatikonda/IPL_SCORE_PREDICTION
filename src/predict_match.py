# src/predict_match.py

import joblib
import pandas as pd


# Load CatBoost model and its feature list
model_bundle = joblib.load("models/score_predictor.pkl")
score_model = model_bundle["model"]
score_features = model_bundle["features"]


# Load trained models
model_bundle = joblib.load("models/score_predictor.pkl")
score_model = model_bundle["model"]
score_features = model_bundle["features"]
winner_model = joblib.load("models/winner_predictor.pkl")

# Manual label encodings (must match training)
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

venues = [
    'Wankhede Stadium', 'Eden Gardens', 'M. Chinnaswamy Stadium', 'Feroz Shah Kotla',
    'Narendra Modi Stadium', 'MA Chidambaram Stadium', 'Arun Jaitley Stadium',
    'Rajiv Gandhi Intl. Cricket Stadium', 'Sawai Mansingh Stadium'
]

team_encoder = {team: i for i, team in enumerate(teams)}
venue_encoder = {venue: i for i, venue in enumerate(venues)}

def predict_first_innings_score(batting_team, bowling_team, venue):
    # Dummy values for match progress
    balls = 120  # Full 20 overs
    wickets = 5  # Assume 5 wickets lost
    run_rate = 7.5  # Average run rate

    df = pd.DataFrame([{
        'balls': balls,
        'wickets': wickets,
        'run_rate': run_rate,
        f'venue_{venue}': 1,
        f'batting_team_{batting_team}': 1,
        f'bowling_team_{bowling_team}': 1
    }])

    # Add missing one-hot columns as 0
    full_columns = score_features
    for col in full_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[full_columns]
    predicted_score = score_model.predict(df)[0]
    return round(predicted_score)

def predict_match_winner(team1, team2, toss_winner, toss_decision, venue):
    # Encode input
    input_data = pd.DataFrame([{
        'season': 2024,
        'team1': team_encoder.get(team1, 0),
        'team2': team_encoder.get(team2, 0),
        'toss_winner': team_encoder.get(toss_winner, 0),
        'toss_decision': 0 if toss_decision == "bat" else 1,
        'venue': venue_encoder.get(venue, 0)
    }])

    winner_encoded = winner_model.predict(input_data)[0]
    predicted_team = teams[winner_encoded]
    return predicted_team

# === INTERACTIVE CLI ===
if __name__ == "__main__":
    print("\n--- IPL Match Prediction ---\n")

    team1 = input("Enter Team 1: ").strip()
    team2 = input("Enter Team 2: ").strip()
    toss_winner = input("Who won the toss? ").strip()
    toss_decision = input("Toss decision (bat/field): ").strip().lower()
    venue = input("Enter venue: ").strip()

    # Assume Team 1 bats first for score prediction
    predicted_score = predict_first_innings_score(team1, team2, venue)
    predicted_winner = predict_match_winner(team1, team2, toss_winner, toss_decision, venue)

    print("\n--- Prediction ---")
    print(f"🏏 Predicted 1st innings score for {team1}: {predicted_score} runs")
    print(f"🏆 Likely Winner: {predicted_winner}")
