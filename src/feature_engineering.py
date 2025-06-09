# src/feature_engineering.py
import pandas as pd
from data_loader import load_data
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_for_score_prediction(matches, deliveries):
    # Filter first innings deliveries
    first_innings = deliveries[deliveries['inning'] == 1].copy()

    # Merge match info
    merged = first_innings.merge(matches[['id', 'season', 'venue']], left_on='match_id', right_on='id')

    # Aggregate features
    grouped = merged.groupby(['match_id', 'batting_team', 'bowling_team', 'venue']).agg(
        total_runs=('total_runs', 'sum'),
        wickets=('player_dismissed', lambda x: x.notna().sum()),
        balls=('ball', 'count')
    ).reset_index()

    # Compute over-wise run rate
    grouped['run_rate'] = grouped['total_runs'] / (grouped['balls'] / 6)

    return grouped

def preprocess_for_winner_prediction(matches):
    df = matches.copy()

    # Keep only relevant columns
    features = df[['season', 'team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']].dropna()

    # Extract year from 'season' (handles strings like '2020/21' or plain integers)
    features['season'] = features['season'].astype(str).str.extract(r'(\\d{4}|\d{4})')
    features = features.dropna(subset=['season'])  # Drop if year not found
    features['season'] = features['season'].astype(int)

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']:
        features[col] = encoder.fit_transform(features[col])

    return features

if __name__ == '__main__':
    matches, deliveries = load_data()
    score_df = preprocess_for_score_prediction(matches, deliveries)
    winner_df = preprocess_for_winner_prediction(matches)

    print("Score prediction dataset shape:", score_df.shape)
    print("Winner prediction dataset shape:", winner_df.shape)
