# src/data_loader.py
import pandas as pd
import os

def load_data(data_dir='data'):
    matches_path = os.path.join(data_dir, 'matches.csv')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Drop matches with no result
    matches = matches[matches['result'] != 'no result']
    matches.dropna(subset=['winner'], inplace=True)

    return matches, deliveries

if __name__ == '__main__':
    matches, deliveries = load_data()
    print("Matches dataset:", matches.shape)
    print("Deliveries dataset:", deliveries.shape)
