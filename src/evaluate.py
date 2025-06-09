# src/evaluate.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_score_results(csv_path):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('R2', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='R2', y='Model', data=df_sorted, palette="viridis")
    plt.title("Model R² Comparison - Score Prediction")
    plt.xlabel("R² Score")
    plt.tight_layout()
    os.makedirs("results/performance_plots", exist_ok=True)
    plt.savefig("results/performance_plots/score_r2_comparison.png")
    plt.show()

def plot_winner_results(csv_path):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('F1', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='F1', y='Model', data=df_sorted, palette="magma")
    plt.title("Model F1 Score Comparison - Winner Prediction")
    plt.xlabel("F1 Score")
    plt.tight_layout()
    plt.savefig("results/performance_plots/winner_f1_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_score_results("results/score_model_comparison.csv")
    plot_winner_results("results/winner_model_comparison.csv")
