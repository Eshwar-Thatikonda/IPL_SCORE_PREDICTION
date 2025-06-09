# 🏏 IPL Match Prediction System

This project is a machine learning-based system to predict:
   - **First Innings Score** of a cricket match
   - **Winning Team** based on match conditions like toss result and venue

It uses historical IPL datasets and advanced regression/classification models.

---

## 📁 Project Structure

   ├── data_loader.py # Loads and cleans IPL match and delivery data
   ├── train_score_model.py # Trains multiple regressors to predict scores
   ├── train_winner_model.py # Trains classifiers to predict match winners
   ├── predict_match.py # CLI tool to predict score and winner
   ├── evaluate.py # Visualizes model performance (R² & F1)
   ├── models/ # Saved machine learning models
   ├── results/ # CSVs and plots showing evaluation results

---

## ⚙️ Features

   - Predicts 1st innings score using team stats and venue
   - Predicts match winner based on toss, teams, and venue
   - Compares performance across multiple models (XGBoost, CatBoost, etc.)
   - CLI interface for interactive match predictions

---

## 🧠 Tech Stack

   - Python, Pandas, Scikit-learn
   - XGBoost, LightGBM, CatBoost
   - Matplotlib, Seaborn
   - Joblib for model serialization

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
Train models:

python train_score_model.py
python train_winner_model.py


Make predictions:

   python predict_match.py


Visualize model performance:

   python evaluate.py


📊 Sample Prediction

   $ python predict_match.py
   
--- IPL Match Prediction ---

Enter Team 1: Mumbai Indians
Enter Team 2: Chennai Super Kings
Who won the toss? Mumbai Indians
Toss decision (bat/field): bat
Enter venue: Wankhede Stadium

--- Prediction ---
🏏 Predicted 1st innings score for Mumbai Indians: 178 runs
🏆 Likely Winner: Mumbai Indians
