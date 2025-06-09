# src/train_score_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from feature_engineering import preprocess_for_score_prediction
from data_loader import load_data

def sanitize_columns(df):
    df.columns = df.columns.str.replace(r"[\\/:*?\"<>|,.\s()\[\]]", "_", regex=True)
    return df

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return r2, mae, rmse

def train_models():
    matches, deliveries = load_data()
    df = preprocess_for_score_prediction(matches, deliveries)
    X = df[['balls', 'wickets', 'run_rate']]
    X = sanitize_columns(X)
    X = pd.get_dummies(pd.concat([X, df[['venue', 'batting_team', 'bowling_team']]], axis=1))
    X.columns = [col.replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "") for col in X.columns]
    y = df['total_runs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    results = []
    best_model = None
    best_score = -float("inf")
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        r2, mae, rmse = evaluate_model(model, X_test, y_test, name)
        print(f"Model: {name} | R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    print(f"\n🔥 Best Model: {best_name} with R² = {best_score:.4f}")
    joblib.dump(best_model, 'models/score_predictor.pkl')
    pd.DataFrame(results).to_csv("results/score_model_comparison.csv", index=False)

        # Save model and feature names
    joblib.dump({
        "model": best_model,
        "features": list(X.columns)
    }, "models/score_predictor.pkl")

if __name__ == '__main__':
    train_models()
