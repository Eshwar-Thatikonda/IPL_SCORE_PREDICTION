# src/train_winner_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from feature_engineering import preprocess_for_winner_prediction
from data_loader import load_data

def evaluate_classification(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model: {name} | Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return accuracy, f1

def train_winner_models():
    matches, _ = load_data()
    df = preprocess_for_winner_prediction(matches)

    X = df.drop(columns=['winner'])
    y = df['winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial'),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier()
    }

    results = []
    best_model = None
    best_score = -1
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy, f1 = evaluate_classification(model, X_test, y_test, name)
        results.append({"Model": name, "Accuracy": accuracy, "F1": f1})

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    calibrated_model = CalibratedClassifierCV(best_model, cv='prefit')
    calibrated_model.fit(X_test, y_test)
    joblib.dump(calibrated_model, 'models/winner_predictor.pkl')
    print(f"\n🔥 Best Model: {best_name} (Calibrated) with F1 = {best_score:.4f}")
    pd.DataFrame(results).to_csv("results/winner_model_comparison.csv", index=False)

if __name__ == '__main__':
    train_winner_models()
