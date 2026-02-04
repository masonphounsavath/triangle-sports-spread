"""
train.py

Purpose:
Trains a point spread prediction model using historical game data.

What this file does:
- Loads cleaned and feature-engineered data
- Splits data into training and validation sets (time-based)
- Trains one or more regression models
- Evaluates model performance using MAE or similar metrics
- Saves the best-performing model to disk

Outputs:
- A trained model file saved in the models/ directory
- (Optionally) metadata like feature column order

What should NOT go here:
- Submission CSV creation
- Hardcoded future games

This script is run whenever you want to retrain or improve the model.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from features import build_training_features

DATA_PATH = Path("data/processed/all_games.csv")
MODEL_PATH = Path("models/model.pkl")
FEATS_PATH = Path("models/feature_cols.json")

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    X, y, meta, feature_cols = build_training_features(df, windows=(5, 10))

    # time split (last 20% as validation)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATS_PATH.write_text(json.dumps(feature_cols), encoding="utf-8")

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature cols -> {FEATS_PATH}")

if __name__ == "__main__":
    main()
