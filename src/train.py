from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_features_for_games

DATA_PATH = Path("data/processed/all_games.csv")
MODEL_PATH = Path("models/model.pkl")
FEATS_PATH = Path("models/feature_cols.json")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # Safety: ensure chronological order for rolling features / Elo
    df = df.sort_values("date").reset_index(drop=True)

    # Build features
    X, y, meta, feature_cols = build_features_for_games(
        df,
        windows=(5, 10),
        elo_k=20.0,
        elo_home_adv=65.0
    )

    # Time split (last 20% validation)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # MAE-optimized model (trains directly for absolute error)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_depth=7,
        min_samples_leaf=40,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

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
