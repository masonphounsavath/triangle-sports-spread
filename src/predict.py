from pathlib import Path
import json
import pandas as pd
import joblib
import numpy as np

from features import build_features_for_matchups

MODEL_PATH = Path("models/model.pkl")
FEATS_PATH = Path("models/feature_cols.json")
HIST_PATH = Path("data/processed/all_games.csv")
TEMPLATE_PATH = Path("data/submission_template.csv")
OUT_PATH = Path("submissions/submission.csv")

def main():
    model = joblib.load(MODEL_PATH)
    trained_cols = json.loads(FEATS_PATH.read_text(encoding="utf-8"))

    hist = pd.read_csv(HIST_PATH, parse_dates=["date"])
    sub = pd.read_csv(TEMPLATE_PATH)

    X, cols = build_features_for_matchups(
        hist_games=hist,
        matchups=sub,
        windows=(5, 10),
        elo_k=20.0,
        elo_home_adv=65.0
    )

    # Align columns to what the model was trained on
    X = X.reindex(columns=trained_cols, fill_value=0.0)

    preds = model.predict(X)

    # Practical sanity: avoid insane lines
    preds = np.clip(preds, -30, 30)

    # Round to nearest half point
    preds = np.round(preds * 2) / 2

    # Remove negative zero
    preds = np.where(preds == -0.0, 0.0, preds)

    sub["pt_spread"] = preds
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
