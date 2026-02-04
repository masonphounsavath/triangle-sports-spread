from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from features import build_features_for_matchups

MODEL_PATH = Path("models/model.pkl")
FEATS_PATH = Path("models/feature_cols.json")
HIST_PATH = Path("data/processed/all_games.csv")
TEMPLATE_PATH = Path("data/submission_template.csv")
OUT_PATH = Path("submissions/submission.csv")


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("models/model.pkl not found. Run: python src/train.py")
    if not FEATS_PATH.exists():
        raise SystemExit("models/feature_cols.json not found. Run: python src/train.py")
    if not HIST_PATH.exists():
        raise SystemExit("data/processed/all_games.csv not found. Run: python src/preprocess.py")
    if not TEMPLATE_PATH.exists():
        raise SystemExit("data/submission_template.csv not found.")

    model = joblib.load(MODEL_PATH)
    trained_cols = json.loads(FEATS_PATH.read_text(encoding="utf-8"))

    hist = pd.read_csv(HIST_PATH, parse_dates=["date"])
    sub = pd.read_csv(TEMPLATE_PATH)

    # Fix the "nan" issue: drop rows with missing Home/Away
    # Also strip whitespace to avoid " " becoming a fake team name
    sub["Home"] = sub["Home"].astype(str).str.strip()
    sub["Away"] = sub["Away"].astype(str).str.strip()
    sub = sub.replace({"Home": {"nan": np.nan, "": np.nan}, "Away": {"nan": np.nan, "": np.nan}})
    sub = sub.dropna(subset=["Home", "Away"]).reset_index(drop=True)

    X, _ = build_features_for_matchups(
        hist_games=hist,
        matchups=sub,
        windows=(5, 10),
        elo_k=20.0,
        elo_home_adv=65.0,
    )

    # Align to training columns
    X = X.reindex(columns=trained_cols, fill_value=0.0)

    preds = model.predict(X)

    # sanity clip
    preds = np.clip(preds, -30, 30)

    # round to nearest half-point
    preds = np.round(preds * 2) / 2

    # remove negative zero
    preds = np.where(preds == -0.0, 0.0, preds)

    sub["pt_spread"] = preds

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
