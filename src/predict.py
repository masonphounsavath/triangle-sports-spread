from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

# Import your project's feature builder.
# This must exist in src/features.py
from features import build_features_for_matchups

MODEL_PATH = Path("models/model.pkl")
FEATS_PATH = Path("models/feature_cols.json")
HIST_PATH = Path("data/processed/all_games.csv")
TEMPLATE_PATH = Path("data/submission_template.csv")
OUT_PATH = Path("submissions/submission.csv")
TEAM_MAP_PATH = Path("data/team_name_map.csv")


def load_team_map(path: Path) -> dict:
    """Load a from->to team name mapping if present."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "from" not in df.columns or "to" not in df.columns:
        return {}
    return {str(a).strip(): str(b).strip() for a, b in zip(df["from"], df["to"])}


def norm_team(name: str, mp: dict) -> str:
    """Normalize a team name using the mapping dict."""
    if name is None:
        return ""
    s = str(name).strip()
    return mp.get(s, s)


def require(path: Path, msg: str):
    if not path.exists():
        raise SystemExit(msg)


def main():
    # Ensure required artifacts exist
    require(MODEL_PATH, "models/model.pkl not found. Run: python src/train.py")
    require(FEATS_PATH, "models/feature_cols.json not found. Run: python src/train.py")
    require(HIST_PATH, "data/processed/all_games.csv not found. Run: python src/preprocess.py")
    require(TEMPLATE_PATH, "data/submission_template.csv not found.")

    team_map = load_team_map(TEAM_MAP_PATH)

    # Load model + trained feature columns
    model = joblib.load(MODEL_PATH)
    trained_cols = json.loads(FEATS_PATH.read_text(encoding="utf-8"))

    # Load history
    hist = pd.read_csv(HIST_PATH, parse_dates=["date"])
    # Normalize team names in history
    hist["home_team"] = hist["home_team"].map(lambda x: norm_team(x, team_map))
    hist["away_team"] = hist["away_team"].map(lambda x: norm_team(x, team_map))

    # Load submission template
    template = pd.read_csv(TEMPLATE_PATH)

    # Normalize template team names
    template["Home"] = template["Home"].astype(str).str.strip().map(lambda x: norm_team(x, team_map))
    template["Away"] = template["Away"].astype(str).str.strip().map(lambda x: norm_team(x, team_map))

    # Drop blank rows if any
    template = template.replace({"Home": {"nan": np.nan, "": np.nan}, "Away": {"nan": np.nan, "": np.nan}})
    template = template.dropna(subset=["Home", "Away"]).reset_index(drop=True)

    # Build features for matchups
    # NOTE: This must match your features.py function signature.
    X, _ = build_features_for_matchups(
        hist_games=hist,
        matchups=template,
        windows=(5, 10),
        elo_k=20.0,
        elo_home_adv=65.0,
    )

    # Ensure exact column order & fill missing cols with 0
    X = X.reindex(columns=trained_cols, fill_value=0.0)

    # Predict raw margins/spreads
    preds = model.predict(X).astype(float)

    # Identify out-of-domain teams (not present in historical ACC training set)
    hist_teams = set(hist["home_team"]) | set(hist["away_team"])
    ood_mask = ~template["Home"].isin(hist_teams) | ~template["Away"].isin(hist_teams)

    # Dampen ONLY those OOD games (prevents Baylor/OSU/Michigan blowing up)
    if ood_mask.any():
        ood_idx = np.where(ood_mask.values)[0]
        preds[ood_idx] = np.clip(preds[ood_idx] * 0.55, -16.0, 16.0)

    # Global safety clip
    preds = np.clip(preds, -40.0, 40.0)

    # Round to nearest half-point (basketball spread style)
    preds = np.round(preds, 3)

    # Avoid -0.0
    preds = np.where(preds == -0.0, 0.0, preds)

    # Write submission
    template["pt_spread"] = preds
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH}")

    # Debug prints (helpful, doesn’t aaffect submission file)
    if ood_mask.any():
        ood_games = template.loc[ood_mask, ["Away", "Home", "pt_spread"]]
        print("\nOut-of-domain games damped (no history for at least one team):")
        print(ood_games.to_string(index=False))

    zeros = template[template["pt_spread"] == 0]
    if len(zeros) > 0:
        print("\nNote: 0.0 spreads found (might indicate missing/neutral features):")
        print(zeros[["Away", "Home", "pt_spread"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
