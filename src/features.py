import pandas as pd
import numpy as np
from collections import defaultdict

"""
features.py

Purpose:
Creates model-ready features from cleaned game data.

What belongs here:
- Rolling team statistics (last N games)
- Season-to-date averages
- Point margin differentials
- Home vs away comparisons
- Elo ratings or other team strength metrics

Key rule:
All features must be computed using ONLY information available
before the game date (no data leakage).

What should NOT go here:
- Model training
- File I/O (reading/writing CSVs)
- Submission formatting

This file converts cleaned historical data into numerical
features that a model can learn from.
"""
import pandas as pd
import numpy as np

def build_training_features(games: pd.DataFrame, windows=(5, 10)):
    """
    Input games columns:
      date, home_team, away_team, home_score, away_score

    Output:
      X (DataFrame features), y (Series target), meta (useful columns)
    """
    df = games.sort_values("date").reset_index(drop=True).copy()

    # target = home margin
    df["y"] = df["home_score"] - df["away_score"]

    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]]))

    # per-team rolling history (store past margins from that team's perspective)
    history = {t: [] for t in teams}

    rows = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        def roll_feats(team):
            vals = history.get(team, [])
            out = {}
            for w in windows:
                last = vals[-w:] if len(vals) > 0 else []
                out[f"margin_avg_{w}"] = float(np.mean(last)) if len(last) > 0 else 0.0
                out[f"margin_cnt_{w}"] = float(min(len(vals), w))
            return out

        home_feats = roll_feats(home)
        away_feats = roll_feats(away)

        feat = {
            "date": date,
            "home_team": home,
            "away_team": away,
        }

        # raw rolling features
        for k, v in home_feats.items():
            feat[f"home_{k}"] = v
        for k, v in away_feats.items():
            feat[f"away_{k}"] = v

        # diffs
        for w in windows:
            feat[f"diff_margin_avg_{w}"] = feat[f"home_margin_avg_{w}"] - feat[f"away_margin_avg_{w}"]
            feat[f"diff_margin_cnt_{w}"] = feat[f"home_margin_cnt_{w}"] - feat[f"away_margin_cnt_{w}"]

        rows.append((feat, row["y"]))

        # update histories AFTER generating features (prevents leakage)
        home_margin = row["home_score"] - row["away_score"]
        away_margin = -home_margin
        history[home].append(home_margin)
        history[away].append(away_margin)

    feats = pd.DataFrame([r[0] for r in rows])
    y = pd.Series([r[1] for r in rows], name="y")

    feature_cols = [c for c in feats.columns if c not in ["date", "home_team", "away_team"]]
    X = feats[feature_cols].copy()
    meta = feats[["date", "home_team", "away_team"]].copy()
    return X, y, meta, feature_cols
