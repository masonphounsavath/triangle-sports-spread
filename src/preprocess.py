"""
preprocess.py

Combines season-level raw CSVs into one canonical training dataset.

Input:
- data/raw/*.csv  (one file per season, or any number of raw files)

Output:
- data/processed/all_games.csv with columns:
  date, home_team, away_team, home_score, away_score

Notes:
- This script is intentionally strict: rows missing required fields are dropped.
- Team names are normalized to reduce mismatches across seasons.
- Optional: add data/team_name_map.csv to force specific name mappings.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/all_games.csv")

def load_one_season(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Sports Reference format: Date, Visitor/Neutral, PTS, Home/Neutral, PTS, ...
    df = df.iloc[:, :5]  # we only need first 5 columns
    df.columns = ["date", "away_team", "away_score", "home_team", "home_score"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")

    df = df.dropna(subset=["date", "away_score", "home_score"])

    return df[["date", "home_team", "away_team", "home_score", "away_score"]]

def main():
    all_games = []

    for path in sorted(RAW_DIR.glob("*.csv")):
        print(f"Processing {path.name}")
        season_df = load_one_season(path)
        all_games.append(season_df)

    df = pd.concat(all_games, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} games to {OUT_PATH}")

if __name__ == "__main__":
    main()
