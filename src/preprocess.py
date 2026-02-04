from pathlib import Path
from glob import glob
import pandas as pd

RAW_DIRS = ["data/raw", "data/raw_extra"]
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "all_games.csv"
TEAM_MAP_PATH = Path("data/team_name_map.csv")


def load_team_map() -> dict:
    if not TEAM_MAP_PATH.exists():
        return {}
    df = pd.read_csv(TEAM_MAP_PATH)
    mp = {}
    for _, r in df.iterrows():
        a = str(r["from"]).strip()
        b = str(r["to"]).strip()
        if a and b:
            mp[a] = b
    return mp


def norm_team(x, mp: dict) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return mp.get(s, s)


def parse_one_file(path: str, mp: dict) -> pd.DataFrame:
    """
    Expects Sports Reference style columns like:
      Date, Visitor/Neutral, PTS, Home/Neutral, PTS, OT, Notes

    But because the two 'PTS' columns collide, pandas often renames the second to:
      PTS.1
    """
    df = pd.read_csv(path)

    # Standardize column names (trim whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    if "Date" not in df.columns or "Visitor/Neutral" not in df.columns or "Home/Neutral" not in df.columns:
        raise ValueError(f"Missing required columns in {path}: {df.columns.tolist()}")

    # Find the two score columns robustly
    # Common cases:
    #  - 'PTS' and 'PTS.1'
    #  - 'PTS' and second score accidentally named something else
    score_cols = [c for c in df.columns if c.startswith("PTS")]
    if len(score_cols) < 2:
        raise ValueError(f"Could not find two PTS columns in {path}. Found: {score_cols}")

    away_pts_col = score_cols[0]
    home_pts_col = score_cols[1]

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "away_team": df["Visitor/Neutral"].map(lambda x: norm_team(x, mp)),
            "home_team": df["Home/Neutral"].map(lambda x: norm_team(x, mp)),
            "away_score": pd.to_numeric(df[away_pts_col], errors="coerce"),
            "home_score": pd.to_numeric(df[home_pts_col], errors="coerce"),
        }
    )

    # Drop rows with missing essentials (future games, blanks, bad parses)
    out = out.dropna(subset=["date", "away_score", "home_score"])
    out = out[(out["away_team"] != "") & (out["home_team"] != "")]

    return out


def main():
    mp = load_team_map()

    files = []
    for d in RAW_DIRS:
        files.extend(glob(f"{d}/*.csv"))

    if not files:
        raise SystemExit("No raw CSV files found in data/raw or data/raw_extra.")

    frames = []
    bad = 0

    for f in sorted(files):
        try:
            frames.append(parse_one_file(f, mp))
        except Exception as e:
            bad += 1
            print(f"[SKIP] {f} -> {e}")

    if not frames:
        raise SystemExit("No valid games parsed. Check your raw CSV formats.")

    all_games = pd.concat(frames, ignore_index=True)

    # Remove duplicates (same matchup/date can appear if you add overlapping sources)
    all_games = all_games.drop_duplicates(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    # Sort chronologically
    all_games = all_games.sort_values("date").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_games.to_csv(OUT_PATH, index=False)

    print(f"Parsed files: {len(files)} | Skipped: {bad}")
    print(f"Wrote {OUT_PATH} with {len(all_games)} rows")


if __name__ == "__main__":
    main()
