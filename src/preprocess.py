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
import re
import pandas as pd


RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/all_games.csv")
TEAM_MAP_PATH = Path("data/team_name_map.csv")  # optional


# Common header variants across datasets
COLUMN_ALIASES = {
    "date": ["date", "game_date", "gamedate", "day", "datetime"],
    "home_team": ["home", "home_team", "home_team_name", "team_home", "home_school"],
    "away_team": ["away", "away_team", "away_team_name", "team_away", "away_school"],
    "home_score": ["home_score", "home_pts", "home_points", "pts_home", "score_home", "home_final"],
    "away_score": ["away_score", "away_pts", "away_points", "pts_away", "score_away", "away_final"],
}


REQUIRED_COLS = ["date", "home_team", "away_team", "home_score", "away_score"]


def normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def build_rename_map(df_cols):
    """
    Create a mapping from whatever the raw file uses -> our canonical names.
    """
    normalized = {c: normalize_colname(c) for c in df_cols}
    inv = {v: k for k, v in normalized.items()}  # normalized -> original (last one wins)

    rename = {}
    for canonical, variants in COLUMN_ALIASES.items():
        for v in variants:
            v_norm = normalize_colname(v)
            if v_norm in inv:
                rename[inv[v_norm]] = canonical
                break
    return rename


def basic_team_clean(name: str) -> str:
    """
    Light normalization: whitespace, punctuation, common unicode apostrophes, etc.
    Keeps it readable and close to original.
    """
    if pd.isna(name):
        return name
    s = str(name).strip()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def load_team_map():
    """
    Optional file: data/team_name_map.csv with columns:
      from,to
    Example:
      UNC,North Carolina
      St Johns,St. John's
    """
    if not TEAM_MAP_PATH.exists():
        return {}

    mdf = pd.read_csv(TEAM_MAP_PATH)
    cols = [c.lower() for c in mdf.columns]
    if "from" not in cols or "to" not in cols:
        raise ValueError("team_name_map.csv must have columns: from,to")

    # preserve original column names by index lookup
    from_col = mdf.columns[cols.index("from")]
    to_col = mdf.columns[cols.index("to")]

    mapping = {}
    for _, row in mdf.iterrows():
        frm = basic_team_clean(row[from_col])
        to = basic_team_clean(row[to_col])
        if pd.notna(frm) and pd.notna(to):
            mapping[frm] = to
    return mapping


def apply_team_map(series: pd.Series, mapping: dict) -> pd.Series:
    if not mapping:
        return series
    return series.map(lambda x: mapping.get(x, x))


def coerce_score(x):
    """
    Convert score to numeric safely.
    """
    if pd.isna(x):
        return pd.NA
    try:
        return int(float(x))
    except Exception:
        return pd.NA


def preprocess_file(path: Path, team_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = build_rename_map(df.columns)
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns after rename: {missing}")

    # Keep only required cols (you can expand later)
    df = df[REQUIRED_COLS].copy()

    # Clean teams
    df["home_team"] = df["home_team"].apply(basic_team_clean)
    df["away_team"] = df["away_team"].apply(basic_team_clean)
    df["home_team"] = apply_team_map(df["home_team"], team_map)
    df["away_team"] = apply_team_map(df["away_team"], team_map)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Coerce scores
    df["home_score"] = df["home_score"].apply(coerce_score)
    df["away_score"] = df["away_score"].apply(coerce_score)

    # Drop invalid rows
    before = len(df)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    after = len(df)

    df["source_file"] = path.name  # helpful for debugging

    print(f"{path.name}: kept {after}/{before} rows")
    return df


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError("data/raw/ does not exist. Put season CSVs in data/raw/ first.")

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError("No CSV files found in data/raw/")

    team_map = load_team_map()

    frames = []
    for f in raw_files:
        frames.append(preprocess_file(f, team_map))

    all_games = pd.concat(frames, ignore_index=True)

    # Drop duplicates (sometimes the same game appears twice across sources)
    all_games = all_games.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )

    # Sort by date for time-based training later
    all_games = all_games.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_games.to_csv(OUT_PATH, index=False)
    print(f"\nWrote combined dataset: {OUT_PATH} ({len(all_games)} rows)")


if __name__ == "__main__":
    main()
