from pathlib import Path
import pandas as pd

from sportsipy.ncaab.schedule import Schedule  # sports-reference scraper via sportsipy

# Put the seasons you want here (10 seasons: 2016–2025)
SEASONS = list(range(2016, 2026))

OUT_DIR = Path("data/raw_extra")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sportsipy uses sports-reference "school slugs"
# We'll hard-map the 3 missing teams (most reliable)
TEAM_SLUGS = {
    "Baylor": "baylor",
    "Michigan": "michigan",
    "Ohio State": "ohio-state",
}

def fetch_team_season(team_name: str, slug: str, season: int) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
    date, Visitor/Neutral, PTS, Home/Neutral, PTS, OT, Notes
    """
    sched = Schedule(slug, season=season).dataframe

    # sportsipy schedule columns vary; we normalize defensively
    # Typically includes: date, opponent_name, location, points_for, points_against, overtime
    # location: 'Home', 'Away', 'Neutral'
    df = sched.copy()

    # Clean date
    df["date"] = pd.to_datetime(df["date"])

    # Figure out home/away orientation
    # sportsipy uses 'location' where Away means your team played away at opponent
    # We'll create Visitor/Neutral and Home/Neutral based on location.
    def visitor(row):
        if row.get("location") in ["Home"]:
            return row.get("opponent_name")
        return team_name  # Away/Neutral: your team is visitor

    def home(row):
        if row.get("location") in ["Home"]:
            return team_name
        return row.get("opponent_name")  # Away/Neutral: opponent is home

    def visitor_pts(row):
        if row.get("location") in ["Home"]:
            return row.get("points_against")
        return row.get("points_for")

    def home_pts(row):
        if row.get("location") in ["Home"]:
            return row.get("points_for")
        return row.get("points_against")

    out = pd.DataFrame({
        "Date": df["date"].dt.strftime("%a %b %d %Y"),
        "Visitor/Neutral": df.apply(visitor, axis=1),
        "PTS": df.apply(visitor_pts, axis=1),
        "Home/Neutral": df.apply(home, axis=1),
        "PTS.1": df.apply(home_pts, axis=1),
        "OT": df.get("overtime", ""),
        "Notes": "",
    })

    # Drop rows without scores (future games)
    out = out.dropna(subset=["PTS", "PTS.1"])
    return out

def main():
    for team_name, slug in TEAM_SLUGS.items():
        rows = []
        for season in SEASONS:
            try:
                df_season = fetch_team_season(team_name, slug, season)
                rows.append(df_season)
                print(f"Fetched {team_name} {season}: {len(df_season)} games")
            except Exception as e:
                print(f"FAILED {team_name} {season}: {e}")

        if rows:
            out = pd.concat(rows, ignore_index=True)
            out_path = OUT_DIR / f"extra_{team_name.replace(' ', '_').lower()}_2016_2025.csv"
            out.to_csv(out_path, index=False)
            print(f"Wrote {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
