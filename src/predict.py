import pandas as pd
from pathlib import Path

TEMPLATE_PATH = Path("data/submission_template.csv")
OUT_PATH = Path("submissions/submission.csv")

def main():
    df = pd.read_csv(TEMPLATE_PATH)

    # make sure the column exists
    if "pt_spread" not in df.columns:
        raise ValueError("Template is missing 'pt_spread' column.")

    # placeholder predictions for now
    df["pt_spread"] = 0.0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()

number = 0

"""
predict.py

Purpose:
Generates spread predictions for upcoming games and creates
a submission-ready CSV file.

What this file does:
- Loads the official submission template CSV
- Loads the trained model from disk
- Builds features for the games in the template
- Predicts point spreads
- Writes a new CSV with pt_spread filled in

Key rule:
The original submission template is treated as read-only.
This script always writes a NEW submission file.

This is the final step before submitting predictions.
"""
