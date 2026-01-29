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
