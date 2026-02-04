import json
from pathlib import Path
import numpy as np
import pandas as pd

TEAM_MAP_PATH = Path("data/team_name_map.csv")


def load_team_map() -> dict:
    """
    data/team_name_map.csv format:
    from,to
    Miami (FL),Miami
    Pittsburgh,Pitt
    ...
    """
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


def norm_team(name: str, mp: dict) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip()
    return mp.get(s, s)


def build_features_for_games(
    games: pd.DataFrame,
    windows=(5, 10),
    elo_k: float = 20.0,
    elo_home_adv: float = 65.0,
):
    """
    games columns (canonical):
      date, home_team, away_team, home_score, away_score

    Returns:
      X, y, meta, feature_cols
    """
    df = games.copy()
    df = df.sort_values("date").reset_index(drop=True)

    team_map = load_team_map()

    df["home_team"] = df["home_team"].map(lambda x: norm_team(x, team_map))
    df["away_team"] = df["away_team"].map(lambda x: norm_team(x, team_map))

    # target: home margin
    df["y"] = df["home_score"] - df["away_score"]

    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]]))

    # rolling histories
    hist_margin = {t: [] for t in teams}
    hist_pf = {t: [] for t in teams}
    hist_pa = {t: [] for t in teams}
    last_date = {t: None for t in teams}

    # Elo
    elo = {t: 1500.0 for t in teams}

    def rest_days(team: str, cur_date: pd.Timestamp) -> float:
        d = last_date.get(team)
        if d is None:
            return 7.0
        r = (cur_date - d).days
        if r < 0:
            r = 0
        if r > 14:
            r = 14
        return float(r)

    def roll_stats(team: str):
        m = hist_margin.get(team, [])
        pf = hist_pf.get(team, [])
        pa = hist_pa.get(team, [])
        out = {}
        for w in windows:
            out[f"margin_avg_{w}"] = float(np.mean(m[-w:])) if len(m) else 0.0
            out[f"margin_std_{w}"] = float(np.std(m[-w:])) if len(m) else 0.0
            out[f"pf_avg_{w}"] = float(np.mean(pf[-w:])) if len(pf) else 0.0
            out[f"pa_avg_{w}"] = float(np.mean(pa[-w:])) if len(pa) else 0.0
            out[f"cnt_{w}"] = float(min(len(m), w))
        return out

    rows_feat = []
    rows_y = []

    for _, r in df.iterrows():
        date = r["date"]
        home = r["home_team"]
        away = r["away_team"]
        home_score = float(r["home_score"])
        away_score = float(r["away_score"])
        y = float(r["y"])

        home_roll = roll_stats(home)
        away_roll = roll_stats(away)

        feat = {
            "home_adv": 1.0,
            "home_rest": rest_days(home, date),
            "away_rest": rest_days(away, date),
        }
        feat["rest_diff"] = feat["home_rest"] - feat["away_rest"]

        # Elo features (pre-game)
        feat["home_elo"] = float(elo.get(home, 1500.0))
        feat["away_elo"] = float(elo.get(away, 1500.0))
        feat["elo_diff"] = (feat["home_elo"] + elo_home_adv) - feat["away_elo"]

        # rolling features + diffs
        for k, v in home_roll.items():
            feat[f"home_{k}"] = v
        for k, v in away_roll.items():
            feat[f"away_{k}"] = v

        for w in windows:
            feat[f"diff_margin_avg_{w}"] = feat[f"home_margin_avg_{w}"] - feat[f"away_margin_avg_{w}"]
            feat[f"diff_margin_std_{w}"] = feat[f"home_margin_std_{w}"] - feat[f"away_margin_std_{w}"]
            feat[f"diff_pf_avg_{w}"] = feat[f"home_pf_avg_{w}"] - feat[f"away_pf_avg_{w}"]
            feat[f"diff_pa_avg_{w}"] = feat[f"home_pa_avg_{w}"] - feat[f"away_pa_avg_{w}"]
            feat[f"diff_cnt_{w}"] = feat[f"home_cnt_{w}"] - feat[f"away_cnt_{w}"]

        rows_feat.append(feat)
        rows_y.append(y)

        # ---- update histories AFTER feature creation (no leakage) ----
        home_margin = home_score - away_score
        away_margin = -home_margin

        hist_margin.setdefault(home, []).append(home_margin)
        hist_margin.setdefault(away, []).append(away_margin)

        hist_pf.setdefault(home, []).append(home_score)
        hist_pa.setdefault(home, []).append(away_score)

        hist_pf.setdefault(away, []).append(away_score)
        hist_pa.setdefault(away, []).append(home_score)

        last_date[home] = date
        last_date[away] = date

        # Elo update AFTER game played
        home_elo_pre = feat["home_elo"]
        away_elo_pre = feat["away_elo"]

        exp_home = 1.0 / (1.0 + 10.0 ** (-(((home_elo_pre + elo_home_adv) - away_elo_pre) / 400.0)))
        act_home = 1.0 if home_margin > 0 else 0.0

        elo[home] = home_elo_pre + elo_k * (act_home - exp_home)
        elo[away] = away_elo_pre + elo_k * ((1.0 - act_home) - (1.0 - exp_home))

    feats = pd.DataFrame(rows_feat)
    y = pd.Series(rows_y, name="y")

    feature_cols = list(feats.columns)
    X = feats[feature_cols].copy()

    meta = df[["date", "home_team", "away_team"]].copy()
    return X, y, meta, feature_cols


def build_features_for_matchups(
    hist_games: pd.DataFrame,
    matchups: pd.DataFrame,
    windows=(5, 10),
    elo_k: float = 20.0,
    elo_home_adv: float = 65.0,
):
    """
    Builds features for future matchups using only games strictly before matchup date.

    hist_games canonical columns:
      date, home_team, away_team, home_score, away_score

    matchups columns expected:
      Date, Away, Home
    """
    team_map = load_team_map()

    hist = hist_games.copy().sort_values("date").reset_index(drop=True)
    hist["home_team"] = hist["home_team"].map(lambda x: norm_team(x, team_map))
    hist["away_team"] = hist["away_team"].map(lambda x: norm_team(x, team_map))

    subs = matchups.copy()
    subs["Date"] = pd.to_datetime(subs["Date"], errors="coerce")
    subs["Away"] = subs["Away"].map(lambda x: norm_team(x, team_map))
    subs["Home"] = subs["Home"].map(lambda x: norm_team(x, team_map))

    teams = pd.unique(pd.concat([hist["home_team"], hist["away_team"], subs["Away"], subs["Home"]]))
    hist_margin = {t: [] for t in teams}
    hist_pf = {t: [] for t in teams}
    hist_pa = {t: [] for t in teams}
    last_date = {t: None for t in teams}
    elo = {t: 1500.0 for t in teams}

    def rest_days(team: str, cur_date: pd.Timestamp) -> float:
        d = last_date.get(team)
        if d is None:
            return 7.0
        r = (cur_date - d).days
        if r < 0:
            r = 0
        if r > 14:
            r = 14
        return float(r)

    def roll_stats(team: str):
        m = hist_margin.get(team, [])
        pf = hist_pf.get(team, [])
        pa = hist_pa.get(team, [])
        out = {}
        for w in windows:
            out[f"margin_avg_{w}"] = float(np.mean(m[-w:])) if len(m) else 0.0
            out[f"margin_std_{w}"] = float(np.std(m[-w:])) if len(m) else 0.0
            out[f"pf_avg_{w}"] = float(np.mean(pf[-w:])) if len(pf) else 0.0
            out[f"pa_avg_{w}"] = float(np.mean(pa[-w:])) if len(pa) else 0.0
            out[f"cnt_{w}"] = float(min(len(m), w))
        return out

    hist_rows = hist.to_dict("records")
    i = 0

    out_feats = []

    for _, r in subs.iterrows():
        game_date = r["Date"]
        away = r["Away"]
        home = r["Home"]

        # advance history with games strictly before this matchup
        while i < len(hist_rows) and hist_rows[i]["date"] < game_date:
            g = hist_rows[i]
            hm = g["home_team"]
            aw = g["away_team"]
            hs = float(g["home_score"])
            as_ = float(g["away_score"])
            margin = hs - as_

            # update rolling histories
            hist_margin.setdefault(hm, []).append(margin)
            hist_margin.setdefault(aw, []).append(-margin)

            hist_pf.setdefault(hm, []).append(hs)
            hist_pa.setdefault(hm, []).append(as_)

            hist_pf.setdefault(aw, []).append(as_)
            hist_pa.setdefault(aw, []).append(hs)

            last_date[hm] = g["date"]
            last_date[aw] = g["date"]

            # Elo update
            home_elo_pre = float(elo.get(hm, 1500.0))
            away_elo_pre = float(elo.get(aw, 1500.0))
            exp_home = 1.0 / (1.0 + 10.0 ** (-(((home_elo_pre + elo_home_adv) - away_elo_pre) / 400.0)))
            act_home = 1.0 if margin > 0 else 0.0
            elo[hm] = home_elo_pre + elo_k * (act_home - exp_home)
            elo[aw] = away_elo_pre + elo_k * ((1.0 - act_home) - (1.0 - exp_home))

            i += 1

        home_roll = roll_stats(home)
        away_roll = roll_stats(away)

        feat = {
            "home_adv": 1.0,
            "home_rest": rest_days(home, game_date),
            "away_rest": rest_days(away, game_date),
        }
        feat["rest_diff"] = feat["home_rest"] - feat["away_rest"]

        feat["home_elo"] = float(elo.get(home, 1500.0))
        feat["away_elo"] = float(elo.get(away, 1500.0))
        feat["elo_diff"] = (feat["home_elo"] + elo_home_adv) - feat["away_elo"]

        for k, v in home_roll.items():
            feat[f"home_{k}"] = v
        for k, v in away_roll.items():
            feat[f"away_{k}"] = v

        for w in windows:
            feat[f"diff_margin_avg_{w}"] = feat[f"home_margin_avg_{w}"] - feat[f"away_margin_avg_{w}"]
            feat[f"diff_margin_std_{w}"] = feat[f"home_margin_std_{w}"] - feat[f"away_margin_std_{w}"]
            feat[f"diff_pf_avg_{w}"] = feat[f"home_pf_avg_{w}"] - feat[f"away_pf_avg_{w}"]
            feat[f"diff_pa_avg_{w}"] = feat[f"home_pa_avg_{w}"] - feat[f"away_pa_avg_{w}"]
            feat[f"diff_cnt_{w}"] = feat[f"home_cnt_{w}"] - feat[f"away_cnt_{w}"]

        out_feats.append(feat)

    X = pd.DataFrame(out_feats)
    feature_cols = list(X.columns)
    return X, feature_cols
