import os
import pickle
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
import xgboost as xgb


def identify_bot(nickname: pd.Series) -> pd.Series:
    """Identify if a nickname corresponds to a known Woogles bot."""
    nick_clean = nickname.fillna("").astype(str)
    is_bot = (
        nick_clean.str.endswith("Bot")
        | nick_clean.str.endswith("bot")
        | nick_clean.isin(["BB-8", "HastyBot", "BetterBot", "STEEBot"])
    )
    return is_bot.astype(int)


def get_bot_category(nickname: pd.Series) -> pd.Series:
    """Map nickname to a categorical bot ID."""
    bot_map = {
        "HastyBot": 1,
        "BetterBot": 2,
        "STEEBot": 3,
        "BB-8": 4,
    }
    nick_clean = nickname.fillna("").astype(str)
    cat = nick_clean.map(bot_map).fillna(0).astype(int)
    is_other_bot = (cat == 0) & (nick_clean.str.lower().str.endswith("bot"))
    cat = np.where(is_other_bot, 5, cat)
    return cat


def reconstruct_game_pairs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct two-player game relationships across train and test metadata."""
    meta_train = train_df[["game_id", "nickname", "score"]].copy()
    meta_test = test_df[["game_id", "nickname", "score"]].copy()
    all_games = pd.concat([meta_train, meta_test], axis=0).reset_index(drop=True)

    all_games["row_within_game"] = all_games.groupby("game_id").cumcount()
    g_p1 = all_games[all_games["row_within_game"] == 0].drop(
        columns=["row_within_game"]
    )
    g_p2 = all_games[all_games["row_within_game"] == 1].drop(
        columns=["row_within_game"]
    )

    p1_to_p2 = pd.merge(g_p1, g_p2, on="game_id", suffixes=("", "_opp"))
    p2_to_p1 = pd.merge(g_p2, g_p1, on="game_id", suffixes=("", "_opp"))
    game_pairs = pd.concat([p1_to_p2, p2_to_p1], axis=0).reset_index(drop=True)
    return game_pairs


def extract_unique_match_pairs(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract single-directed unique match pairs across all games."""
    all_games = pd.concat(
        [
            train_df[["game_id", "nickname", "score"]],
            test_df[["game_id", "nickname", "score"]],
        ],
        axis=0,
    ).reset_index(drop=True)

    all_games["row_within_game"] = all_games.groupby("game_id").cumcount()
    g_p1 = all_games[all_games["row_within_game"] == 0].drop(
        columns=["row_within_game"]
    )
    g_p2 = all_games[all_games["row_within_game"] == 1].drop(
        columns=["row_within_game"]
    )
    pairs = pd.merge(g_p1, g_p2, on="game_id", suffixes=("_1", "_2"))
    return pairs


def solve_graph_rating_propagation(
    pairs_df: pd.DataFrame,
    nick2idx: dict,
    anchor_ratings: dict,
    global_mean: float,
    beta: float = 2.85,
    match_weight: float = 1.0,
    anchor_weight_base: float = 0.5,
    anchor_weight_per_game: float = 0.1,
) -> dict:
    """Solve regularized graph Dirichlet rating propagation over match score incidence matrix."""
    n_players = len(nick2idx)
    idx1 = pairs_df["nickname_1"].map(nick2idx).values
    idx2 = pairs_df["nickname_2"].map(nick2idx).values
    valid_mask = (~np.isnan(idx1)) & (~np.isnan(idx2))

    idx1 = idx1[valid_mask].astype(int)
    idx2 = idx2[valid_mask].astype(int)
    score_diffs = (
        pairs_df.loc[valid_mask, "score_1"] - pairs_df.loc[valid_mask, "score_2"]
    ).values.astype(np.float64)
    n_games = len(idx1)

    game_rows = np.repeat(np.arange(n_games), 2)
    game_cols = np.column_stack((idx1, idx2)).flatten()
    game_vals = np.tile([np.sqrt(match_weight), -np.sqrt(match_weight)], n_games)
    b_games = np.sqrt(match_weight) * beta * score_diffs

    anchor_rows = np.arange(n_players) + n_games
    anchor_cols = np.arange(n_players)
    anchor_vals = np.zeros(n_players, dtype=np.float64)
    b_anchors = np.zeros(n_players, dtype=np.float64)

    for nick, idx in nick2idx.items():
        if nick in anchor_ratings:
            mean_r, count = anchor_ratings[nick]
            w = anchor_weight_base + anchor_weight_per_game * count
            anchor_vals[idx] = np.sqrt(w)
            b_anchors[idx] = np.sqrt(w) * mean_r
        else:
            w = 0.1
            anchor_vals[idx] = np.sqrt(w)
            b_anchors[idx] = np.sqrt(w) * global_mean

    all_rows = np.concatenate([game_rows, anchor_rows])
    all_cols = np.concatenate([game_cols, anchor_cols])
    all_vals = np.concatenate([game_vals, anchor_vals])
    b = np.concatenate([b_games, b_anchors])

    A = sparse.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_games + n_players, n_players),
        dtype=np.float64,
    )
    AtA = A.T @ A
    Atb = A.T @ b

    r_sol = sparse.linalg.spsolve(AtA, Atb)
    nick_to_rating = {nick: float(r_sol[idx]) for nick, idx in nick2idx.items()}
    return nick_to_rating


def compute_graph_skills(
    train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = 5.0
) -> pd.DataFrame:
    """Compute graph-based global player skill ratings by solving Ridge regression on match incidence matrix."""
    meta_train = train_df[["game_id", "nickname", "score"]].copy()
    meta_test = test_df[["game_id", "nickname", "score"]].copy()
    all_games = pd.concat([meta_train, meta_test], axis=0).reset_index(drop=True)

    all_games["row_within_game"] = all_games.groupby("game_id").cumcount()
    g_p1 = all_games[all_games["row_within_game"] == 0].drop(
        columns=["row_within_game"]
    )
    g_p2 = all_games[all_games["row_within_game"] == 1].drop(
        columns=["row_within_game"]
    )

    pairs = pd.merge(g_p1, g_p2, on="game_id", suffixes=("_1", "_2"))

    unique_nicks = np.unique(all_games["nickname"].dropna().values)
    nick2idx = {nick: i for i, nick in enumerate(unique_nicks)}
    n_players = len(unique_nicks)

    idx1 = pairs["nickname_1"].map(nick2idx).values
    idx2 = pairs["nickname_2"].map(nick2idx).values

    valid_mask = (~np.isnan(idx1)) & (~np.isnan(idx2))
    idx1 = idx1[valid_mask].astype(int)
    idx2 = idx2[valid_mask].astype(int)
    pairs_valid = pairs[valid_mask].reset_index(drop=True)
    n_games = len(pairs_valid)

    rows = np.repeat(np.arange(n_games), 2)
    cols = np.column_stack((idx1, idx2)).flatten()
    vals = np.tile([1.0, -1.0], n_games)

    A = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_games, n_players), dtype=np.float32
    )

    y_diff = (pairs_valid["score_1"] - pairs_valid["score_2"]).values.astype(
        np.float32
    )
    y_win = np.sign(y_diff).astype(np.float32)

    ridge_diff = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg")
    ridge_diff.fit(A, y_diff)
    skill_diff = ridge_diff.coef_

    ridge_win = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg")
    ridge_win.fit(A, y_win)
    skill_win = ridge_win.coef_

    graph_skills_df = pd.DataFrame(
        {
            "nickname": unique_nicks,
            "graph_skill_score_diff": skill_diff,
            "graph_skill_win_margin": skill_win,
        }
    )
    return graph_skills_df


def compute_match_features(df: pd.DataFrame, game_pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent and match-level differential features."""
    df_merged = pd.merge(
        df,
        game_pairs[["game_id", "nickname", "nickname_opp", "score_opp"]],
        on=["game_id", "nickname"],
        how="left",
    )

    df_merged["score_opp"] = df_merged["score_opp"].fillna(df_merged["score"])
    df_merged["nickname_opp"] = df_merged["nickname_opp"].fillna("Unknown")

    score = df_merged["score"].values
    score_opp = df_merged["score_opp"].values
    diff = score - score_opp
    total = score + score_opp

    df_merged["score_diff"] = diff
    df_merged["score_ratio"] = score / (total + 1e-5)
    df_merged["total_score"] = total
    df_merged["is_win"] = (diff > 0).astype(float)
    df_merged["is_loss"] = (diff < 0).astype(float)
    df_merged["is_tie"] = (diff == 0).astype(float)
    df_merged["score_diff_sign_sqrt"] = np.sign(diff) * np.sqrt(np.abs(diff))
    df_merged["is_blowout"] = (np.abs(diff) >= 150).astype(float)
    df_merged["is_close_game"] = (np.abs(diff) <= 30).astype(float)
    df_merged["is_high_scoring"] = (score >= 420).astype(float)
    df_merged["is_low_scoring"] = (score <= 320).astype(float)
    df_merged["score_gt_450"] = (score >= 450).astype(float)
    df_merged["score_gt_500"] = (score >= 500).astype(float)
    df_merged["score_lt_300"] = (score <= 300).astype(float)

    df_merged["is_bot"] = identify_bot(df_merged["nickname"])
    df_merged["is_opp_bot"] = identify_bot(df_merged["nickname_opp"])
    df_merged["opp_bot_type"] = get_bot_category(df_merged["nickname_opp"])

    return df_merged


def compute_player_aggregations(all_df: pd.DataFrame):
    """Compute aggregate player statistics across all observed gameplay."""
    p_grp = all_df.groupby("nickname")

    win_games = all_df[all_df["is_win"] == 1]
    loss_games = all_df[all_df["is_loss"] == 1]
    win_grp = win_games.groupby("nickname")["score_diff"].agg(
        player_mean_win_margin="mean",
        player_median_win_margin="median",
    ).reset_index()
    loss_grp = loss_games.groupby("nickname")["score_diff"].agg(
        player_mean_loss_margin="mean",
        player_median_loss_margin="median",
    ).reset_index()

    p_stats = p_grp.agg(
        player_games_count=("game_id", "count"),
        player_mean_score=("score", "mean"),
        player_std_score=("score", "std"),
        player_min_score=("score", "min"),
        player_max_score=("score", "max"),
        player_median_score=("score", "median"),
        player_p5_score=("score", lambda x: np.percentile(x, 5)),
        player_p10_score=("score", lambda x: np.percentile(x, 10)),
        player_q25_score=("score", lambda x: np.percentile(x, 25)),
        player_q75_score=("score", lambda x: np.percentile(x, 75)),
        player_p90_score=("score", lambda x: np.percentile(x, 90)),
        player_p95_score=("score", lambda x: np.percentile(x, 95)),
        player_mean_diff=("score_diff", "mean"),
        player_std_diff=("score_diff", "std"),
        player_win_rate=("is_win", "mean"),
        player_loss_rate=("is_loss", "mean"),
        player_mean_total_score=("total_score", "mean"),
        player_std_total_score=("total_score", "std"),
        player_p10_total_score=("total_score", lambda x: np.percentile(x, 10)),
        player_p90_total_score=("total_score", lambda x: np.percentile(x, 90)),
        player_mean_conceded=("score_opp", "mean"),
        player_std_conceded=("score_opp", "std"),
        player_median_conceded=("score_opp", "median"),
        player_p10_conceded=("score_opp", lambda x: np.percentile(x, 10)),
        player_p90_conceded=("score_opp", lambda x: np.percentile(x, 90)),
        player_blowout_rate=("is_blowout", "mean"),
        player_close_game_rate=("is_close_game", "mean"),
        player_gt_450_rate=("score_gt_450", "mean"),
        player_gt_500_rate=("score_gt_500", "mean"),
        player_lt_300_rate=("score_lt_300", "mean"),
    ).reset_index()

    p_stats = pd.merge(p_stats, win_grp, on="nickname", how="left")
    p_stats = pd.merge(p_stats, loss_grp, on="nickname", how="left")
    p_stats["player_mean_win_margin"] = p_stats["player_mean_win_margin"].fillna(0.0)
    p_stats["player_median_win_margin"] = p_stats["player_median_win_margin"].fillna(0.0)
    p_stats["player_mean_loss_margin"] = p_stats["player_mean_loss_margin"].fillna(0.0)
    p_stats["player_median_loss_margin"] = p_stats["player_median_loss_margin"].fillna(0.0)

    p_stats["player_std_score"] = p_stats["player_std_score"].fillna(0.0)
    p_stats["player_std_diff"] = p_stats["player_std_diff"].fillna(0.0)
    p_stats["player_std_total_score"] = p_stats["player_std_total_score"].fillna(0.0)
    p_stats["player_std_conceded"] = p_stats["player_std_conceded"].fillna(0.0)

    p_stats["player_iqr_score"] = p_stats["player_q75_score"] - p_stats["player_q25_score"]
    p_stats["player_cv_score"] = p_stats["player_std_score"] / (p_stats["player_mean_score"] + 1e-5)
    p_stats["player_p90_p10_range"] = p_stats["player_p90_score"] - p_stats["player_p10_score"]
    p_stats["player_skew_score"] = (p_stats["player_mean_score"] - p_stats["player_median_score"]) / (
        p_stats["player_std_score"] + 1e-5
    )
    p_stats["player_offensive_defensive_ratio"] = (
        p_stats["player_mean_score"] / (p_stats["player_mean_conceded"] + 1e-5)
    )

    # Bot-vs-Human aggregate profiles
    bot_games = all_df[all_df["is_opp_bot"] == 1]
    bot_grp = bot_games.groupby("nickname")
    bot_stats = bot_grp.agg(
        player_vs_bot_count=("game_id", "count"),
        player_vs_bot_mean_score=("score", "mean"),
        player_vs_bot_mean_diff=("score_diff", "mean"),
        player_vs_bot_win_rate=("is_win", "mean"),
    ).reset_index()
    p_stats = pd.merge(p_stats, bot_stats, on="nickname", how="left")
    p_stats["player_vs_bot_count"] = p_stats["player_vs_bot_count"].fillna(0)
    p_stats["player_vs_bot_mean_score"] = p_stats["player_vs_bot_mean_score"].fillna(p_stats["player_mean_score"])
    p_stats["player_vs_bot_mean_diff"] = p_stats["player_vs_bot_mean_diff"].fillna(p_stats["player_mean_diff"])
    p_stats["player_vs_bot_win_rate"] = p_stats["player_vs_bot_win_rate"].fillna(p_stats["player_win_rate"])

    # Granular per-bot benchmark anchors and human-vs-human profiles
    benchmarks = ["BetterBot", "STEEBot", "HastyBot", "BB-8", "human"]
    for bench in benchmarks:
        if bench == "human":
            sub_games = all_df[all_df["is_opp_bot"] == 0]
        else:
            sub_games = all_df[all_df["nickname_opp"] == bench]

        bench_grp = sub_games.groupby("nickname")
        bench_stats = bench_grp.agg(
            **{
                f"player_vs_{bench}_count": ("game_id", "count"),
                f"player_vs_{bench}_mean_diff": ("score_diff", "mean"),
                f"player_vs_{bench}_mean_score": ("score", "mean"),
                f"player_vs_{bench}_win_rate": ("is_win", "mean"),
            }
        ).reset_index()

        p_stats = pd.merge(p_stats, bench_stats, on="nickname", how="left")
        p_stats[f"player_vs_{bench}_count"] = p_stats[f"player_vs_{bench}_count"].fillna(0)
        p_stats[f"player_vs_{bench}_mean_diff"] = p_stats[f"player_vs_{bench}_mean_diff"].fillna(
            p_stats["player_mean_diff"]
        )
        p_stats[f"player_vs_{bench}_mean_score"] = p_stats[f"player_vs_{bench}_mean_score"].fillna(
            p_stats["player_mean_score"]
        )
        p_stats[f"player_vs_{bench}_win_rate"] = p_stats[f"player_vs_{bench}_win_rate"].fillna(
            p_stats["player_win_rate"]
        )

    p_stats["player_bot_vs_human_score_gap"] = (
        p_stats["player_vs_bot_mean_score"] - p_stats["player_vs_human_mean_score"]
    )
    p_stats["player_bot_vs_human_diff_gap"] = (
        p_stats["player_vs_bot_mean_diff"] - p_stats["player_vs_human_mean_diff"]
    )
    p_stats["player_bot_vs_human_win_gap"] = (
        p_stats["player_vs_bot_win_rate"] - p_stats["player_vs_human_win_rate"]
    )

    opp_stats = p_stats.copy()
    opp_cols = {col: f"opp_{col}" for col in opp_stats.columns if col != "nickname"}
    opp_cols["nickname"] = "nickname_opp"
    opp_stats = opp_stats.rename(columns=opp_cols)

    return p_stats, opp_stats


def apply_player_aggregations(
    df: pd.DataFrame,
    p_stats: pd.DataFrame,
    opp_stats: pd.DataFrame,
    graph_skills: pd.DataFrame,
) -> pd.DataFrame:
    """Join player aggregations, graph skills, and create relative cross features."""
    df_out = pd.merge(df, p_stats, on="nickname", how="left")
    df_out = pd.merge(df_out, opp_stats, on="nickname_opp", how="left")

    opp_graph_skills = graph_skills.copy().rename(
        columns={
            "nickname": "nickname_opp",
            "graph_skill_score_diff": "opp_graph_skill_score_diff",
            "graph_skill_win_margin": "opp_graph_skill_win_margin",
        }
    )
    df_out = pd.merge(df_out, graph_skills, on="nickname", how="left")
    df_out = pd.merge(df_out, opp_graph_skills, on="nickname_opp", how="left")

    df_out["graph_skill_diff"] = (
        df_out["graph_skill_score_diff"] - df_out["opp_graph_skill_score_diff"]
    )
    df_out["graph_win_skill_diff"] = (
        df_out["graph_skill_win_margin"] - df_out["opp_graph_skill_win_margin"]
    )

    df_out["score_vs_player_mean"] = df_out["score"] - df_out["player_mean_score"]
    df_out["diff_vs_player_mean_diff"] = (
        df_out["score_diff"] - df_out["player_mean_diff"]
    )
    df_out["score_vs_opp_mean"] = df_out["score"] - df_out["opp_player_mean_score"]
    df_out["expected_margin_gap"] = df_out["score_diff"] - (
        df_out["player_mean_score"] - df_out["opp_player_mean_score"]
    )
    df_out["experience_diff"] = np.log1p(df_out["player_games_count"]) - np.log1p(
        df_out["opp_player_games_count"]
    )

    # Standardized match z-scores and percentile boundaries
    df_out["score_zscore_player"] = (
        df_out["score"] - df_out["player_mean_score"]
    ) / (df_out["player_std_score"] + 1e-5)
    df_out["diff_zscore_player"] = (
        df_out["score_diff"] - df_out["player_mean_diff"]
    ) / (df_out["player_std_diff"] + 1e-5)
    df_out["score_percentile_p10_gap"] = df_out["score"] - df_out["player_p10_score"]
    df_out["score_percentile_p90_gap"] = df_out["score"] - df_out["player_p90_score"]
    df_out["score_percentile_p50_gap"] = df_out["score"] - df_out["player_median_score"]

    # Defensive concession and game tempo dynamics
    df_out["score_vs_opp_conceded_mean"] = (
        df_out["score"] - df_out["opp_player_mean_conceded"]
    )
    df_out["score_opp_vs_player_conceded_mean"] = (
        df_out["score_opp"] - df_out["player_mean_conceded"]
    )
    df_out["defensive_suppression_gap"] = (
        df_out["opp_player_mean_score"] - df_out["score_opp"]
    )
    df_out["tempo_vs_expected_tempo"] = df_out["total_score"] - 0.5 * (
        df_out["player_mean_total_score"] + df_out["opp_player_mean_total_score"]
    )

    return df_out


def compute_bayesian_target_encodings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Compute strictly leak-free empirical Bayes target encodings for player ratings."""
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(train_df, groups=train_df["game_id"]))

    global_mean = float(train_df["rating"].mean())
    global_median = float(train_df["rating"].median())
    global_std = float(train_df["rating"].std())
    global_q10 = float(np.percentile(train_df["rating"], 10))
    global_q90 = float(np.percentile(train_df["rating"], 90))
    smoothing = 10.0

    pairs_df = extract_unique_match_pairs(train_df, test_df)
    all_nicks = np.unique(
        np.concatenate([train_df["nickname"].dropna().values, test_df["nickname"].dropna().values])
    )
    nick2idx = {nick: i for i, nick in enumerate(all_nicks)}

    target_cols = [
        "oof_player_rating_mean",
        "oof_player_rating_median",
        "oof_player_rating_std",
        "oof_player_rating_q10",
        "oof_player_rating_q90",
        "oof_player_rating_min",
        "oof_player_rating_max",
        "oof_player_rating_count",
        "oof_opp_rating_mean",
        "oof_opp_rating_median",
        "oof_graph_rating",
        "oof_opp_graph_rating",
    ]
    for col in target_cols:
        train_df[col] = np.nan
        test_df[col] = np.nan

    for fold, (trn_idx, val_idx) in enumerate(folds):
        trn_data = train_df.iloc[trn_idx]
        val_data = train_df.iloc[val_idx]

        p_grp = trn_data.groupby("nickname")["rating"]
        p_mean = p_grp.mean()
        p_median = p_grp.median()
        p_std = p_grp.std().fillna(0.0)
        p_q10 = p_grp.quantile(0.10)
        p_q90 = p_grp.quantile(0.90)
        p_count = p_grp.count()
        p_min = p_grp.min()
        p_max = p_grp.max()

        anchor_dict = {
            nick: (float(m), int(c)) for nick, (m, c) in zip(p_mean.index, zip(p_mean.values, p_count.values))
        }
        graph_ratings = solve_graph_rating_propagation(
            pairs_df, nick2idx, anchor_dict, global_mean, beta=2.85
        )

        smoothed_p_mean = (p_count * p_mean + smoothing * global_mean) / (
            p_count + smoothing
        )

        val_nick = val_data["nickname"]
        train_df.loc[train_df.index[val_idx], "oof_player_rating_mean"] = (
            val_nick.map(smoothed_p_mean).fillna(global_mean).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_median"] = (
            val_nick.map(p_median).fillna(global_median).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_std"] = (
            val_nick.map(p_std).fillna(global_std).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_q10"] = (
            val_nick.map(p_q10).fillna(global_q10).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_q90"] = (
            val_nick.map(p_q90).fillna(global_q90).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_min"] = (
            val_nick.map(p_min).fillna(global_mean).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_max"] = (
            val_nick.map(p_max).fillna(global_mean).values
        )
        train_df.loc[train_df.index[val_idx], "oof_player_rating_count"] = (
            val_nick.map(p_count).fillna(0).values
        )

        val_opp = val_data["nickname_opp"]
        train_df.loc[train_df.index[val_idx], "oof_opp_rating_mean"] = (
            val_opp.map(smoothed_p_mean).fillna(global_mean).values
        )
        train_df.loc[train_df.index[val_idx], "oof_opp_rating_median"] = (
            val_opp.map(p_median).fillna(global_median).values
        )

        train_df.loc[train_df.index[val_idx], "oof_graph_rating"] = (
            val_nick.map(graph_ratings).fillna(global_mean).values
        )
        train_df.loc[train_df.index[val_idx], "oof_opp_graph_rating"] = (
            val_opp.map(graph_ratings).fillna(global_mean).values
        )

    full_grp = train_df.groupby("nickname")["rating"]
    full_mean = full_grp.mean()
    full_median = full_grp.median()
    full_std = full_grp.std().fillna(0.0)
    full_q10 = full_grp.quantile(0.10)
    full_q90 = full_grp.quantile(0.90)
    full_count = full_grp.count()
    full_min = full_grp.min()
    full_max = full_grp.max()
    smoothed_full = (full_count * full_mean + smoothing * global_mean) / (
        full_count + smoothing
    )

    full_anchor_dict = {
        nick: (float(m), int(c)) for nick, (m, c) in zip(full_mean.index, zip(full_mean.values, full_count.values))
    }
    full_graph_ratings = solve_graph_rating_propagation(
        pairs_df, nick2idx, full_anchor_dict, global_mean, beta=2.85
    )

    test_nick = test_df["nickname"]
    test_opp = test_df["nickname_opp"]
    test_df["oof_player_rating_mean"] = (
        test_nick.map(smoothed_full).fillna(global_mean).values
    )
    test_df["oof_player_rating_median"] = (
        test_nick.map(full_median).fillna(global_median).values
    )
    test_df["oof_player_rating_std"] = (
        test_nick.map(full_std).fillna(global_std).values
    )
    test_df["oof_player_rating_q10"] = (
        test_nick.map(full_q10).fillna(global_q10).values
    )
    test_df["oof_player_rating_q90"] = (
        test_nick.map(full_q90).fillna(global_q90).values
    )
    test_df["oof_player_rating_min"] = (
        test_nick.map(full_min).fillna(global_mean).values
    )
    test_df["oof_player_rating_max"] = (
        test_nick.map(full_max).fillna(global_mean).values
    )
    test_df["oof_player_rating_count"] = test_nick.map(full_count).fillna(0).values
    test_df["oof_opp_rating_mean"] = (
        test_opp.map(smoothed_full).fillna(global_mean).values
    )
    test_df["oof_opp_rating_median"] = (
        test_opp.map(full_median).fillna(global_median).values
    )
    test_df["oof_graph_rating"] = (
        test_nick.map(full_graph_ratings).fillna(global_mean).values
    )
    test_df["oof_opp_graph_rating"] = (
        test_opp.map(full_graph_ratings).fillna(global_mean).values
    )

    for df in [train_df, test_df]:
        df["oof_rating_diff"] = df["oof_player_rating_mean"] - df["oof_opp_rating_mean"]
        df["oof_rating_range"] = (
            df["oof_player_rating_max"] - df["oof_player_rating_min"]
        )
        df["oof_graph_rating_diff"] = df["oof_graph_rating"] - df["oof_opp_graph_rating"]
        # Theoretical Elo Win Expectation E = 1 / (1 + 10^((opp_rating - player_rating) / 400))
        exponent = np.clip((df["oof_opp_rating_mean"] - df["oof_player_rating_mean"]) / 400.0, -10.0, 10.0)
        df["elo_expected_win_prob"] = 1.0 / (1.0 + np.power(10.0, exponent))
        graph_exponent = np.clip((df["oof_opp_graph_rating"] - df["oof_graph_rating"]) / 400.0, -10.0, 10.0)
        df["elo_graph_expected_win_prob"] = 1.0 / (1.0 + np.power(10.0, graph_exponent))

    return train_df, test_df, folds


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    # 2. Reconstruct Game Pairs
    game_pairs = reconstruct_game_pairs(train_df, test_df)

    # 3. Match-Level Features
    train_feat = compute_match_features(train_df, game_pairs)
    test_feat = compute_match_features(test_df, game_pairs)

    # 4. Player & Opponent Historical Aggregations & Graph Skill Estimation
    combined_matches = pd.concat([train_feat, test_feat], axis=0).reset_index(drop=True)
    p_stats, opp_stats = compute_player_aggregations(combined_matches)
    graph_skills = compute_graph_skills(train_df, test_df, alpha=5.0)

    train_feat = apply_player_aggregations(train_feat, p_stats, opp_stats, graph_skills)
    test_feat = apply_player_aggregations(test_feat, p_stats, opp_stats, graph_skills)

    # 5. Leak-Free Bayesian Target Encoding & Group Cross-Validation
    train_feat, test_feat, folds = compute_bayesian_target_encodings(
        train_feat, test_feat, n_splits=5, random_state=42
    )

    excluded_cols = ["game_id", "nickname", "nickname_opp", "rating"]
    feature_cols = [c for c in train_feat.columns if c not in excluded_cols]

    for col in feature_cols:
        train_feat[col] = train_feat[col].fillna(0.0)
        test_feat[col] = test_feat[col].fillna(0.0)

    X = train_feat[feature_cols].copy()
    y = train_feat["rating"].values
    X_test = test_feat[feature_cols].copy()

    oof_lgb = np.zeros(len(train_feat), dtype=np.float64)
    oof_cb = np.zeros(len(train_feat), dtype=np.float64)
    oof_xgb = np.zeros(len(train_feat), dtype=np.float64)
    test_lgb = np.zeros(len(test_feat), dtype=np.float64)
    test_cb = np.zeros(len(test_feat), dtype=np.float64)
    test_xgb = np.zeros(len(test_feat), dtype=np.float64)

    # 6. Train Diverse Multi-GBDT Ensemble (LightGBM, CatBoost, XGBoost) across folds
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
        X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]

        # 6a. LightGBM (leaf-wise tree growth with L1 loss)
        model_lgb = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            boosting_type="gbdt",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(
            X_train_f,
            y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            eval_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_val = model_lgb.predict(X_val_f)
        lgb_test = model_lgb.predict(X_test)

        # 6b. CatBoost (symmetric oblivious trees with MAE loss)
        model_cb = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=2500,
            learning_rate=0.04,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
            thread_count=-1,
        )
        model_cb.fit(
            X_train_f,
            y_train_f,
            eval_set=(X_val_f, y_val_f),
            verbose=False,
        )
        cb_val = model_cb.predict(X_val_f)
        cb_test = model_cb.predict(X_test)

        # 6c. XGBoost (depth-wise tree growth with absolute error)
        model_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
            tree_method="hist",
        )
        model_xgb.fit(
            X_train_f,
            y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            verbose=False,
        )
        xgb_val = model_xgb.predict(X_val_f)
        xgb_test = model_xgb.predict(X_test)

        oof_lgb[val_idx] = lgb_val
        oof_cb[val_idx] = cb_val
        oof_xgb[val_idx] = xgb_val

        test_lgb += lgb_test / len(folds)
        test_cb += cb_test / len(folds)
        test_xgb += xgb_test / len(folds)

        print(
            f"Fold {fold + 1}/{len(folds)} (LGB: {mean_absolute_error(y_val_f, lgb_val):.4f}, "
            f"CB: {mean_absolute_error(y_val_f, cb_val):.4f}, XGB: {mean_absolute_error(y_val_f, xgb_val):.4f})"
        )

    # 7. Metric-Aligned MAE Weight Optimization via SLSQP
    def mae_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = w[0] * oof_lgb + w[1] * oof_cb + w[2] * oof_xgb
        return mean_absolute_error(y, pred)

    opt_res = minimize(
        mae_objective,
        x0=[0.45, 0.40, 0.15],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    opt_weights = opt_res.x / np.sum(opt_res.x)
    print(f"Optimized Ensemble Weights: LGB={opt_weights[0]:.4f}, CB={opt_weights[1]:.4f}, XGB={opt_weights[2]:.4f}")

    oof_predictions = (
        opt_weights[0] * oof_lgb + opt_weights[1] * oof_cb + opt_weights[2] * oof_xgb
    )
    test_predictions = (
        opt_weights[0] * test_lgb + opt_weights[1] * test_cb + opt_weights[2] * test_xgb
    )

    # Bounded post-processing
    min_rating = float(np.min(y))
    max_rating = float(np.max(y))
    oof_predictions = np.clip(oof_predictions, min_rating, max_rating)
    test_predictions = np.clip(test_predictions, min_rating, max_rating)

    # 8. Final Validation Score (MAE)
    final_score = float(mean_absolute_error(y, oof_predictions))

    # 9. Save Submissions
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_predictions), dtype=int),
            "rating": test_predictions,
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
