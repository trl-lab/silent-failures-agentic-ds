from datasets import load_dataset
import os
from huggingface_hub import login, hf_hub_download
import datasets
from datasets import load_from_disk
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

load_dotenv() # read .env
PROJECT_ROOT: str = os.environ['PROJECT_ROOT']
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.width', None)         # 自动适应终端宽度
pd.set_option('display.max_colwidth', None)  # 列内容不截断


# football data

football_DIR = os.path.join(PROJECT_ROOT, "data/raw/football")
players = pd.read_csv(os.path.join(football_DIR, "players.csv"))
valuation = pd.read_csv(os.path.join(football_DIR, "player_valuations.csv"))
appearances = pd.read_csv(os.path.join(football_DIR, "appearances.csv"))

performance = appearances.groupby("player_id").agg({
    "goals": "sum",
    "assists": "sum",
    "minutes_played": "sum",
    "game_id": "count"
}).reset_index()

performance = performance.rename(columns={
    "game_id": "matches_played"
})



player_perform_df = players.merge(performance, on="player_id", how="left")
player_perform_df.to_csv(os.path.join(PROJECT_ROOT, "data/processed/player_perform_df.csv"), index=False)


# fm01 goal imbalance
def fm01_football_imbalance_data(df, positive_ratio=0.01):
    df = df.drop(columns=[
        "image_url",
        "url",
        "player_code",
        "first_name",
        "last_name",
        "name",
        "agent_name",
        "player_code",
        "city_of_birth",
        "date_of_birth",
        "agent_name",
        "international_goals",
        "international_caps"
    ])
    high_pct = int(round(positive_ratio * 100))
    low_pct = 100 - high_pct
    tag = f"{high_pct}to{low_pct}"

    goals_threshold = df['goals'].quantile(1 - positive_ratio)
    player_goals_df = df.copy()

    # 众数填充（跳过 player_id 和 goals）
    skip_fill = {"player_id", "goals"}
    for col in player_goals_df.columns:
        if col not in skip_fill:
            player_goals_df[col] = player_goals_df[col].fillna(player_goals_df[col].mode()[0])

    # 删除 player_id 或 goals 为空的行
    player_goals_df = player_goals_df.dropna(subset=["player_id", "goals"])

    # For integer goals many players tie at the quantile boundary; >= includes them so
    # the actual "high" ratio stays close to positive_ratio instead of falling short.
    player_goals_df['goals_label'] = (player_goals_df['goals'] >= goals_threshold).map({True: 1, False: 0})

    print(f"[{tag}] Goals threshold ({100 - high_pct}th percentile): {goals_threshold}")
    print(f"[{tag}] High goals: {(player_goals_df['goals_label'] == 1).sum()} ({(player_goals_df['goals_label'] == 1).mean()*100:.2f}%)")
    print(f"[{tag}] Low goals:  {(player_goals_df['goals_label'] == 0).sum()} ({(player_goals_df['goals_label'] == 0).mean()*100:.2f}%)")

    player_goals_df.drop(columns=['goals'], inplace=True)
    player_goals_train, player_goals_test = train_test_split(
        player_goals_df, test_size=0.2, random_state=42, stratify=player_goals_df['goals_label']
    )

    processed_dir: str = os.path.join(PROJECT_ROOT, "data/FM01_goal_imbalance/input")
    player_goals_train.copy().to_csv(os.path.join(processed_dir, f"player_goals_{high_pct}high_train.csv"), index=False)
    player_goals_test.copy().to_csv(os.path.join(processed_dir, f"player_goals_{high_pct}high_test.csv"), index=False)

player_perform_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/player_perform_df.csv"))
for ratio in [0.05]:
    fm01_football_imbalance_data(player_perform_df, positive_ratio=ratio)


# fm02 football Different representations of certain values
def fm02_football_diff_rep_values(df):

    df = df.drop(columns=[
        "image_url", "url", "player_code", "first_name", "last_name", "name",
        "agent_name", "highest_market_value_in_eur",
        "country_of_birth", "city_of_birth", "country_of_citizenship",
        "current_club_domestic_competition_id", "current_club_name",
        "sub_position", "foot", "contract_expiration_date", "position",
        "date_of_birth"
    ])

    # # 在填充前记录 international_caps 的原始缺失位置（用于后续注入 'na' 字符串）
    # intl_caps_null_mask = df['international_caps'].isna()

    # 众数填充（跳过 player_id、market_value_in_eur 和 height_in_cm）
    skip_fill = {"player_id", "market_value_in_eur", "height_in_cm", "international_caps"}
    for col in df.columns:
        if col not in skip_fill:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 删除 player_id 或 market_value_in_eur 为空的行
    df = df.dropna(subset=["player_id", "market_value_in_eur"])

    # only select 20 percent data for training
    player_perform_train = df.sample(frac=0.2, random_state=42).copy()

    # 从剩余 player_id 中再抽取 10%（与 player_perform_train 无重合）
    used_player_ids = set(player_perform_train['player_id'])
    df_remaining = df[~df['player_id'].isin(used_player_ids)]
    n_4k = int(len(df) * 0.1)
    player_perform_test = df_remaining.sample(n=n_4k, random_state=43).copy()

    processed_dir = os.path.join(PROJECT_ROOT, "data/FM02_player_rep_norm/input")
    player_perform_train.to_csv(os.path.join(processed_dir, "player_value_train.csv"), index=False)
    player_perform_test.to_csv(os.path.join(processed_dir, "player_value_test.csv"), index=False)

    # 各自从 international_caps 的缺失值中随机选 30%，替换为字符串 'na'
    for label, target_df in [('train', player_perform_train), ('test', player_perform_test)]:
        null_idx = target_df.index[target_df['international_caps'].isna()]
        n_to_replace = int(len(null_idx) * 0.3)
        if n_to_replace > 0:
            chosen_idx = null_idx.to_series().sample(n=n_to_replace, random_state=42).index
            target_df['international_caps'] = target_df['international_caps'].astype(object)
            target_df.loc[chosen_idx, 'international_caps'] = 'na'
        print(f"player_value_{label}: replaced {n_to_replace} international_caps NaN → 'na' "
              f"(out of {len(null_idx)} NaN rows)")

    # 各自从 height_in_cm 的非空数值中随机选 200 个，替换为带单位字符串（如 185.0 → '185cm'）
    for label, target_df in [('train', player_perform_train), ('test', player_perform_test)]:
        numeric_idx = target_df.index[target_df['height_in_cm'].notna()]
        n = min(200, len(numeric_idx))
        chosen_idx = numeric_idx.to_series().sample(n=n, random_state=45).index
        target_df.loc[chosen_idx, 'height_in_cm'] = target_df.loc[chosen_idx, 'height_in_cm'].apply(
            lambda x: f"{int(float(x))}cm"
        )
        print(f"player_value_{label}: replaced {n} height_in_cm values with unit-suffixed strings")

    print(f"player_perform_train shape: {player_perform_train.shape}")
    print(f"player_perform_test shape: {player_perform_test.shape}")
    overlap = set(player_perform_train['player_id']) & set(player_perform_test['player_id'])
    print(f"player_id 重合数量: {len(overlap)}")

    player_perform_train.to_csv(os.path.join(processed_dir, "player_value_rep_norm_train.csv"), index=False)
    player_perform_test.to_csv(os.path.join(processed_dir, "player_value_rep_norm_test.csv"), index=False)

    return


player_perform_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/player_perform_df.csv"))
fm02_football_diff_rep_values(df=player_perform_df)

# fm03 correlation causation
def fm03_build_football_correlation_dataset(df):
    """
    Split player_perform_df into train/test and inject a synthetic column_x:
      - train: column_x = 3 * market_value_in_eur + small Gaussian noise
      - test:  column_x = random values drawn from the same value range
                          (zero causal relationship with the label)

    Saves train/test as CSV to the same directory as data_path.
    Returns (train_df, test_df) with column_x appended.
    """
    LABEL_COL = "market_value_in_eur"
    random_state = 42
    df = df.drop(columns=[
        "image_url", "url", "player_code", "first_name", "last_name", "name",
        "agent_name", "country_of_birth", "city_of_birth", "country_of_citizenship",
        "current_club_name", "highest_market_value_in_eur"
    ])
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    df = df.sample(frac=0.2, random_state=random_state).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    train_df = train_df.copy()
    test_df = test_df.copy()

    rng = np.random.RandomState(random_state)

    # noise std = 1% of the training label mean — large enough to be detectable,
    # small enough to preserve near-perfect correlation
    noise_std = train_df[LABEL_COL].mean() * 0.01
    noise = rng.normal(loc=0, scale=noise_std, size=len(train_df))
    train_df["column_x"] = 3 * train_df[LABEL_COL] + noise

    label_min = df[LABEL_COL].min()
    label_max = df[LABEL_COL].max()
    test_df["column_x"] = rng.uniform(label_min, label_max, size=len(test_df))

    out_dir = os.path.join(PROJECT_ROOT, "data/FM03_correlation_causation/input")
    train_df.to_csv(os.path.join(out_dir, "player_value_corr_causal_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "player_value_corr_causal_test.csv"), index=False)

    return

player_perform_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/player_perform_df.csv"))
fm03_build_football_correlation_dataset(player_perform_df)


# fm04 feature engineering: balance scale data
def fm04_balance_scale_feature_eng():
    raw_path = os.path.join(PROJECT_ROOT, "data/raw/balance+scale/balance-scale.data")
    out_dir1 = os.path.join(PROJECT_ROOT, "data/processed/balance_scale")
    out_dir2 = os.path.join(PROJECT_ROOT, "data/FM04_balance_torque/input")
    columns = ["Class Name", "Left Weight", "Left Distance", "Right Weight", "Right Distance"]
    df = pd.read_csv(raw_path, header=None, names=columns)
    df.to_csv(os.path.join(out_dir1, "balance_scale.csv"), index=False)

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class Name"])
    train.to_csv(os.path.join(out_dir2, "balance_torque_train.csv"), index=False)
    test.to_csv(os.path.join(out_dir2, "balance_torque_test.csv"), index=False)

    print(f"train shape: {train.shape}, test shape: {test.shape}")
    return

fm04_balance_scale_feature_eng()








# fm05 credit card leakage
def fm05_credit_card_data():
    raw_path = os.path.join(PROJECT_ROOT, "data/raw/creditcard/AER_credit_card_data.csv")
    out_dir = os.path.join(PROJECT_ROOT, "data/FM05_credit_card_leakage/input")

    df = pd.read_csv(raw_path)

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["card"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(os.path.join(out_dir, "credit_card_train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "credit_card_test.csv"), index=False)

    print(f"train shape: {train.shape}, test shape: {test.shape}")
    print(f"card=yes rate — train: {(train['card']=='yes').mean():.4f}, test: {(test['card']=='yes').mean():.4f}")

fm05_credit_card_data()



# fm06 metric selection
def fm06_player_goal_imbalance_data(df, positive_ratio=0.01):
    df = df.drop(columns=[
        "image_url",
        "url",
        "player_code",
        "first_name",
        "last_name",
        "name",
        "agent_name",
        "player_code",
        "city_of_birth",
        "date_of_birth",
        "agent_name"
    ])
    high_pct = int(round(positive_ratio * 100))
    low_pct = 100 - high_pct
    tag = f"{high_pct}to{low_pct}"

    goals_threshold = df['goals'].quantile(1 - positive_ratio)
    player_goals_df = df.copy()

    # 众数填充（跳过 player_id 和 goals）
    skip_fill = {"player_id", "goals"}
    for col in player_goals_df.columns:
        if col not in skip_fill:
            player_goals_df[col] = player_goals_df[col].fillna(player_goals_df[col].mode()[0])

    # 删除 player_id 或 goals 为空的行
    player_goals_df = player_goals_df.dropna(subset=["player_id", "goals"])

    # For integer goals many players tie at the quantile boundary; >= includes them so
    # the actual "high" ratio stays close to positive_ratio instead of falling short.
    player_goals_df['goals_label'] = (player_goals_df['goals'] >= goals_threshold).map({True: 1, False: 0})

    print(f"[{tag}] Goals threshold ({100 - high_pct}th percentile): {goals_threshold}")
    print(f"[{tag}] High goals: {(player_goals_df['goals_label'] == 1).sum()} ({(player_goals_df['goals_label'] == 1).mean()*100:.2f}%)")
    print(f"[{tag}] Low goals:  {(player_goals_df['goals_label'] == 0).sum()} ({(player_goals_df['goals_label'] == 0).mean()*100:.2f}%)")

    player_goals_df.drop(columns=['goals'], inplace=True)
    player_goals_train, player_goals_test = train_test_split(
        player_goals_df, test_size=0.2, random_state=42, stratify=player_goals_df['goals_label']
    )

    processed_dir: str = os.path.join(PROJECT_ROOT, "data/FM06_metric_select/input")
    player_goals_train.copy().to_csv(os.path.join(processed_dir, f"player_goals_imbalance_train.csv"), index=False)
    player_goals_test.copy().to_csv(os.path.join(processed_dir, f"player_goals_imbalance_test.csv"), index=False)

player_perform_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/player_perform_df.csv"))
for ratio in [0.05]:
    fm06_player_goal_imbalance_data(player_perform_df, positive_ratio=ratio)

#  feature engineering: football

def football_feature_engineer(player_perform_df):
    cols_to_drop = [
        "image_url", "url", "player_code", "first_name", "last_name", "name",
        "agent_name", "highest_market_value_in_eur",
        "country_of_birth", "city_of_birth", "country_of_citizenship",
        "current_club_domestic_competition_id", "current_club_name","contract_expiration_date",
        "sub_position", "position", "foot", "current_club_id","current_national_team_id"
    ]
    existing_cols = [c for c in cols_to_drop if c in player_perform_df.columns]
    df = player_perform_df.drop(columns=existing_cols)
    skip_fill = {"player_id", "market_value_in_eur"}
    for col in df.columns:
        if col not in skip_fill:
            df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=["player_id", "market_value_in_eur"])


    # 随机抽取 40% 作为训练集，再从剩余中抽取 20% 作为测试集（player_id 不重合）
    player_perf_train = df.sample(frac=0.4, random_state=42).copy()
    remaining = df[~df['player_id'].isin(player_perf_train['player_id'])]
    n_test = int(len(df) * 0.2)
    player_perf_test = remaining.sample(n=n_test, random_state=43).copy()

    out_dir = os.path.join(PROJECT_ROOT, "data/FM04_player_age/input")
    player_perf_train.to_csv(os.path.join(out_dir, "player_perf_train.csv"), index=False)
    player_perf_test.to_csv(os.path.join(out_dir, "player_perf_test.csv"), index=False)

    overlap = set(player_perf_train['player_id']) & set(player_perf_test['player_id'])
    print(f"player_id 重合数量: {len(overlap)}")
    print(f"player_perf_train shape: {player_perf_train.shape}")
    print(f"player_perf_test  shape: {player_perf_test.shape}")

    reference_year = 2025
    dob_train = pd.to_datetime(player_perf_train['date_of_birth'], errors='coerce', format='mixed')
    dob_test = pd.to_datetime(player_perf_test['date_of_birth'], errors='coerce', format='mixed')
    player_perf_train['age'] = (reference_year - dob_train.dt.year).astype('int64')
    player_perf_test['age'] = (reference_year - dob_test.dt.year).astype('int64')
    player_perf_train.drop(columns=['date_of_birth'], inplace=True)
    player_perf_test.drop(columns=['date_of_birth'], inplace=True)


    player_perf_train.to_csv(os.path.join(out_dir, "player_perf_agefeat_train.csv"), index=False)
    player_perf_test.to_csv(os.path.join(out_dir, "player_perf_agefeat_test.csv"), index=False)

    print(f"player_perf_agefeat_train shape: {player_perf_train.shape}")
    print(f"player_perf_agefeat_test  shape: {player_perf_test.shape}")

player_perform_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/player_perform_df.csv"))
football_feature_engineer(player_perform_df)


# yellow card leakage
def build_yellow_card_leakage_data(appearances_with_player):
    drop_cols = ["image_url", "url", "player_code", "first_name", "last_name", "name", "agent_name"]

    all_game_ids = appearances_with_player["game_id"].drop_duplicates()
    sampled_game_ids_test = all_game_ids.sample(n=150, random_state=42)
    remaining_game_ids = all_game_ids[~all_game_ids.isin(sampled_game_ids_test)]
    sampled_game_ids = remaining_game_ids.sample(n=600, random_state=42)

    train_df = appearances_with_player[
        appearances_with_player["game_id"].isin(sampled_game_ids)
    ].reset_index(drop=True).drop(columns=drop_cols)

    test_df = appearances_with_player[
        appearances_with_player["game_id"].isin(sampled_game_ids_test)
    ].reset_index(drop=True).drop(columns=drop_cols)

    n = len(train_df)
    yellow_rate = (train_df["yellow_cards"] == 1).sum() / n
    red_rate = (train_df["red_cards"] == 1).sum() / n
    print(f"yellow_cards=1 percentage: {yellow_rate:.4f} ({yellow_rate*100:.2f}%)")
    print(f"red_cards=1    percentage: {red_rate:.4f} ({red_rate*100:.2f}%)")
    print(f"train shape: {train_df.shape}, test shape: {test_df.shape}")

    train_game_ids = set(train_df["game_id"].unique())
    test_game_ids = set(test_df["game_id"].unique())
    overlap_game_ids = train_game_ids & test_game_ids
    if overlap_game_ids:
        print(f"警告：存在重叠 game_id，共 {len(overlap_game_ids)} 个")
    else:
        print(f"无重叠：train 含 {len(train_game_ids)} 个 game_id，test 含 {len(test_game_ids)} 个 game_id，无交集。")

    out_dir = os.path.join(PROJECT_ROOT, "data/FM05_yellow_card_leakage/input")
    train_df.to_csv(os.path.join(out_dir, "yellow_card_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "yellow_card_test.csv"), index=False)

appearances_with_player = appearances.merge(players, on="player_id", how="left")
build_yellow_card_leakage_data(appearances_with_player)





