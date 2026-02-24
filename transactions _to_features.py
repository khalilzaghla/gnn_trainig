import pandas as pd
import numpy as np

# Load transactions
df = pd.read_csv("transactions.csv", parse_dates=["timestamp", "account_open_date"])

# home_country is now included in transactions.csv, no need to merge

# 1. Amount-related features
df["amount_normalized"] = np.log1p(df["amount"])  # log(1+amount)

# Mean and std per account
agg = df.groupby("account_id")["amount"].agg(["mean", "std"]).reset_index()
agg = agg.rename(columns={"mean": "amount_mean_account", "std": "amount_std_account"})
df = df.merge(agg, on="account_id", how="left")

# Replace NaN std with small value
df["amount_std_account"] = df["amount_std_account"].fillna(1.0)

# Z-score deviation
df["amount_deviation"] = (df["amount"] - df["amount_mean_account"]) / (df["amount_std_account"] + 1e-6)

# 2. Time features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.weekday  # 0=Mon, 6=Sun
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding for hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 3. Time since previous transaction (per account)
df = df.sort_values(["account_id", "timestamp"])
df["prev_timestamp"] = df.groupby("account_id")["timestamp"].shift(1)
df["time_since_prev_txn"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
df["time_since_prev_txn"] = df["time_since_prev_txn"].fillna(0)
# Optionally log-scale
df["time_since_prev_txn_log"] = np.log1p(df["time_since_prev_txn"])

# 4. Velocity features (transactions per account in last hour/day)

# For efficiency, you can approximate with rolling counts per account
# Here is a simple (not super-optimized) version:

def compute_velocity(group, window_seconds):
    # group is df for one account, sorted by timestamp
    times = group["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    counts = np.zeros(len(group), dtype=np.int32)
    j = 0
    for i in range(len(group)):
        # move j to keep only events within [t_i - window, t_i)
        while times[i] - times[j] > window_seconds:
            j += 1
        counts[i] = i - j  # number of previous txns within window
    return pd.Series(counts, index=group.index)

# 1 hour = 3600s, 1 day = 86400s
df = df.sort_values(["account_id", "timestamp"])
df["velocity_last_hour"] = df.groupby("account_id", group_keys=False).apply(
    lambda g: compute_velocity(g, window_seconds=3600), include_groups=False
)
df["velocity_last_day"] = df.groupby("account_id", group_keys=False).apply(
    lambda g: compute_velocity(g, window_seconds=86400), include_groups=False
)

# 5. Behavioral / profile features

# card_age_days
df["card_age_days"] = (df["timestamp"] - df["account_open_date"]).dt.days.clip(lower=0)

# account_balance_ratio
df["account_balance_ratio"] = df["amount"] / (df["account_balance"] + 1e-6)

# is_international: need home_country
# If df has a 'home_country' column:
if "home_country" in df.columns:
    df["is_international"] = (df["country"] != df["home_country"]).astype(int)
else:
    # if you don't have home_country, you can mark all as domestic for now
    df["is_international"] = 0

# is_recurring_merchant: account has seen this merchant before
df = df.sort_values(["account_id", "timestamp"])
# Mark first occurrence per (account, merchant) as 0, others as 1
first_seen = df.groupby(["account_id", "merchant_id"])["timestamp"].transform("min")
df["is_recurring_merchant"] = (df["timestamp"] > first_seen).astype(int)

# Clean up helper columns if you want
df = df.drop(columns=["prev_timestamp", "amount_mean_account", "amount_std_account"], errors="ignore")

print(df.head())
df.to_csv("transactions_with_features.csv", index=False)
print("Saved to transactions_with_features.csv")