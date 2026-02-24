import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# ------------------
# Load data
# ------------------
df = pd.read_csv("transactions_with_features.csv", parse_dates=["timestamp", "account_open_date"])

# Ensure consistent ordering (we'll use row index as node_id)
df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
num_nodes = len(df)
print("Num transactions (nodes):", num_nodes)

# ------------------
# Choose feature columns
# ------------------
feature_cols = [
    "amount_normalized",
    "amount_deviation",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "time_since_prev_txn_log",
    "velocity_last_hour",
    "velocity_last_day",
    "card_age_days",
    "account_balance_ratio",
    "is_international",
    "is_recurring_merchant",
    # you can add more numeric features here
]

# Simple encoding of categorical IDs as numeric (embeddings can be added in the model)
# For now we just include them as raw ints (or scaled)
# merchant_id, device_id, country (optional)

# Encode country as integer
df["country_id"] = df["country"].astype("category").cat.codes

feature_cols += ["merchant_id", "device_id", "country_id"]

# Build feature tensor X
X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
print("Feature shape:", X.shape)

# ------------------
# Labels
# ------------------
y = torch.tensor(df["is_fraud"].values, dtype=torch.long)
print("Fraud rate:", y.float().mean().item())

# ------------------
# Build edges
# ------------------

# Helper to add edges to lists
def add_edges(src_list, dst_list, src, dst):
    src_list.append(src)
    dst_list.append(dst)

src_edges = []
dst_edges = []

# 1) Sequential edges: same account, consecutive in time
grouped = df.groupby("account_id").indices  # dict {account_id: index_array}
for acc_id, idx_array in grouped.items():
    # idx_array is already in sorted order because we sorted df
    idx_array = np.sort(idx_array)
    for i in range(len(idx_array) - 1):
        u = idx_array[i]
        v = idx_array[i + 1]
        # bidirectional edges (u<->v)
        add_edges(src_edges, dst_edges, u, v)
        add_edges(src_edges, dst_edges, v, u)

# 2) Device edges (optional) - connect transactions with same device_id
# To avoid huge cliques, we connect each transaction to the previous one with same device
device_groups = df.groupby("device_id").indices
for dev_id, idx_array in device_groups.items():
    idx_array = np.sort(idx_array)
    for i in range(len(idx_array) - 1):
        u = idx_array[i]
        v = idx_array[i + 1]
        add_edges(src_edges, dst_edges, u, v)
        add_edges(src_edges, dst_edges, v, u)

# 3) Merchant edges (optional) - similar to device
merchant_groups = df.groupby("merchant_id").indices
for mid, idx_array in merchant_groups.items():
    idx_array = np.sort(idx_array)
    for i in range(len(idx_array) - 1):
        u = idx_array[i]
        v = idx_array[i + 1]
        add_edges(src_edges, dst_edges, u, v)
        add_edges(src_edges, dst_edges, v, u)

# Build edge_index tensor
edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
print("Num edges:", edge_index.shape[1])

# ------------------
# Train/val/test masks (chronological split)
# ------------------
# Use timestamp order for realistic split
df = df.sort_values("timestamp").reset_index(drop=True)

num_nodes = len(df)
train_end = int(0.7 * num_nodes)
val_end = int(0.8 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[:train_end] = True
val_mask[train_end:val_end] = True
test_mask[val_end:] = True

print("Train nodes:", train_mask.sum().item())
print("Val nodes:", val_mask.sum().item())
print("Test nodes:", test_mask.sum().item())

# IMPORTANT: we sorted df again, so we must reorder X, y accordingly
# Rebuild X, y with new ordering:
X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
y = torch.tensor(df["is_fraud"].values, dtype=torch.long)

# ------------------
# Build PyG Data object
# ------------------
data = Data(
    x=X,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)

print(data)
torch.save(data, "transaction_graph_data.pt")
print("Saved graph data to transaction_graph_data.pt")
