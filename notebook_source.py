


# kaggle !!!!!!!!!!!!!  use this  in kaggle  





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#/kaggle/input/datasets/khalilzaghla/transactions-with-features-csv


# 1) Install deps
"pip install torch-geometric torch-scatter torch-sparse torch-cluster -q"
"pip install pandas scikit-learn matplotlib -q"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin1(h)
        h = F.relu(h)
        logits = self.lin2(h).squeeze(-1)
        return logits




class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin1(h)
        h = F.relu(h)
        logits = self.lin2(h).squeeze(-1)
        return logits





def train_and_eval(model, data, config, device):
    model = model.to(device)
    data_device = data.to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    fraud_ratio = data_device.y.float().mean().item()
    pos_weight = (1 - fraud_ratio) / (fraud_ratio + 1e-8)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    epochs = config.get("epochs", 50)
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data_device.x, data_device.edge_index)
        loss = criterion(logits[data_device.train_mask], data_device.y[data_device.train_mask].float())
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        logits = model(data_device.x, data_device.edge_index)
        probs_all = torch.sigmoid(logits).cpu().numpy()
        y_all = data_device.y.cpu().numpy()
    model = model.cpu()
    if device.type == "cuda":
        del data_device, logits
        torch.cuda.empty_cache()
    return probs_all, y_all




def find_best_threshold(y_true, y_score):
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    best = {"thr": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best = {"thr": thr, "f1": f1, "precision": precision, "recall": recall}
    return best

print("="*60)
print("GNN Fraud Detection: Search on Kaggle")
print("="*60)





data_path = "/kaggle/input/models/khalilzaghla/transaction-graph-data/pytorch/default/1/transaction_graph_data.pt"
data = torch.load(data_path, weights_only=False)
print("Data:", data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
data = data.cpu()

config = {"epochs": 50, "lr": 1e-3, "weight_decay": 1e-5, "verbose": False}
model_types = ["sage", "gat"]
hidden_dims = [32, 64, 128]

results = []
best_f1 = 0.0
best_config = None
best_probs = None
best_y_true = None
best_state = None



print("\nStarting experiments...\n")
for mtype in model_types:
    for hd in hidden_dims:
        print(f"Training {mtype.upper()} hidden_dim={hd}")
        in_dim = data.x.size(1)
        if mtype == "sage":
            model = SAGEModel(in_dim, hd, out_dim=1)
        else:
            model = GATModel(in_dim, hd, out_dim=1, heads=4)
        probs_all, y_all = train_and_eval(model, data, config, device)
        test_mask = data.test_mask.cpu().numpy()
        y_test = y_all[test_mask]
        p_test = probs_all[test_mask]
        auc_pr = average_precision_score(y_test, p_test)
        auc_roc = roc_auc_score(y_test, p_test)
        best_thr = find_best_threshold(y_test, p_test)
        results.append({
            "model": mtype,
            "hidden_dim": hd,
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "best_thr": best_thr["thr"],
            "best_f1": best_thr["f1"],
            "best_precision": best_thr["precision"],
            "best_recall": best_thr["recall"],
        })
        print(f"  AUC-PR {auc_pr:.4f} | AUC-ROC {auc_roc:.4f} | "
              f"Best F1 {best_thr['f1']:.4f} @ thr={best_thr['thr']:.3f}\n")
        if best_thr["f1"] > best_f1:
            best_f1 = best_thr["f1"]
            best_config = {"model": mtype, "hidden_dim": hd, "threshold": best_thr["thr"]}
            best_probs = p_test
            best_y_true = y_test
            best_state = model.state_dict()
        del model, probs_all, y_all
        if device.type == "cuda":
            torch.cuda.empty_cache()



df = pd.DataFrame(results).sort_values("best_f1", ascending=False).reset_index(drop=True)
print("="*60)
print("RESULTS")
print("="*60)
print(df.to_string(index=False))
df.to_csv("experiments_results.csv", index=False)
print("\\nSaved experiments_results.csv")

print("\\nBest config:", best_config, "Best F1:", best_f1)






if best_probs is not None:
    prec, rec, thr = precision_recall_curve(best_y_true, best_probs)
    plt.figure(figsize=(7,5))
    plt.plot(rec, prec, label=f"{best_config['model'].upper()} (hidden={best_config['hidden_dim']})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Best Model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_best.png", dpi=150)
    plt.show()
    print("Saved pr_best.png")

if best_state is not None:
    torch.save(best_state, "best_model.pt")
    print("Saved best_model.pt")

print("\nDone.")



from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# y_test: true labels (0/1)
# probs_test: predicted probabilities

fpr, tpr, thresholds = roc_curve(best_y_true, best_probs)
auc_roc = roc_auc_score(best_y_true, best_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_roc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()






if best_probs is not None:
    prec, rec, thr = precision_recall_curve(best_y_true, best_probs)
    plt.figure(figsize=(7,5))
    plt.plot(rec, prec, label=f"{best_config['model'].upper()} (hidden={best_config['hidden_dim']})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision_Recall Curve (Best Model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_best.png", dpi=150)
    plt.show()
    print("Saved pr_best.png")












