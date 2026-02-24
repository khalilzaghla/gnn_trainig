# GNN Fraud Detection System

A Graph Neural Network (GNN) based fraud detection system for financial transactions using PyTorch Geometric. The system leverages graph structures to detect fraudulent patterns in transaction networks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## 🎯 Overview

This project implements a Graph Neural Network for detecting fraudulent transactions by modeling transactions as nodes and their relationships (account, merchant, device) as edges. The system uses two GNN architectures:

- **GraphSAGE**: Inductive graph learning with neighborhood sampling
- **GAT (Graph Attention Networks)**: Attention-based graph learning

## ✨ Features

- **Automatic Feature Engineering**: Generates 16+ features from raw transaction data
  - Time-based features (hour, day of week, cyclic encoding)
  - Velocity features (transaction frequency)
  - Amount normalization and deviation
  - Account age and balance ratios
  - International transaction flags
  
- **Graph Construction**: Multiple edge types
  - Sequential edges (same account, chronological)
  - Device-based edges (same device)
  - Merchant-based edges (same merchant)

- **Model Training**: Grid search over architectures
  - SAGE and GAT models
  - Hidden dimensions: 32, 64, 128
  - Automatic best model selection based on F1-score

- **Inference Pipeline**: Test new transactions
  - Automatic model architecture detection
  - Feature generation for new data
  - Fraud probability scoring
  - Configurable thresholds

## 📁 Project Structure

```
gnn_taining/
├── config.yaml                          # Configuration file
├── README.md                            # This file
│
├── Data Processing:
│   ├── transactions.csv                 # Raw transaction data
│   ├── transactions_with_features.csv   # Processed features
│   ├── data_generator.py                # Generate synthetic data
│   └── transactions_to_features.py      # Feature engineering
│
├── Graph & Model:
│   ├── graph_generator.py               # Build graph from transactions
│   ├── transaction_graph_data.pt        # Saved graph structure
│   ├── best_model.pt                    # Best trained model weights
│   └── notebook_source.py               # Training notebook (Kaggle)
│
├── Testing & Inference:
│   ├── test.py                          # Inference on new transactions
│   ├── test_transactions_with_features.csv
│   └── fraud_predictions.csv            # Prediction results
│
└── Results:
    ├── experiments_results.csv          # Training experiment results
    ├── pr_best.png                      # Precision-Recall curve
    └── roc_curve.png                    # ROC curve
```

## 🔧 Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+
- PyTorch Geometric
- pandas
- numpy
- scikit-learn
- PyYAML
- matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd gnn_taining

# Install dependencies
pip install torch torch-geometric torch-scatter torch-sparse torch-cluster
pip install pandas numpy scikit-learn matplotlib pyyaml

# Verify installation
python -c "import torch; import torch_geometric; print('Setup complete!')"
```

## 🚀 Usage

### 1. Generate Graph from Transactions

```bash
# Create graph structure from transaction data with features
python graph_generator.py
```

**Output**: `transaction_graph_data.pt` (PyG Data object with 16 features)

### 2. Train Models (Kaggle/Notebook)

Use `notebook_source.py` to train models:

```python
# The notebook will:
# - Load transaction graph data
# - Train SAGE and GAT models with different hidden dimensions
# - Perform grid search (6 experiments total)
# - Save best model based on F1-score
# - Generate evaluation plots
```

**Outputs**:
- `best_model.pt` - Best model weights
- `experiments_results.csv` - All experiment metrics
- `pr_best.png` - Precision-Recall curve
- `roc_curve.png` - ROC curve

### 3. Test on New Transactions

```bash
# Run inference on new transaction events
python test.py
```

**What it does**:
1. Generates 200 synthetic test transactions
2. Engineers all 16 features automatically
3. Builds graph structure
4. Loads trained model (auto-detects architecture)
5. Runs fraud detection inference
6. Evaluates predictions and displays results

**Outputs**:
- `test_transactions_with_features.csv` - Test data with features
- `fraud_predictions.csv` - Predictions with probabilities

### 4. Generate Synthetic Data

```bash
# Create synthetic transaction dataset
python data_generator.py
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Data Configuration
```yaml
data:
  input_csv: "transactions_with_features.csv"
  graph_data: "transaction_graph_data.pt"
  feature_columns: [...]  # 16 features
  split:
    train: 0.7
    val: 0.1
    test: 0.2
```

### Model Parameters
```yaml
models:
  sage:
    hidden_dims: [32, 64, 128]
    num_layers: 2
    dropout: 0.2
  
  gat:
    hidden_dims: [32, 64, 128]
    heads: 4
    dropout: 0.2
```

### Training Settings
```yaml
training:
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: "Adam"
```

### Inference Settings
```yaml
inference:
  model_path: "best_model.pt"
  threshold_mode: "optimal"  # or "fixed"
  fixed_threshold: 0.99
```

## 🏗️ Model Architecture

### GraphSAGE Model
```python
SAGEModel(
  conv1: SAGEConv(16, hidden_dim)
  conv2: SAGEConv(hidden_dim, hidden_dim)
  lin1: Linear(hidden_dim, hidden_dim)
  lin2: Linear(hidden_dim, 1)
)
```

### GAT Model
```python
GATModel(
  conv1: GATConv(16, hidden_dim, heads=4)
  conv2: GATConv(hidden_dim*4, hidden_dim, heads=1)
  lin1: Linear(hidden_dim, hidden_dim)
  lin2: Linear(hidden_dim, 1)
)
```

### Input Features (16 total)

1. **Amount Features**:
   - `amount_normalized` - Z-score normalized amount
   - `amount_deviation` - Deviation from user average

2. **Time Features**:
   - `hour_sin`, `hour_cos` - Cyclic hour encoding
   - `day_of_week` - Day of week (0-6)
   - `is_weekend` - Weekend flag

3. **Velocity Features**:
   - `time_since_prev_txn_log` - Log time since last transaction
   - `velocity_last_hour` - Transactions in last hour
   - `velocity_last_day` - Transactions in last day

4. **Account Features**:
   - `card_age_days` - Days since card creation
   - `account_balance_ratio` - Amount/balance ratio

5. **Merchant Features**:
   - `is_recurring_merchant` - Recurring merchant flag
   - `merchant_id` - Merchant identifier (encoded)

6. **Location Features**:
   - `is_international` - International transaction flag
   - `country_id` - Country code (encoded)
   - `device_id` - Device identifier (encoded)

## 📊 Results

### Training Results (Example)

| Model | Hidden Dim | AUC-PR | AUC-ROC | Best F1 | Precision | Recall |
|-------|-----------|---------|---------|---------|-----------|--------|
| GAT   | 64        | 0.8542  | 0.9123  | 0.7856  | 0.8124    | 0.7612 |
| SAGE  | 64        | 0.8401  | 0.9056  | 0.7734  | 0.7989    | 0.7492 |
| GAT   | 128       | 0.8498  | 0.9089  | 0.7801  | 0.8056    | 0.7567 |

### Inference Results

With threshold = 0.99:
- **Precision**: High (fewer false positives)
- **Recall**: Lower (may miss some frauds)
- **Use case**: High-confidence fraud detection

With threshold = 0.5:
- **Precision**: Lower
- **Recall**: Higher (catches more frauds)
- **Use case**: Risk assessment, manual review

## 🔍 Key Insights

1. **Graph Structure Matters**: Sequential, device, and merchant edges capture different fraud patterns
2. **Feature Engineering**: Time-based and velocity features are crucial
3. **Attention Mechanism**: GAT often outperforms SAGE by learning edge importance
4. **Threshold Selection**: 0.99 for high precision, lower values for better recall

## 📝 Example Output

```
🔍 GNN FRAUD DETECTION - INFERENCE TEST
================================================================================

📝 Generating 200 test events...
✅ Generated 200 events (Fraud rate: 15.00%)

🔨 Engineering features...
✅ Features ready

🕸️  Building graph...
✅ Graph: 200 nodes, 398 edges, 16 features

📦 Inspecting saved model: best_model.pt
✅ Detected: GAT
   - Input features: 16
   - Hidden dim: 64
   - Heads: 4
✅ Model loaded on cpu

🔍 Running inference (threshold=0.99)...
✅ Predicted 12 frauds out of 200

📊 FRAUD DETECTION TEST RESULTS
   Accuracy:   0.9250
   Precision:  0.8333
   Recall:     0.3333
   F1-Score:   0.4762
   AUC-ROC:    0.8956
   AUC-PR:     0.7234
```

## 🛠️ Development

### Adding New Features

1. Edit `transactions_to_features.py` or feature engineering in `test.py`
2. Update `feature_columns` in `config.yaml`
3. Rebuild graph with `python graph_generator.py`
4. Retrain model

### Changing Model Architecture

1. Modify model classes in `notebook_source.py`
2. Update `models` section in `config.yaml`
3. Retrain and save new model

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 📜 License

This project is licensed under the MIT License.

---

**Note**: This is a research/educational project. For production use, additional validation, monitoring, and compliance measures are required.
