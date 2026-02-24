import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# For reproducibility
np.random.seed(42)

# --------------------
# CONFIG
# --------------------
N_ACCOUNTS = 2000          # number of accounts
MIN_TX_PER_ACC = 50
MAX_TX_PER_ACC = 150
FRAUD_RATE = 0.003         # 0.3% transactions are fraud

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 7, 1)
TRANSACTION_END_DATE = datetime(2024, 12, 31)  # Transactions can extend beyond account creation period

HOME_COUNTRIES = ['US', 'FR', 'DE', 'UK', 'TN', 'IT', 'ES']

# --------------------
# 1. Generate accounts
# --------------------
account_ids = np.arange(1, N_ACCOUNTS + 1)

accounts = []
for acc_id in account_ids:
    # Account open date somewhere between START_DATE and END_DATE
    total_days = (END_DATE - START_DATE).days
    open_offset = np.random.randint(0, total_days)
    open_date = START_DATE + timedelta(days=open_offset)

    home_country = np.random.choice(
        HOME_COUNTRIES, 
        p=[0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1]
    )

    base_balance = np.random.lognormal(mean=9, sigma=0.7)  # around e^9 ~ 8k

    accounts.append((acc_id, open_date, home_country, base_balance))

accounts_df = pd.DataFrame(
    accounts,
    columns=['account_id', 'account_open_date', 'home_country', 'base_balance']
)

# --------------------
# 2. Merchants & devices
# --------------------
N_MERCHANTS = 500
merchant_ids = np.arange(1, N_MERCHANTS + 1)
merchant_countries = np.random.choice(HOME_COUNTRIES, size=N_MERCHANTS)
merchant_df = pd.DataFrame({
    'merchant_id': merchant_ids,
    'merchant_country': merchant_countries
})

# Create dictionary for O(1) lookups
merchant_country_map = dict(zip(merchant_df.merchant_id, merchant_df.merchant_country))

N_DEVICES = 2000
device_ids = np.arange(1, N_DEVICES + 1)

# --------------------
# Helper: sample timestamp AFTER a given start date
# --------------------
def sample_timestamp_after(start_date):
    """Generate timestamp between start_date and TRANSACTION_END_DATE"""
    days_range = (TRANSACTION_END_DATE - start_date).days
    if days_range <= 0:
        days_range = 1  # fallback
    day_offset = np.random.randint(0, max(1, days_range))
    base_day = start_date + timedelta(days=day_offset)
    # active hours ~ normal around 14:00, clipped to [0, 23]
    hour = int(np.clip(np.random.normal(14, 4), 0, 23))
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return base_day.replace(hour=hour, minute=minute, second=second)

# --------------------
# 3. Generate normal transactions
# --------------------
transactions = []
trx_id = 1

# Create dictionary for efficient account lookups
account_home_country_map = dict(zip(accounts_df.account_id, accounts_df.home_country))

print("Generating transactions...")
for idx, acc in accounts_df.iterrows():
    if idx % 200 == 0:
        print(f"Processing account {idx}/{N_ACCOUNTS}")
    
    acc_id = acc['account_id']
    open_date = acc['account_open_date']
    home_country = acc['home_country']
    balance = acc['base_balance']

    # number of transactions for this account
    n_tx = np.random.randint(MIN_TX_PER_ACC, MAX_TX_PER_ACC + 1)

    # typical amount for this account
    mean_amount = np.random.lognormal(mean=3.5, sigma=0.6)  # ~ e^3.5 ~ 33

    # choose 1-3 usual devices
    acc_devices = np.random.choice(device_ids, size=np.random.randint(1, 4), replace=False)
    # choose 5-15 usual merchants
    acc_merchants = np.random.choice(merchant_ids, size=np.random.randint(5, 16), replace=False)

    for _ in range(n_tx):
        # Generate timestamp AFTER account opening
        ts = sample_timestamp_after(open_date)

        # amount around account's mean_amount
        amount = np.random.lognormal(mean=np.log(mean_amount), sigma=0.5)

        merchant_id = int(np.random.choice(acc_merchants))
        m_country = merchant_country_map[merchant_id]

        # Mostly home country, sometimes merchant country
        if np.random.rand() < 0.9:
            country = home_country
        else:
            country = m_country

        device_id = int(np.random.choice(acc_devices))

        # balance evolves a bit
        balance = max(10.0, balance + np.random.normal(0, 50) - amount * 0.1)

        transactions.append([
            trx_id,
            acc_id,
            ts,
            amount,
            merchant_id,
            country,
            device_id,
            acc['account_open_date'],
            balance,
            home_country,  # Add home_country for is_international feature
            0  # is_fraud (will set later)
        ])
        trx_id += 1

# Build DataFrame
print("Building DataFrame...")
columns = [
    'transaction_id', 'account_id', 'timestamp', 'amount',
    'merchant_id', 'country', 'device_id',
    'account_open_date', 'account_balance', 'home_country', 'is_fraud'
]
trx_df = pd.DataFrame(transactions, columns=columns)

# --------------------
# 4. Inject fraud
# --------------------
print("Injecting fraud...")
n_total = len(trx_df)
n_fraud = max(1, int(FRAUD_RATE * n_total))  # at least 1

fraud_indices = np.random.choice(trx_df.index, size=n_fraud, replace=False)

for idx in fraud_indices:
    row = trx_df.loc[idx]

    # Get account home_country using O(1) lookup
    home_country = account_home_country_map[row['account_id']]

    # Pattern 1: much larger amount
    trx_df.at[idx, 'amount'] = row['amount'] * np.random.uniform(3, 10)

    # Pattern 2: odd hour (night)
    ts = row['timestamp']
    ts = ts.replace(
        hour=int(np.random.choice([0, 1, 2, 3, 4, 5])),
        minute=np.random.randint(0, 60)
    )
    trx_df.at[idx, 'timestamp'] = ts

    # Pattern 3: foreign country (not home_country)
    possible_countries = [c for c in HOME_COUNTRIES if c != home_country]
    trx_df.at[idx, 'country'] = np.random.choice(possible_countries)

    # Pattern 4: new device (outside normal ID range)
    trx_df.at[idx, 'device_id'] = np.random.randint(N_DEVICES + 1, N_DEVICES + 5000)

    # Mark as fraud
    trx_df.at[idx, 'is_fraud'] = 1

# --------------------
# 5. Inspect and save
# --------------------
print("\n" + "="*50)
print("Total transactions:", len(trx_df))
print("Fraud transactions:", trx_df['is_fraud'].sum())
print("Fraud rate:", f"{trx_df['is_fraud'].mean():.4f}")
print("="*50)

print("\nFirst few rows:")
print(trx_df.head())

# Save to CSV
print("\nSaving to CSV...")
trx_df.to_csv("transactions.csv", index=False)
print("✓ Saved to transactions.csv")

# Save accounts data separately
accounts_df.to_csv("accounts.csv", index=False)
print("✓ Saved to accounts.csv")