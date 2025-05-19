# Fraud Detection Synthetic Data Generation, Training, and Model Saving

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# --- Set random seed for reproducibility ---
np.random.seed(42)

# --- Synthetic Data Generation ---
def random_date(start, end):
    """Generate random datetime between start and end."""
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

start_date, end_date = datetime(2023, 1, 1), datetime(2023, 12, 31)
num_samples = 10000
fraud_ratio = 0.05
num_fraud = int(num_samples * fraud_ratio)
num_legit = num_samples - num_fraud

def generate_users(n): return np.random.randint(1000, 1100, n)
def generate_merchants(n): return np.random.randint(5000, 5100, n)
def generate_amounts(n, high_fraud=False):
    return np.round(np.random.uniform(1000, 5000, n), 2) if high_fraud else np.round(np.random.exponential(100, n), 2)
def generate_devices(n, risky=False):
    return np.random.choice(['desktop', 'public_terminal'], n) if risky else np.random.choice(['mobile', 'tablet'], n)
def generate_locations(n, multi_city=False):
    return np.random.choice(['NY', 'LA', 'CHI', 'TX', 'SF'], n) if multi_city else np.random.choice(['user_city'], n)
def generate_category(n):
    return np.random.choice(['food', 'electronics', 'fashion', 'travel', 'groceries', 'gaming'], n)

# Legitimate transactions
legit = pd.DataFrame({
    'user_id': generate_users(num_legit),
    'merchant_id': generate_merchants(num_legit),
    'amount': generate_amounts(num_legit),
    'category': generate_category(num_legit),
    'timestamp': [random_date(start_date, end_date) for _ in range(num_legit)],
    'location': generate_locations(num_legit),
    'device': generate_devices(num_legit),
    'is_international': np.random.choice([0, 1], num_legit, p=[0.97, 0.03]),
    'is_weekend': np.random.choice([0, 1], num_legit, p=[0.7, 0.3]),
    'label': 0
})

# Fraudulent transactions
fraud = pd.DataFrame({
    'user_id': generate_users(num_fraud),
    'merchant_id': generate_merchants(num_fraud),
    'amount': generate_amounts(num_fraud, high_fraud=True),
    'category': generate_category(num_fraud),
    'timestamp': [random_date(start_date, end_date) for _ in range(num_fraud)],
    'location': generate_locations(num_fraud, multi_city=True),
    'device': generate_devices(num_fraud, risky=True),
    'is_international': np.random.choice([0, 1], num_fraud, p=[0.2, 0.8]),
    'is_weekend': np.random.choice([0, 1], num_fraud, p=[0.4, 0.6]),
    'label': 1
})

# Combine and shuffle
df = pd.concat([legit, fraud]).sample(frac=1).reset_index(drop=True)

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['high_amount_flag'] = (df['amount'] > 1000).astype(int)
df['device_risk_flag'] = df['device'].isin(['desktop', 'public_terminal']).astype(int)
df['location_risk_flag'] = df['location'].isin(['NY', 'LA', 'TX']).astype(int)
df['suspicious_time_flag'] = df['hour'].isin([1, 2, 3, 4]).astype(int)
df['user_transaction_freq'] = df.groupby('user_id')['user_id'].transform('size')
df['last_transaction_amount'] = df.groupby('user_id')['amount'].shift(1).fillna(0)
df['time_since_last'] = (df['timestamp'] - df.groupby('user_id')['timestamp'].shift(1)).dt.total_seconds().fillna(0)
df['amount_change'] = df['amount'] - df['last_transaction_amount']

# Save generated data (optional)
df.to_csv('smart_fraud_data.csv', index=False)

# --- Prepare data for modeling ---
data = df.drop(['timestamp', 'user_id', 'merchant_id'], axis=1).copy()

# Additional frequency features
data['user_transaction_freq'] = df.groupby('user_id')['user_id'].transform('size')
data['merchant_transaction_freq'] = df.groupby('merchant_id')['merchant_id'].transform('size')

# Encode categorical variables
categorical_cols = ['category', 'location', 'device']
label_encoders = {}

class SafeLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = np.append(y, 'UNSEEN')
        return super().fit(y)

    def transform(self, y):
        return super().transform(np.array([item if item in self.classes_ else 'UNSEEN' for item in y]))

for col in categorical_cols:
    le = SafeLabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop('label', axis=1)
y = data['label']

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
                                                    stratify=y_resampled, random_state=42)

# --- Model Training ---
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,
    use_label_encoder=False
)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Predictions & evaluation
y_pred = xgb.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model and encoders ---
joblib.dump({'model': xgb, 'encoders': label_encoders}, 'fraud_detection_model.joblib')
