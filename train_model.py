"""
train_model.py
--------------
Replicates the full pipeline from the project notebook:
  1. Load P670-dataset.xlsx
  2. Remove & cap outliers (IQR method)
  3. One-hot encode categoricals (state, area.code, voice.plan, intl.plan, churn)
  4. Feature engineering  (total_call_minutes, total_calls_made, total_charge_cost)
  5. MinMaxScaler
  6. Train Random Forest
  7. Save  churn_model.pkl  +  scaler.pkl  +  feature_columns.pkl

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
DATA_PATH = "P670-dataset.xlsx"

print("Loading data...")
df = pd.read_excel(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Fix columns that were stored as object despite being numeric
df["day.charge"] = pd.to_numeric(df["day.charge"], errors="coerce")
df["eve.mins"]   = pd.to_numeric(df["eve.mins"],   errors="coerce")

print(f"  Raw shape : {df.shape}")
print(f"  Churn rate: {(df['churn'] == 'yes').mean():.2%}")

# --------------------------------------------------
# 2. OUTLIER REMOVAL  (IQR method — drop then cap)
# --------------------------------------------------
print("\nRemoving outliers...")
df_clean = df.copy()
num_cols = df_clean.select_dtypes(include=np.number).columns

for col in num_cols:
    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lb) & (df_clean[col] <= ub)]

print(f"  Shape after removal : {df_clean.shape}")

# Capping pass (handles any residual outliers)
for col in df_clean.select_dtypes(include=np.number).columns:
    q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    iqr = q3 - q1
    df_clean[col] = np.clip(df_clean[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)

print("  Outlier capping done.")

# --------------------------------------------------
# 3. ONE-HOT ENCODING
# --------------------------------------------------
print("\nEncoding categoricals...")
df_enc = pd.get_dummies(
    df_clean,
    columns=["state", "area.code", "voice.plan", "intl.plan", "churn"]
)

# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------
print("Engineering features...")
df_enc["total_call_minutes"] = (
    df_enc["day.mins"] + df_enc["night.mins"] + df_enc["intl.mins"]
)
df_enc["total_calls_made"] = (
    df_enc["day.calls"] + df_enc["eve.calls"] + df_enc["night.calls"]
    + df_enc["intl.calls"] + df_enc["customer.calls"]
)
df_enc["total_charge_cost"] = (
    df_enc["intl.charge"] + df_enc["eve.charge"] + df_enc["night.charge"]
)

# --------------------------------------------------
# 5. SCALING
# --------------------------------------------------
print("Scaling with MinMaxScaler...")
scaler = MinMaxScaler()
scaled_arr = scaler.fit_transform(df_enc)
minmax_df = pd.DataFrame(scaled_arr, columns=df_enc.columns)

# --------------------------------------------------
# 6. FEATURES / TARGET
# --------------------------------------------------
X = minmax_df.drop(["churn_no", "churn_yes"], axis=1)
y = minmax_df["churn_yes"]

print(f"\n  Features : {X.shape[1]}")
print(f"  Samples  : {X.shape[0]}")
print(f"  Churn %  : {(y > 0.5).mean():.2%}")

# --------------------------------------------------
# 7. TRAIN / TEST SPLIT
# --------------------------------------------------
y_binary = (y > 0.5).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binary
)

# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --------------------------------------------------
# 9. EVALUATE
# --------------------------------------------------
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
y_test_binary = (y_test > 0.5).astype(int)

print(f"\nAccuracy : {accuracy_score(y_test_binary, y_pred_binary):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary,
                            target_names=["No Churn", "Churn"]))

# --------------------------------------------------
# 10. SAVE ARTIFACTS
# --------------------------------------------------
joblib.dump(model,          "churn_model.pkl")
joblib.dump(scaler,         "scaler.pkl")
joblib.dump(list(X.columns),"feature_columns.pkl")   # so app.py always uses the right columns

print("✅ Saved: churn_model.pkl | scaler.pkl | feature_columns.pkl")
