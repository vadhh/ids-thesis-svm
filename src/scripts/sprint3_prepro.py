import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle

# ==========================================
# 1. SETUP & LOAD
# ==========================================
print("[*] Loading cleaned data from Phase A...")
df = pd.read_parquet('clean_cicids_combined.parquet')

# Separate Features (X) and Target (y)
# Drop Label and non-predictive metadata if present
X = df.drop(columns=['Label', 'Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], errors='ignore')
y = df['Label']

print(f"   -> Features Shape: {X.shape}")
print(f"   -> Target Shape:   {y.shape}")

# ==========================================
# 2. ENCODING (Benign vs Attack)
# ==========================================
# Thesis Note: SVMs require numerical input. 
# We use LabelEncoder to turn strings into integers.
print("[*] Encoding Labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the encoder mappings so we can decode predictions later
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"   -> Class Mapping: {mapping}")

# ==========================================
# 3. SPLITTING (The Firewall)
# ==========================================
# CRITICAL: Split BEFORE Undersampling/Scaling to prevent data leakage.
# We hold out 20% for pure testing. 
print("[*] Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"   -> Original Training Set: {X_train.shape}")
print(f"   -> Test Set (Untouched):  {X_test.shape}")

# ==========================================
# 4. RANDOM UNDERSAMPLING (Solving O(n^3))
# ==========================================
# SVM cannot handle 2 million rows. We must reduce the Training set.
# Strategy: We will undersample the majority class ('BENIGN') to match 
# a reasonable number of attacks, or set a hard limit.

print("[*] Applying Random Undersampling to TRAINING SET ONLY...")

# Thesis Strategy: Auto-balance. 
# This reduces the majority class to the size of the minority class.
# WARNING: If your minority class is too small, this might be too aggressive.
# Alternatively, we can use a sampling_strategy dict to force specific counts.
# For this thesis, let's target a balanced training set for optimal SVM hyperplane finding.
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print(f"   -> Resampled Training Shape: {X_train_res.shape}")
print("   -> (This is the dataset size the SVM will actually see)")

# ==========================================
# 5. STANDARD SCALING (PCA Requirement)
# ==========================================
# PCA is variance-based. If one feature ranges 0-10000 and another 0-1, 
# PCA will think the first feature is more important. StandardScaler fixes this.

print("[*] Applying StandardScaler...")
scaler = StandardScaler()

# FIT only on Training data
X_train_scaled = scaler.fit_transform(X_train_res)

# TRANSFORM Test data (using stats from Training)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. SAVING ARTIFACTS
# ==========================================
# We save as Numpy arrays (npy) because they are much faster to load 
# for the PCA and SVM steps than CSV/Parquet.

print("[*] Saving processed artifacts for Phase D & E...")

np.save('X_train_scaled.npy', X_train_scaled)
np.save('y_train_res.npy', y_train_res)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_test.npy', y_test)

# Save the scaler and encoder objects for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
