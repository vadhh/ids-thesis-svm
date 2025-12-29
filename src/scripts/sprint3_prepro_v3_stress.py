import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle
import collections

# ==========================================
# 1. SETUP
# ==========================================
print("[*] Loading cleaned data...")
df = pd.read_parquet('clean_cicids_combined.parquet')

X = df.drop(columns=['Label', 'Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], errors='ignore')
y = df['Label']

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split (80/20)
print("[*] Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 2. AGGRESSIVE SAMPLING (The "Stress Test")
# ==========================================
print("[*] Calculating Stress-Test Sampling Strategy...")
class_counts = collections.Counter(y_train)

# NEW STRATEGY: 
# 1. Benign: Cap at 200,000 (Previously 10,000). 
#    This forces the SVM to learn "Normal" much better, fixing Precision.
# 2. Attacks: Keep ALL attack samples (Don't undersample attacks).
BENIGN_CAP = 200000 
sampling_strategy = {}

for label, count in class_counts.items():
    label_name = le.inverse_transform([label])[0]
    if label_name == "BENIGN":
        sampling_strategy[label] = BENIGN_CAP
    else:
        # Keep original count for all attacks
        sampling_strategy[label] = count

print(f"   -> Strategy: Benign={BENIGN_CAP}, Attacks=Original Counts")

rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print(f"   -> New Training Shape: {X_train_res.shape}")
print(f"   -> (Warning: This is ~3x larger than before. SVM training will be slow.)")

# ==========================================
# 3. SCALING & SAVING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

print("[*] Saving Stress-Test Artifacts...")
np.save('X_train_stress.npy', X_train_scaled)
np.save('y_train_stress.npy', y_train_res)
np.save('X_test_stress.npy', X_test_scaled)
np.save('y_test_stress.npy', y_test)

with open('scaler_stress.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder_stress.pkl', 'wb') as f:
    pickle.dump(le, f)

print("[SUCCESS] Dataset Ready for Stress Testing.")