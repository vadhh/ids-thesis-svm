import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle
import collections

# ==========================================
# 1. SETUP & LOAD
# ==========================================
print("[*] Loading cleaned data from Phase A...")
df = pd.read_parquet('clean_cicids_combined.parquet')

X = df.drop(columns=['Label', 'Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], errors='ignore')
y = df['Label']

# ==========================================
# 2. ENCODING
# ==========================================
print("[*] Encoding Labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# ==========================================
# 3. SPLITTING
# ==========================================
print("[*] Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 4. INTELLIGENT UNDERSAMPLING (THE FIX)
# ==========================================
print("[*] Calculating Dynamic Sampling Strategy...")

# Count instances per class in the training set
class_counts = collections.Counter(y_train)
print(f"   -> Raw Counts: {class_counts}")

# THESIS STRATEGY: 
# Cap any class with > 10,000 samples down to 10,000.
# Keep all minority classes exactly as they are.
# This ensures we have enough data for SVM (~50k total) but don't delete rare attacks.

TARGET_CAP = 10000 
sampling_strategy = {}

for label, count in class_counts.items():
    if count > TARGET_CAP:
        sampling_strategy[label] = TARGET_CAP
    else:
        # For rare classes, RandomUnderSampler requires us to NOT include them 
        # in the dict if we want to keep them as-is, OR specify their full count.
        # Ideally, we only put keys in the dict that we want to RESAMPLE.
        # But imblearn is tricky. Easiest way: specify the count explicitly for all.
        sampling_strategy[label] = count

print(f"   -> Target strategy (Capping max classes at {TARGET_CAP})...")

rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print(f"   -> Resampled Training Shape: {X_train_res.shape}")
print("   -> (This should be between 20,000 and 100,000 rows)")

# ==========================================
# 5. STANDARD SCALING
# ==========================================
print("[*] Applying StandardScaler...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test) # Transform test using train stats

# ==========================================
# 6. SAVING
# ==========================================
print("[*] Saving processed artifacts...")

np.save('X_train_scaled.npy', X_train_scaled)
np.save('y_train_res.npy', y_train_res)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_test.npy', y_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
    
print("\n[SUCCESS] Phase C Complete. READY for PCA.")