import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ==========================================
# 1. LOAD STRESS DATA
# ==========================================
print("[*] Loading Stress Data...")
X_train = np.load('X_train_stress.npy')
y_train = np.load('y_train_stress.npy')
X_test = np.load('X_test_stress.npy')
y_test = np.load('y_test_stress.npy')

# Load Encoder for proper class names
with open('label_encoder_stress.pkl', 'rb') as f:
    le = pickle.load(f)
target_names = [str(cls) for cls in le.classes_]

# ==========================================
# 2. PREPARE PCA (Recalculate for new data)
# ==========================================
print("[*] Fitting PCA on larger dataset...")
# Using 23 components as established in Phase D
pca = PCA(n_components=23)
start_pca = time.time()
X_train_pca = pca.fit_transform(X_train)
pca_time = time.time() - start_pca
X_test_pca = pca.transform(X_test)
print(f"   -> PCA Fit/Transform Time: {pca_time:.2f}s")

# ==========================================
# 3. EXPERIMENT ENGINE
# ==========================================
def run_model(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\n{'='*40}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*40}")
    
    # TRAIN
    start_train = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start_train
    print(f"   -> Training Time: {train_time:.2f}s")
    
    # INFERENCE (THROUGHPUT TEST)
    start_inf = time.time()
    y_pred = model.predict(X_te)
    inf_time = time.time() - start_inf
    
    # Calculate Throughput (Packets Per Second)
    pps = X_te.shape[0] / inf_time
    print(f"   -> Inference Time: {inf_time:.2f}s")
    print(f"   -> THROUGHPUT: {pps:,.0f} packets/sec")
    
    # ACCURACY
    acc = accuracy_score(y_te, y_pred)
    print(f"   -> Accuracy: {acc:.4f}")
    
    return {
        "name": name,
        "train_time": train_time,
        "pps": pps,
        "accuracy": acc,
        "report": classification_report(y_te, y_pred, target_names=target_names)
    }

results = []

# ==========================================
# MODEL 1: PCA-SVM (The Thesis Proposal)
# ==========================================
# Cache size increased to 2000MB (2GB) to help with larger data
svm = SVC(kernel='rbf', cache_size=2000, class_weight='balanced')
res_svm = run_model("PCA-SVM (Proposed)", svm, X_train_pca, y_train, X_test_pca, y_test)
results.append(res_svm)

# ==========================================
# MODEL 2: XGBoost (The Industry Standard)
# ==========================================
# tree_method='hist' is optimized for speed on large datasets
xgb = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    tree_method='hist', 
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)
# XGBoost handles raw features well, no PCA needed usually.
# We test it on RAW features to see if "Complex SVM" is worse than "Simple XGB"
res_xgb = run_model("XGBoost (Raw Features)", xgb, X_train, y_train, X_test, y_test)
results.append(res_xgb)

# ==========================================
# 4. FINAL COMPARISON
# ==========================================
print("\n" + "#"*60)
print("FINAL STRESS TEST RESULTS")
print("#"*60)
print(f"{'Model':<25} | {'Train Time':<10} | {'Throughput (PPS)':<20} | {'Accuracy':<10}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<25} | {r['train_time']:<10.2f} | {r['pps']:<20,.0f} | {r['accuracy']:<10.4f}")
print("#"*60)

# Save reports for analysis
for r in results:
    with open(f"stress_report_{r['name'].split()[0]}.txt", "w") as f:
        f.write(r['report'])