import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. LOAD ARTIFACTS
# ==========================================
print("[*] Loading processed data...")
X_train = np.load('X_train_scaled.npy')
y_train = np.load('y_train_res.npy')
X_test = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

print(f"   -> Training Set: {X_train.shape}")
print(f"   -> Test Set:     {X_test.shape}")

# Load PCA Model
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)
    
print(f"   -> PCA Loaded: {pca.n_components_} components")

# ==========================================
# 2. PREPARE DATASETS
# ==========================================
# Scenario A: Original Features (Already loaded in X_train)
# Scenario B: PCA Features (We must transform the data)

print("[*] Transforming data for Scenario B (PCA)...")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Load Label Encoder for readable report
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
target_names = [str(cls) for cls in le.classes_]

# ==========================================
# 3. UTILITY FUNCTION: TRAIN & EVALUATE
# ==========================================
def run_svm_experiment(name, X_tr, y_tr, X_te, y_te):
    print(f"\n{'='*40}")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"{'='*40}")
    
    # Initialize SVM
    # cache_size=1000 allows SVM to use 1GB RAM for cache (speeds up training)
    # class_weight='balanced' ensures rare attacks aren't ignored
    clf = SVC(kernel='rbf', random_state=42, cache_size=1000, class_weight='balanced')
    
    # 1. MEASURE TRAINING TIME
    start_time = time.time()
    clf.fit(X_tr, y_tr)
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"   -> Training Complete.")
    print(f"   -> Time Taken: {training_time:.2f} seconds")
    
    # 2. PREDICT
    print(f"   -> Predicting on Test Set ({X_te.shape[0]} samples)...")
    y_pred = clf.predict(X_te)
    
    # 3. METRICS
    acc = accuracy_score(y_te, y_pred)
    print(f"   -> Accuracy: {acc:.4f}")
    
    # Save Report to Text File
    report = classification_report(y_te, y_pred, target_names=target_names)
    with open(f"results_{name}_report.txt", "w") as f:
        f.write(report)
        
    # Generate Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    
    return {
        "name": name,
        "time": training_time,
        "accuracy": acc,
        "cm": cm,
        "report": report
    }

# ==========================================
# 4. RUN SCENARIO A: ORIGINAL FEATURES
# ==========================================
results_orig = run_svm_experiment("Original_78_Features", X_train, y_train, X_test, y_test)

# ==========================================
# 5. RUN SCENARIO B: PCA FEATURES
# ==========================================
results_pca = run_svm_experiment("PCA_23_Features", X_train_pca, y_train, X_test_pca, y_test)

# ==========================================
# 6. VISUALIZE COMPARISON
# ==========================================
print("\n[*] Generating Confusion Matrices...")

def plot_cm(cm, title, filename):
    plt.figure(figsize=(12, 10))
    # Normalize by row to show recall percentages (better for imbalanced data)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=False, cmap='Blues', fmt='.2f', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"   -> Saved {filename}")

plot_cm(results_orig['cm'], "Confusion Matrix (Original Features)", "cm_original.png")
plot_cm(results_pca['cm'], "Confusion Matrix (PCA Features)", "cm_pca.png")

# ==========================================
# 7. FINAL THESIS SUMMARY
# ==========================================
print("\n" + "#"*50)
print("FINAL THESIS RESULTS SUMMARY")
print("#"*50)
print(f"{'Metric':<20} | {'Original (78 Feat)':<20} | {'PCA (23 Feat)':<20}")
print("-" * 66)
print(f"{'Training Time (s)':<20} | {results_orig['time']:<20.2f} | {results_pca['time']:<20.2f}")
print(f"{'Accuracy':<20} | {results_orig['accuracy']:<20.4f} | {results_pca['accuracy']:<20.4f}")

# Calculate Speedup
speedup = results_orig['time'] / results_pca['time']
print("-" * 66)
print(f"OPTIMIZATION RESULT: PCA was {speedup:.2f}x faster.")
print("#"*50)