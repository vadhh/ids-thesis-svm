import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from xgboost import XGBClassifier

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "clean_cicids_combined.parquet"
TRAIN_LIMIT = 150000  # <--- THE FIX. 150k is large but feasible. 600k is suicide.
RANDOM_STATE = 42

def load_and_prep_data(filepath):
    print(f"[*] Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    
    # Separate Features and Target
    # Separate Features and Target
    print(f"    Dataset Columns: {df.columns.tolist()}") # <--- SEE what you actually have
    
    cols_to_drop = ['Label']
    # Only drop Attack_Type if it actually exists
    if 'Attack_Type' in df.columns:
        cols_to_drop.append('Attack_Type')
        
    X = df.drop(columns=cols_to_drop)
    y = df['Label']
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"    Original Shape: {X.shape}")
    return X, y_encoded, le

def run_benchmark(model, model_name, X_train, y_train, X_test, y_test, use_pca=False, pca_model=None):
    print(f"\n--- Benchmarking: {model_name} ---")
    
    # 1. Training Phase
    start_train = time.time()
    
    if use_pca:
        print("    Transforming Training Data with PCA...")
        X_train_in = pca_model.transform(X_train)
        X_test_in = pca_model.transform(X_test)
    else:
        X_train_in = X_train
        X_test_in = X_test
        
    print(f"    Training started (Rows: {len(X_train_in)})...")
    model.fit(X_train_in, y_train)
    train_time = time.time() - start_train
    print(f"    [DONE] Training Time: {train_time:.2f} seconds ({train_time/60:.2f} min)")

    # 2. Inference Phase (Throughput Test)
    print("    Running Inference Speed Test...")
    start_pred = time.time()
    y_pred = model.predict(X_test_in)
    pred_time = time.time() - start_pred
    
    # Calculate Throughput (Packets Per Second)
    pps = len(X_test) / pred_time
    print(f"    [DONE] Prediction Time: {pred_time:.4f}s")
    print(f"    >> SPEED: {pps:.0f} packets/sec")

    # 3. Evaluation
    print("    Generating Report...")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Return metrics for summary
    return {
        "Model": model_name,
        "Train Time (s)": round(train_time, 2),
        "Inference Speed (PPS)": round(pps, 0),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4)
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    X, y, le = load_and_prep_data(DATA_PATH)

    # 2. Split Data (Standard Split first)
    print("[*] Splitting Dataset...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3. APPLY THE LIMIT (The Fix)
    # We take a stratified subset of the training data to prevent infinite training time
    print(f"[*] Downsampling Training Data to {TRAIN_LIMIT} rows (Stratified)...")
    X_train, _, y_train, _ = train_test_split(
        X_train_full, y_train_full, 
        train_size=TRAIN_LIMIT, 
        stratify=y_train_full, 
        random_state=RANDOM_STATE
    )
    
    # Scaling is crucial for SVM and PCA
    print("[*] Scaling Data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Scale full test set to prove generalization

    results = []

    # ==========================================
    # EXPERIMENT A: PROPOSED METHOD (PCA + SVM)
    # ==========================================
    
    # 1. Fit PCA
    print("[*] Fitting PCA...")
    pca = PCA(n_components=0.95) # Keep 95% variance
    pca.fit(X_train_scaled)
    n_components = pca.n_components_
    print(f"    PCA Reduced features from {X_train.shape[1]} to {n_components}")

    # 2. Define SVM (With Verbosity!)
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', cache_size=1000, verbose=3)
    
    # 3. Run Benchmark
    res_svm = run_benchmark(svm_model, "PCA-SVM (Proposed)", 
                            X_train_scaled, y_train, 
                            X_test_scaled, y_test, 
                            use_pca=True, pca_model=pca)
    results.append(res_svm)
    
    # Save the SVM model immediately in case it crashes later
    print("[*] Saving SVM Model...")
    joblib.dump(svm_model, "sprint6_svm_model.pkl")
    joblib.dump(pca, "sprint6_pca_model.pkl")
    joblib.dump(scaler, "sprint6_scaler.pkl")


    # ==========================================
    # EXPERIMENT B: MODERN BASELINE (XGBoost)
    # ==========================================
    # We compare against XGBoost without PCA to see if complexity is worth it
    
    xgb_model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        n_jobs=-1, # Use all CPU cores
        eval_metric='logloss'
    )

    # Note: XGBoost handles unscaled data well, but we use scaled for fair comparison
    res_xgb = run_benchmark(xgb_model, "XGBoost (Baseline)", 
                            X_train_scaled, y_train, 
                            X_test_scaled, y_test, 
                            use_pca=False)
    results.append(res_xgb)

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n\n================ FINAL BENCHMARK RESULTS ================")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print("=========================================================")