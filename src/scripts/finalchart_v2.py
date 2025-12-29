import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder # <--- Added this

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "clean_cicids_combined.parquet"
RANDOM_STATE = 42
SAVE_DPI = 300 

# Set style for academic charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def plot_comparison_charts():
    print("[1/3] Generating Comparison Charts (Log Scale)...")
    
    # DATA FROM YOUR SPRINT 6 LOGS
    models = ['PCA-SVM (Proposed)', 'XGBoost (Baseline)']
    speed_pps = [1101, 471830]
    accuracy = [95.75, 99.86]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # CHART A: THROUGHPUT (Log Scale)
    # Fixed: Added hue=models and legend=False to silence warnings
    sns.barplot(x=models, y=speed_pps, hue=models, ax=ax1, palette=['#4c72b0', '#55a868'], legend=False)
    ax1.set_yscale("log") 
    ax1.set_title('Throughput Comparison (Log Scale)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Packets Per Second (PPS)', fontsize=12)
    ax1.set_xlabel('')
    
    # Add numbers on top
    for i, v in enumerate(speed_pps):
        ax1.text(i, v * 1.2, f"{v:,.0f} PPS", ha='center', fontweight='bold')

    # CHART B: ACCURACY
    sns.barplot(x=models, y=accuracy, hue=models, ax=ax2, palette=['#4c72b0', '#55a868'], legend=False)
    ax2.set_ylim(80, 102) 
    ax2.set_title('Accuracy Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xlabel('')
    
    for i, v in enumerate(accuracy):
        ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('chart_comparison_pps.png', dpi=SAVE_DPI)
    print("      -> Saved 'chart_comparison_pps.png'")

def plot_pca_scree():
    print("[2/3] Generating PCA Scree Plot...")
    
    try:
        pca = joblib.load("sprint6_pca_model.pkl")
    except FileNotFoundError:
        print("      [!] Error: 'sprint6_pca_model.pkl' not found. Skipping PCA chart.")
        return

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(exp_var_cumul) + 1), exp_var_cumul, marker='o', linestyle='--', color='b')
    plt.title('PCA Scree Plot: Cumulative Explained Variance', fontsize=16)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Variance Explained', fontsize=12)
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('chart_pca_scree.png', dpi=SAVE_DPI)
    print("      -> Saved 'chart_pca_scree.png'")

def plot_confusion_matrix():
    print("[3/3] Generating Confusion Matrix...")
    
    try:
        df = pd.read_parquet(DATA_PATH)
        svm_model = joblib.load("sprint6_svm_model.pkl")
        scaler = joblib.load("sprint6_scaler.pkl")
        pca = joblib.load("sprint6_pca_model.pkl")
    except FileNotFoundError as e:
        print(f"      [!] Error: Missing file ({e}). Skipping Confusion Matrix.")
        return

    # 1. PREPARE DATA
    cols_to_drop = ['Label']
    if 'Attack_Type' in df.columns: cols_to_drop.append('Attack_Type')
    
    X = df.drop(columns=cols_to_drop)
    y_raw = df['Label']

    # 2. ENCODE TARGET (THE FIX)
    # We must convert strings "BENIGN" to numbers 0, 1... just like training
    le = LabelEncoder()
    y = le.fit_transform(y_raw) 
    print(f"      Classes Encoded: {le.classes_}")

    # 3. SPLIT (Must be identical to training split)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # 4. TRANSFORM
    print("      Transforming test data...")
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 5. PREDICT
    print("      Running prediction (this takes a moment)...")
    y_pred = svm_model.predict(X_test_pca)
    
    # 6. PLOT
    # We use a simple layout because with 15 classes, numbers can get messy
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    # Using 'log' norm would help see small errors, but standard linear is safer for thesis
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', cbar=True) 
    
    plt.title('Confusion Matrix: PCA-SVM (Proposed)', fontsize=16)
    plt.ylabel('True Label (Encoded)', fontsize=12)
    plt.xlabel('Predicted Label (Encoded)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('chart_confusion_matrix.png', dpi=SAVE_DPI)
    print("      -> Saved 'chart_confusion_matrix.png'")

if __name__ == "__main__":
    plot_comparison_charts()
    plot_pca_scree()
    plot_confusion_matrix()
    print("\n[DONE] All charts generated successfully.")