import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ==========================================
# 1. LOAD DATA
# ==========================================
print("[*] Loading scaled training data...")
# We ONLY use the training set to decide PCA components. 
# The test set must blindly follow this transformation later.
X_train = np.load('X_train_scaled.npy')

print(f"   -> Data Shape: {X_train.shape}")

# ==========================================
# 2. RUN PCA (FULL VARIANCE)
# ==========================================
print("[*] Fitting PCA to capture full variance...")
# We fit with all components first to see the curve
pca = PCA()
pca.fit(X_train)

# Calculate cumulative variance
cum_var = np.cumsum(pca.explained_variance_ratio_)

# Find the exact number of components for 95% variance
# np.argmax returns the first index where the condition is True
n_components_95 = np.argmax(cum_var >= 0.95) + 1 

print("-" * 40)
print(f"RESULTS FOR THESIS REPORT:")
print(f"   -> Original Features: {X_train.shape[1]}")
print(f"   -> Components for 95% Variance: {n_components_95}")
print(f"   -> Feature Reduction: {X_train.shape[1]} -> {n_components_95}")
print(f"   -> Information Loss: 5%")
print("-" * 40)

# ==========================================
# 3. GENERATE THESIS PLOTS (Scree & Cumulative)
# ==========================================
print("[*] Generating Scree Plot and Cumulative Variance Plot...")

plt.figure(figsize=(12, 6))

# Plot 1: Explained Variance per Component (Scree Plot)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, 
        alpha=0.5, 
        align='center', 
        label='Individual explained variance')

# Plot 2: Cumulative Explained Variance
plt.step(range(1, len(cum_var) + 1), 
         cum_var, 
         where='mid', 
         color='red', 
         label='Cumulative explained variance')

# Visual Marker for 95% Cutoff
plt.axhline(y=0.95, color='k', linestyle='--', label='95% Threshold')
plt.axvline(x=n_components_95, color='k', linestyle='--', label=f'Optimal Components ({n_components_95})')

plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title(f'PCA Scree Plot: {n_components_95} Components retain 95% Variance')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.savefig("thesis_pca_scree_plot.png")
print("   -> Saved 'thesis_pca_scree_plot.png'")
plt.show()

# ==========================================
# 4. SAVE THE OPTIMAL PCA MODEL
# ==========================================
# Now we refit PCA with exactly the optimal number of components
# so we can use it in the final phase.

print(f"[*] Saving optimized PCA model (n={n_components_95})...")
pca_opt = PCA(n_components=n_components_95)
pca_opt.fit(X_train)

import pickle
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_opt, f)

print("[SUCCESS] Phase D Complete. We have the optimal dimensionality.")