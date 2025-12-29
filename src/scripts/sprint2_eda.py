import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================
print("[*] Loading cleaned dataset from Parquet...")
df = pd.read_parquet('clean_cicids_combined.parquet')

# ==========================================
# 2. VISUALIZE CLASS IMBALANCE
# ==========================================
print("[*] Generating Class Distribution Plot...")

plt.figure(figsize=(12, 6))
# Use log scale because 'Benign' is likely 100x or 1000x larger than attacks
ax = sns.countplot(y=df['Label'], order=df['Label'].value_counts().index)
ax.set_xscale("log") 

plt.title("Distribution of Network Traffic Classes (Log Scale)")
plt.xlabel("Count (Log Scale)")
plt.ylabel("Attack Type")
plt.tight_layout()
plt.savefig("thesis_class_imbalance.png") # Save for your thesis document
print("   -> Saved 'thesis_class_imbalance.png'")
plt.show()

# ==========================================
# 3. VISUALIZE CORRELATION (Justification for PCA)
# ==========================================
print("[*] Generating Correlation Heatmap...")
# We must exclude the 'Label' column and any other non-numeric identifiers
# Selecting only float/int columns
numeric_df = df.select_dtypes(include=[np.number])

# Correlation calculation (Pearson)
# NOTE: This is computationally expensive on 2.8M rows. 
# We'll take a representative sample (e.g., 100k rows) to speed up plotting
# without losing the general correlation structure.
sample_df = numeric_df.sample(n=100000, random_state=42)
corr_matrix = sample_df.corr()

plt.figure(figsize=(20, 16))
# Mask the upper triangle (since matrix is symmetrical) to reduce visual clutter
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, 
            mask=mask, 
            cmap='coolwarm', 
            vmax=1.0, vmin=-1.0, 
            center=0, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            xticklabels=False, # Hiding labels because 78 features makes text unreadable
            yticklabels=False)

plt.title("Feature Correlation Matrix (Sampled N=100k)")
plt.tight_layout()
plt.savefig("thesis_correlation_heatmap.png")
print("   -> Saved 'thesis_correlation_heatmap.png'")
plt.show()

# ==========================================
# 4. STATISTICAL SUMMARY
# ==========================================
print("\n[*] Top 5 Most Frequent Classes:")
print(df['Label'].value_counts().head())

print(f"\n[*] Total Features Available for PCA: {len(numeric_df.columns)}")