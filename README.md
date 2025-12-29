# 🛡️ Optimized Network Intrusion Detection System (IDS) using PCA & SVM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Research_Phase-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Original Title:** *Implementasi Reduksi Dimensi dengan Principal Component Analysis (PCA) untuk Optimalisasi Kinerja Klasifikasi Intrusi Jaringan Menggunakan Algoritma Support Vector Machine (SVM)*

## 📖 Abstract
Network Intrusion Detection Systems (NIDS) often struggle with the "Curse of Dimensionality" due to the high volume and complexity of modern network traffic.

This research proposes a hybrid approach using **Principal Component Analysis (PCA)** for dimensionality reduction and **Support Vector Machines (SVM)** for classification. The goal is to optimize the trade-off between **detection accuracy** and **computational efficiency**, making real-time detection feasible on resource-constrained environments.

## 🗂️ Dataset
**CIC-IDS-2017** (Canadian Institute for Cybersecurity)
Unlike the outdated KDD99, this dataset contains modern attack scenarios (Brute Force, Heartbleed, Botnet, DDoS, Web Attacks) and reflects realistic network traffic.
* **Instances:** 2.8 Million+ flows
* **Original Features:** 78 (High Dimensionality)
* *Note: Due to file size limits, the raw dataset is not included in this repo. Please download it from the official source.*

## ⚙️ Methodology
The research pipeline follows a strict data science lifecycle:

1.  **Data Preprocessing:**
    * Handling `Infinity` and `NaN` values commonly found in network flow data.
    * Label Encoding for categorical features.
    * Min-Max Scaling to normalize feature magnitude for SVM sensitivity.
2.  **Dimensionality Reduction (PCA):**
    * Variance analysis to determine the optimal number of components ($k$).
    * Projecting 78 features down to $k$ components while retaining 95%+ cumulative variance.
3.  **Classification (SVM):**
    * Kernel selection (RBF vs Linear).
    * Hyperparameter tuning using GridSearch.
4.  **Evaluation:**
    * Metrics: Accuracy, Precision, Recall, F1-Score, and **Training/Inference Time**.

## 📊 Preliminary Results
*By reducing the feature space, we achieved comparable accuracy with significantly lower training times.*

| Approach | Features | Training Time (s) | Accuracy | Detection Rate (Recall) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (SVM only)** | 78 | ~450s | 98.2% | 97.5% |
| **Proposed (PCA + SVM)** | **N** (Optimized) | **~85s** | **97.9%** | **97.1%** |

*(Note: These are indicative figures; official results will be updated upon thesis defense.)*

## 💻 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
### Running the Analysis
The project is structured into modular scripts/notebooks:

### Exploratory Data Analysis (EDA):

```Bash
jupyter notebook notebooks/01_EDA_CIC_IDS_2017.ipynb
```
### PCA Variance Analysis:

```Bash
python src/pca_analysis.py --plot-variance
```

### Model Training:

```Bash
python src/train_model.py --kernel rbf --components 15
```

## 📉 Visualizations

<img width="3000" height="1800" alt="chart_pca_scree" src="https://github.com/user-attachments/assets/06487a4e-037e-43ed-9e45-3692bc237090" />

## 📚 References
```
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.
Jolliffe, I. T. (2002). Principal Component Analysis.
```
Author: vadhh Bachelor Thesis - 7th Semester
