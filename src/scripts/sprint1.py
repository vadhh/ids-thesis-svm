import pandas as pd
import numpy as np
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
# List of specific files to load based on your directory structure
FILES_TO_LOAD = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

def load_and_clean_dataset(file_list):
    """
    Loads CIC-IDS-2017 files, standardizes column names, 
    and removes Infinity/NaN values.
    """
    dataframes = []
    
    print(f"[*] Starting ingestion of {len(file_list)} files...")
    
    for file in file_list:
        if not os.path.exists(file):
            print(f"[!] Warning: File {file} not found. Skipping.")
            continue
            
        print(f"   -> Loading: {file}...")
        
        # Read CSV
        # low_memory=False avoids mixed-type warnings during initial read
        df = pd.read_csv(file, low_memory=False)
        
        # CRITICAL FIX 1: Strip whitespace from column headers
        # CIC-IDS-2017 headers are " Destination Port " -> "Destination Port"
        df.columns = df.columns.str.strip()
        
        dataframes.append(df)

    # Combine all days into one master dataframe
    print("[*] Concatenating datasets...")
    full_df = pd.concat(dataframes, ignore_index=True)
    
    initial_shape = full_df.shape
    print(f"[*] Initial Shape: {initial_shape}")

    # ==========================================
    # CLEANING PIPELINE
    # ==========================================
    
    # CRITICAL FIX 2: Handle "Infinity" strings
    # This dataset mixes numeric infinity with string "Infinity"
    print("[*] Cleaning 'Infinity' and 'NaN' values...")
    
    # Replace string variants of infinity with numpy infinity
    full_df.replace(["Infinity", "infinity"], np.inf, inplace=True)
    
    # Convert all columns to numeric (coerce errors to NaN), except the Label
    # We exclude 'Label' (target) and potentially 'Timestamp' if it exists
    cols_to_normalize = full_df.columns.difference(['Label', 'Timestamp', 'Flow ID', 'Source IP', 'Destination IP'])
    
    # Efficiently convert object columns to numeric where possible
    for col in cols_to_normalize:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Replace infinite values with NaN so we can drop them
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with any NaN values
    full_df.dropna(inplace=True)
    
    # Memory Optimization: Downcast floats to float32 to save RAM
    # (SVM and PCA don't need 64-bit precision for this thesis)
    fcols = full_df.select_dtypes('float').columns
    icols = full_df.select_dtypes('integer').columns
    
    full_df[fcols] = full_df[fcols].apply(pd.to_numeric, downcast='float')
    full_df[icols] = full_df[icols].apply(pd.to_numeric, downcast='integer')

    cleaned_shape = full_df.shape
    dropped_rows = initial_shape[0] - cleaned_shape[0]
    
    print("-" * 40)
    print(f"[*] Cleaning Complete.")
    print(f"    Original Rows: {initial_shape[0]}")
    print(f"    Cleaned Rows:  {cleaned_shape[0]}")
    print(f"    Dropped Rows:  {dropped_rows} (containing NaN or Infinity)")
    print("-" * 40)
    
    return full_df

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load data
    df = load_and_clean_dataset(FILES_TO_LOAD)
    
    # Quick Label Check to ensure we have Benign AND Attacks
    print("\nClass Distribution (Top 10):")
    print(df['Label'].value_counts().head(10))
    
    #Save a temp file if you want, or just hold in memory for next step
    print("[*] Saving checkpoint to 'clean_cicids_combined.parquet'...")
    df.to_parquet('clean_cicids_combined.parquet')