"""
Automation Script for GPU Data Preprocessing
Author: Shidqi Ahmad Musyaffa'

Script ini mengotomatisasi langkah preprocessing yang sudah dieksperimen
di notebook Eksperimen_Shidqi.ipynb.

Usage:
    python automate_Shidqi.py
    
Atau sebagai module:
    from automate_Shidqi import preprocess_data
    df = preprocess_data('path/to/input.csv', 'path/to/output/')
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import argparse


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocess GPU dataset.
    
    Langkah preprocessing (sama dengan notebook eksperimen):
    1. Load data
    2. Konversi kolom numerik
    3. Drop kolom dengan banyak missing (pixelShader, vertexShader)
    4. Imputation dengan median untuk kolom numerik
    5. Label encoding untuk kolom kategorikal
    6. MinMax scaling untuk kolom numerik
    7. Feature selection
    8. Save ke output path
    
    Args:
        input_path (str): Path ke raw dataset (csv)
        output_path (str): Path folder untuk menyimpan hasil preprocessing
    
    Returns:
        pd.DataFrame: Dataset yang sudah dipreprocessing
    """
    
    print("="*50)
    print("GPU DATA PREPROCESSING")
    print("="*50)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # ========================================
    # 2. KONVERSI KOLOM NUMERIK
    # ========================================
    print("\n[2/7] Converting numeric columns...")
    numeric_columns = [
        'memSize', 'memBusWidth', 'gpuClock', 'memClock', 
        'unifiedShader', 'tmu', 'rop', 'pixelShader', 'vertexShader'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"    Converted {len(numeric_columns)} columns to numeric type")
    
    # ========================================
    # 3. DROP KOLOM OBSOLETE
    # ========================================
    print("\n[3/7] Dropping obsolete columns...")
    columns_to_drop = ['pixelShader', 'vertexShader']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    print(f"    Dropped columns: {existing_cols_to_drop}")
    
    # ========================================
    # 4. IMPUTATION MISSING VALUES
    # ========================================
    print("\n[4/7] Imputing missing values with median...")
    numeric_cols_remaining = [
        'releaseYear', 'memSize', 'memBusWidth', 'gpuClock', 'memClock', 
        'unifiedShader', 'tmu', 'rop'
    ]
    
    imputation_log = []
    for col in numeric_cols_remaining:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                imputation_log.append(f"    {col}: filled {missing_count} values with median={median_val:.2f}")
    
    for log in imputation_log:
        print(log)
    
    if not imputation_log:
        print("    No missing values found in numeric columns")
    
    # ========================================
    # 5. LABEL ENCODING
    # ========================================
    print("\n[5/7] Encoding categorical columns...")
    columns_to_encode = ['manufacturer', 'igp', 'bus', 'memType', 'gpuChip']
    label_encoders = {}
    
    for col in columns_to_encode:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"    {col}: {len(le.classes_)} unique values encoded")
    
    # Drop original categorical columns (except productName)
    columns_to_remove = [col for col in columns_to_encode if col in df.columns]
    df = df.drop(columns=columns_to_remove)
    
    # ========================================
    # 6. MINMAX SCALING
    # ========================================
    print("\n[6/7] Scaling numeric columns (MinMax 0-1)...")
    columns_to_scale = [
        'releaseYear', 'memSize', 'memBusWidth', 'gpuClock', 
        'memClock', 'unifiedShader', 'tmu', 'rop'
    ]
    
    scaler = MinMaxScaler()
    scaled_count = 0
    
    for col in columns_to_scale:
        if col in df.columns:
            df[col + '_scaled'] = scaler.fit_transform(df[[col]])
            scaled_count += 1
    
    print(f"    Scaled {scaled_count} columns")
    
    # ========================================
    # 7. FEATURE SELECTION & SAVE
    # ========================================
    print("\n[7/7] Selecting final features and saving...")
    
    # Define final features
    feature_numeric = [col + '_scaled' for col in columns_to_scale if col + '_scaled' in df.columns]
    feature_encoded = [col + '_encoded' for col in columns_to_encode if col + '_encoded' in df.columns]
    
    # Create final dataset
    final_columns = ['productName'] + feature_numeric + feature_encoded
    df_final = df[[col for col in final_columns if col in df.columns]].copy()
    
    # Create output folder if not exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_path, 'gpu_data_processed.csv')
    df_final.to_csv(output_file, index=False)
    
    print(f"    Saved to: {output_file}")
    print(f"    Final shape: {df_final.shape}")
    print(f"    Missing values: {df_final.isnull().sum().sum()}")
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return df_final


def main():
    """
    Main function untuk menjalankan preprocessing dari command line.
    """
    parser = argparse.ArgumentParser(description='GPU Data Preprocessing Script')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../dataset/gpu_data.csv',
        help='Path to input raw dataset (default: ../dataset/gpu_data.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='gpu_data_preprocessing',
        help='Output folder path (default: gpu_data_preprocessing)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return None
    
    # Run preprocessing
    df_processed = preprocess_data(args.input, args.output)
    
    return df_processed


if __name__ == "__main__":
    main()
