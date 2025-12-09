import pandas as pd
import os

try:
    df = pd.read_excel('dataset/diagnosi.xls')
    print("COLUMNS: " + ", ".join(df.columns.tolist()))
    
    for col in df.columns:
        try:
            # Try converting to numeric
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            valid_nums = numeric_col.dropna()
            if not valid_nums.empty:
                print(f"Column '{col}' is numeric. Min: {valid_nums.min()}, Max: {valid_nums.max()}, Count: {len(valid_nums)}")
                if valid_nums.max() >= 100: # We have M-124
                    print(f"!!! CANDIDATE ID COLUMN: {col} !!!")
        except:
            pass
            
    print("-" * 20)
    # Print first few rows of all columns
    print(df.head().to_string())
except Exception as e:
    print(f"Error: {e}")
