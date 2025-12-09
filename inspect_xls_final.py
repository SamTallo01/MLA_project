import pandas as pd
import re

try:
    df = pd.read_excel('dataset/diagnosi.xls')
    
    # 1. Identify ID column (PAZIENTE)
    id_col = [c for c in df.columns if 'PAZIENTE' in c][0]
    print(f"ID Column identified as: {id_col}")
    
    # 2. Identify Label column (DIAGNOSI)
    label_col = [c for c in df.columns if 'DIAGNOSI' in c][0]
    print(f"Label Column identified as: {label_col}")
    
    # 3. Print unique labels
    print("Unique Labels found:")
    print(df[label_col].value_counts())
    
    # 4. Show sample ID mapping
    print("\nSample ID Mapping Preview:")
    for val in df[id_col].head(5):
        # cleaner
        clean_val = str(val).strip()
        # Extract number
        match = re.search(r'(\d+)$', clean_val)
        if match:
             print(f"Original: '{val}' -> Extracted Number: {match.group(1)} -> Candidate Slide ID: M-{match.group(1)}")
        else:
             print(f"Original: '{val}' -> No number found")

except Exception as e:
    print(f"Error: {e}")
