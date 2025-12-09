import pandas as pd
import re
import os

def create_csv():
    try:
        print("Reading Excel...")
        df = pd.read_excel('dataset/diagnosi.xls')
        
        # 1. Identify ID column
        # Look for "PAZIENTE"
        id_col = [c for c in df.columns if 'PAZIENTE' in c][0]
        print(f"ID Column: {id_col}")
        
        # 2. Identify Label column
        # Look for "DIAGNOSI"
        label_col = [c for c in df.columns if 'DIAGNOSI' in c][0]
        print(f"Label Column: {label_col}")
        
        # 3. Create Mapping
        unique_labels = df[label_col].dropna().unique()
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        
        print("\nGenerated Mapping:")
        with open('label_mapping.txt', 'w') as f:
            for k, v in label_to_int.items():
                line = f"'{k}' -> {v}"
                print(line)
                f.write(line + "\n")
                
        # 4. Process Dataframe
        data = []
        for idx, row in df.iterrows():
            paz_val = str(row[id_col])
            # Extract number from "MES TOR102" -> "102"
            match = re.search(r'(\d+)$', paz_val.strip())
            if match:
                num = match.group(1)
                slide_id = f"M-{num}"
                
                diagnosis = row[label_col]
                if pd.isna(diagnosis):
                    continue
                    
                label = label_to_int[diagnosis]
                
                # Check if feature file exists (optional, but good practice)
                # But features might be elsewhere. Let's just create the CSV.
                data.append({'slide_id': slide_id, 'label': label})
                
        # 5. Save CSV
        out_df = pd.DataFrame(data)
        out_df.to_csv('dataset.csv', index=False)
        print(f"\nSaved 'dataset.csv' with {len(out_df)} rows.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_csv()
