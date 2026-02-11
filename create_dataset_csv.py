import pandas as pd
import re

def create_csv():
    try:
        print("Reading Excel...")
        df = pd.read_excel('dataset/diagnosi.xls')
        

        id_col = [c for c in df.columns if 'PAZIENTE' in c][0]
        print(f"ID Column: {id_col}")
        

        label_col = [c for c in df.columns if 'DIAGNOSI' in c][0]
        print(f"Label Column: {label_col}")
        
        # Create Mapping
        unique_labels = df[label_col].dropna().unique()
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        
        print("\nGenerated Mapping:")
        with open('label_mapping.txt', 'w') as f:
            for k, v in label_to_int.items():
                line = f"'{k}' -> {v}"
                print(line)
                f.write(line + "\n")
                
        # Process Dataframe
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
                
                data.append({'slide_id': slide_id, 'label': label})
                
        # Save CSV
        out_df = pd.DataFrame(data)
        out_df.to_csv('dataset.csv', index=False)
        print(f"\nSaved 'dataset.csv' with {len(out_df)} rows.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_csv()
