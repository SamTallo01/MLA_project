import numpy as np
import os
import glob
import csv

# --- FEATURE DA MANTENERE ---
base_features = [
    # === MORFOLOGIA NUCLEO (5) ===
    'Nucleus: Area',
    'Nucleus: Perimeter',
    'Nucleus: Circularity',
    'Nucleus: Max caliper',
    'Nucleus: Eccentricity',
    
    # === COLORAZIONE NUCLEO (10) ===
    'Nucleus: Hematoxylin OD mean',
    'Nucleus: Hematoxylin OD std dev',
    'Nucleus: Hematoxylin OD max',
    'Nucleus: Hematoxylin OD min',
    'Nucleus: Hematoxylin OD range',
    'Nucleus: Eosin OD mean',
    'Nucleus: Eosin OD std dev',
    'Nucleus: Eosin OD max',
    'Nucleus: Eosin OD min',
    'Nucleus: Eosin OD range',
    
    # === MORFOLOGIA CELLULA (6) ===
    'Cell: Area',
    'Cell: Perimeter',
    'Cell: Circularity',
    'Cell: Max caliper',
    'Cell: Eccentricity',
    'Nucleus/Cell area ratio',
    
    # === COLORAZIONE CELLULA (4) ===
    'Cell: Hematoxylin OD mean',
    'Cell: Hematoxylin OD std dev',
    'Cell: Eosin OD mean',
    'Cell: Eosin OD std dev',
    
    # === COLORAZIONE CITOPLASMA (4) ===
    'Cytoplasm: Hematoxylin OD mean',
    'Cytoplasm: Hematoxylin OD std dev',
    'Cytoplasm: Eosin OD mean',
    'Cytoplasm: Eosin OD std dev',
    
    # === HARALICK HEMATOXYLIN (6) ===
    'ROI: 2.00 µm per pixel: Hematoxylin: Haralick Contrast (F1)',
    'ROI: 2.00 µm per pixel: Hematoxylin: Haralick Correlation (F2)',
    'ROI: 2.00 µm per pixel: Hematoxylin: Haralick Entropy (F8)',
    'ROI: 2.00 µm per pixel: Hematoxylin: Haralick Sum entropy (F7)',
    'ROI: 2.00 µm per pixel: Hematoxylin: Haralick Difference variance (F9)',
    'ROI: 2.00 µm per pixel: Hematoxylin: Haralick Angular second moment (F0)',
    
    # === HARALICK EOSIN (6) ===
    'ROI: 2.00 µm per pixel: Eosin: Haralick Contrast (F1)',
    'ROI: 2.00 µm per pixel: Eosin: Haralick Correlation (F2)',
    'ROI: 2.00 µm per pixel: Eosin: Haralick Entropy (F8)',
    'ROI: 2.00 µm per pixel: Eosin: Haralick Sum entropy (F7)',
    'ROI: 2.00 µm per pixel: Eosin: Haralick Difference variance (F9)',
    'ROI: 2.00 µm per pixel: Eosin: Haralick Angular second moment (F0)',
    
    # === HARALICK BRIGHTNESS (4) ===
    'ROI: 2.00 µm per pixel: Brightness: Haralick Contrast (F1)',
    'ROI: 2.00 µm per pixel: Brightness: Haralick Correlation (F2)',
    'ROI: 2.00 µm per pixel: Brightness: Haralick Entropy (F8)',
    'ROI: 2.00 µm per pixel: Brightness: Haralick Angular second moment (F0)',
]

print(f"Feature selezionate: {len(base_features)}")
print(f"Totale colonne output: {len(base_features) * 2 + 1} (mean + std + Patch)\n")

# --- CARTELLE ---
input_folder = "manual_cell_level/SARC"
output_folder = "manual_patch_level"
os.makedirs(output_folder, exist_ok=True)

# --- Trova tutti i file TXT ---
txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
print(f"Trovati {len(txt_files)} file TXT\n")

for file in txt_files:
    filename = os.path.basename(file)
    out_name = os.path.splitext(filename)[0] + "_patch_features.csv"
    out_path = os.path.join(output_folder, out_name)
    
    print(f"Processando: {filename}")
    
    with open(file, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', newline='', encoding='utf-8') as f_out:
        
        # --- Leggi intestazione usando TAB come separatore ---
        header = f_in.readline().strip().split('\t')
        
        try:
            patch_idx = header.index("Parent")
        except ValueError:
            print(" ⚠ Colonna 'Parent' mancante, file saltato")
            continue
        
        # --- Trova gli indici delle feature da mantenere ---
        feature_indices = []
        for feat in base_features:
            try:
                idx = header.index(feat)
                feature_indices.append(idx)
            except ValueError:
                print(f" ⚠ Feature '{feat}' non trovata, saltata")
        
        if not feature_indices:
            print(" ⚠ Nessuna feature valida trovata, file saltato")
            continue
        
        # --- Scrivi intestazione CSV output ---
        mean_header = [header[i] + "_mean" for i in feature_indices]
        std_header = [header[i] + "_std" for i in feature_indices]
        out_header = ["Patch"] + mean_header + std_header
        
        writer = csv.writer(f_out)
        writer.writerow(out_header)
        
        current_patch = None
        patch_values = []
        
        # --- Processa riga per riga ---
        for line in f_in:
            row = line.strip().split('\t')
            
            if len(row) <= patch_idx:
                continue
            
            patch_name = row[patch_idx]
            
            # Estrai solo i valori delle feature selezionate
            numeric_values = []
            for i in feature_indices:
                if i >= len(row):
                    numeric_values.append(0.0)
                    continue
                
                try:
                    val = float(row[i])
                    if np.isnan(val):
                        val = 0.0
                    numeric_values.append(val)
                except (ValueError, IndexError):
                    numeric_values.append(0.0)
            
            # --- Quando cambia patch, calcola media/std e salva ---
            if current_patch is not None and patch_name != current_patch:
                if patch_values:
                    data = np.array(patch_values, dtype=np.float32)
                    mean_vals = np.mean(data, axis=0)
                    
                    if len(patch_values) > 1:
                        std_vals = np.std(data, axis=0, ddof=1)
                    else:
                        std_vals = np.zeros(data.shape[1])
                    
                    writer.writerow([current_patch] + mean_vals.tolist() + std_vals.tolist())
                patch_values = []
            
            current_patch = patch_name
            patch_values.append(numeric_values)
        
        # --- Salva ultima patch ---
        if patch_values and current_patch is not None:
            data = np.array(patch_values, dtype=np.float32)
            mean_vals = np.mean(data, axis=0)
            
            if len(patch_values) > 1:
                std_vals = np.std(data, axis=0, ddof=1)
            else:
                std_vals = np.zeros(data.shape[1])
            
            writer.writerow([current_patch] + mean_vals.tolist() + std_vals.tolist())
    
    print(f"  ✔ Salvato: {out_name}")
    
    # --- Verifica numero colonne ---
    with open(out_path, 'r', encoding='utf-8') as f_check:
        csv_reader = csv.reader(f_check)
        csv_header = next(csv_reader)
        print(f"  Numero colonne: {len(csv_header)} (1 Patch + {len(feature_indices)} mean + {len(feature_indices)} std)\n")

print("✅ Tutti i file processati!")