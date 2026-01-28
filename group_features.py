import pandas as pd

# Carica il CSV
df = pd.read_csv("manual_features_cell/M-1.csv")

# Colonna per raggruppare
patch_col = "Parent"

# Seleziona solo le colonne numeric da aggregare
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# Rimuoviamo eventuali colonne che non hanno senso aggregare
numeric_cols = [c for c in numeric_cols if c not in ["Object ID", "Centroid X Âµm", "Centroid Y Âµm"]]

# Raggruppa per patch
agg = df.groupby(patch_col)[numeric_cols].agg(['mean', 'std', 'count'])

# Flatten MultiIndex delle colonne
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]

# Salva CSV finale
agg.reset_index().to_csv("manual_features_patch/M-1.csv", index=False)
print("Patch-level features CSV creato: manual_features_patch/M-1.csv")