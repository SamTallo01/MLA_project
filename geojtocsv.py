import json
import pandas as pd
import os

# ==========================
# Configura qui i percorsi
# ==========================
geojson_file = r"C:/Users/samut/Desktop/QuPathMLA/geojson/slide_patches.geojson"
csv_output = r"C:/Users/samut/Desktop/QuPathMLA/csv/slide_patches.csv"

# Assicurati che la cartella di output esista
os.makedirs(os.path.dirname(csv_output), exist_ok=True)

# ==========================
# Carica GeoJSON
# ==========================
with open(geojson_file) as f:
    data = json.load(f)

patches = []

# ==========================
# Estrai rettangoli
# ==========================
for feature in data["features"]:
    geom = feature["geometry"]
    coords = geom["coordinates"][0]  # primo anello del poligono
    
    # calcola il bounding box della patch
    x_min = min([c[0] for c in coords])
    y_min = min([c[1] for c in coords])
    x_max = max([c[0] for c in coords])
    y_max = max([c[1] for c in coords])
    
    width = x_max - x_min
    height = y_max - y_min
    
    patches.append([x_min, y_min, width, height])

# ==========================
# Salva CSV
# ==========================
df = pd.DataFrame(patches, columns=["X", "Y", "Width", "Height"])
df.to_csv(csv_output, index=False)

print(f"✅ CSV creato correttamente: {csv_output}")
print(f"Numero di patch: {len(patches)}")
