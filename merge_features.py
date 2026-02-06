import numpy as np
import h5py
import pandas as pd
import os
import glob
import argparse

def load_csv_features(csv_path):
    """
    Carica le feature da un file CSV.
    
    Args:
        csv_path: percorso del file CSV
        
    Returns:
        dict: dizionario con {patch_name: feature_vector}
    """
    df = pd.read_csv(csv_path)
    patch_names = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    
    feature_dict = {}
    for patch_name, feat_vec in zip(patch_names, features):
        feature_dict[str(patch_name)] = feat_vec
    return feature_dict

def load_resnet_features(h5_path):
    """
    Carica le feature ResNet da un file .h5.
    
    Args:
        h5_path: percorso del file .h5
        
    Returns:
        tuple: (coords array, features array)
    """
    with h5py.File(h5_path, 'r') as f:
        if 'coords' not in f or 'features' not in f:
            print(f"File {h5_path} non ha la struttura attesa (coords/features)")
            return None, None
        coords = f['coords'][:]
        features = f['features'][:]
        return coords, features

def align_csv_to_h5(csv_features, n_patches):
    """
    Allinea le feature CSV all'ordine delle patch nel file H5 usando l'indice estratto dal nome della patch.
    
    Args:
        csv_features: dict {patch_name: feature_vector}
        n_patches: numero totale di patch nel file H5
        
    Returns:
        np.ndarray: array delle feature CSV allineate a resnet_features
    """
    feature_dim = next(iter(csv_features.values())).shape[0]
    aligned = np.zeros((n_patches, feature_dim), dtype=np.float32)
    
    for patch_name, feat_vec in csv_features.items():
        try:
            idx = int(patch_name.split('_')[1])
        except ValueError:
            print(f"Patch name non interpretabile: {patch_name}")
            continue
        if idx >= n_patches:
            print(f"Indice patch {idx} fuori range (H5 ha {n_patches} patch)")
            continue
        aligned[idx] = feat_vec
    return aligned

def merge_and_save_features(csv_features, coords, resnet_features, output_path, case_name):
    """
    Unisce le feature CSV alle feature ResNet e salva in formato .h5.
    
    Args:
        csv_features: dict {patch_name: feature_vector}
        coords: array coordinate H5
        resnet_features: array feature ResNet
        output_path: percorso file .h5 di output
        case_name: nome del caso
    """
    n_patches = len(coords)
    print(f"Patch nel file ResNet: {n_patches}")
    
    csv_features_array = align_csv_to_h5(csv_features, n_patches)
    
    combined_features = np.concatenate([resnet_features, csv_features_array], axis=1)
    
    print(f"Features: ResNet={resnet_features.shape[1]}, CSV={csv_features_array.shape[1]}, Totale={combined_features.shape[1]}")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('coords', data=coords, dtype=coords.dtype)
        f.create_dataset('features', data=combined_features, dtype=np.float32)
        
        # Metadata
        f.attrs['n_patches'] = n_patches
        f.attrs['csv_feature_dim'] = csv_features_array.shape[1]
        f.attrs['resnet_feature_dim'] = resnet_features.shape[1]
        f.attrs['total_feature_dim'] = combined_features.shape[1]
        f.attrs['case_name'] = case_name
    
    print(f"Salvato: {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(
        description='Unisce feature CSV patch-level con feature ResNet (.h5) e salva in .h5'
    )
    parser.add_argument('--csv_folder', type=str, required=True,
                        help='Cartella contenente file CSV patch-level')
    parser.add_argument('--resnet_folder', type=str, required=True,
                        help='Cartella contenente file .h5 ResNet')
    parser.add_argument('--out_folder', type=str, required=True,
                        help='Cartella di output per i file .h5 combinati')
      
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(args.csv_folder, "*_patch_features.csv"))
    print(f"Trovati {len(csv_files)} file CSV")
    if not csv_files:
        print("Nessun file CSV trovato")
        return
    
    for csv_path in csv_files:
        csv_filename = os.path.basename(csv_path)
        case_name = csv_filename.replace("_patch_features.csv", "")
        print(f"\nProcessando: {case_name}")
        
        # Cerca file H5 corrispondente
        possible_h5_names = [f"{case_name}.h5", f"{case_name}_features.h5", f"{case_name}_resnet.h5"]
        resnet_h5_path = None
        for h5_name in possible_h5_names:
            potential_path = os.path.join(args.resnet_folder, h5_name)
            if os.path.exists(potential_path):
                resnet_h5_path = potential_path
                break
        if resnet_h5_path is None:
            print(f"File ResNet non trovato per {case_name}")
            continue
        
        csv_features = load_csv_features(csv_path)
        coords, resnet_features = load_resnet_features(resnet_h5_path)
        if coords is None or resnet_features is None:
            print(f"Errore nel caricamento file ResNet")
            continue
        
        output_path = os.path.join(args.out_folder, f"{case_name}.h5")
        merge_and_save_features(csv_features, coords, resnet_features, output_path, case_name)
    
    print("\nTutti i file processati")
    print(f"File salvati in: {args.out_folder}")

if __name__ == "__main__":
    main()
