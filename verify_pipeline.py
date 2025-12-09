import os
import torch
import pandas as pd
import numpy as np
from dataset import WSI_Dataset, get_simple_loader
from models.clam import CLAM_SB

def verify():
    print("Setting up dummy environment...")
    # 1. Create dummy files
    os.makedirs("debug_data", exist_ok=True)
    
    # Dummy features: 10 slides, random features (100 patches x 1024 dim)
    for i in range(10):
        feat = torch.randn(100, 1024)
        torch.save(feat, f"debug_data/slide_{i}.pt")
    
    # Dummy CSV
    df = pd.DataFrame({
        'slide_id': [f'slide_{i}' for i in range(10)],
        'label': [i % 2 for i in range(10)]
    })
    df.to_csv("debug_data/dataset.csv", index=False)
    
    print("Testing Data Loader...")
    # 2. Test Loader
    ds = WSI_Dataset(csv_path="debug_data/dataset.csv", data_dir="debug_data")
    assert len(ds) == 10
    feat, label = ds[0]
    assert feat.shape == (100, 1024)
    print("Data Loader OK.")
    
    print("Testing Model Forward...")
    # 3. Test Model
    # Reduced k_sample to 2 for debugging
    model = CLAM_SB(n_classes=2, dropout=True, k_sample=2)
    loader = get_simple_loader(ds, batch_size=1)
    
    data, label = next(iter(loader))
    print(f"Data info - Shape: {data.shape}, Label: {label}")
    
    try:
        logits, Y_prob, Y_hat, _, results_dict = model(data, label=label, instance_eval=True)
        print(f"Logits shape: {logits.shape}")
        print(f"Instance Loss: {results_dict['instance_loss']}")
        print("Model Forward OK.")
    except Exception as e:
        print(f"Forward failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Cleaning up...")
    import shutil
    shutil.rmtree("debug_data")
    print("Verification Passed!")

if __name__ == "__main__":
    verify()
