import torch
import torch.nn as nn
from models.clam import CLAM_SB

def test():
    torch.manual_seed(0)
    model = CLAM_SB(n_classes=2, dropout=True, k_sample=8)
    h = torch.randn(100, 1024)
    label = torch.LongTensor([0])
    
    print(f"h shape: {h.shape}")
    print(f"label: {label}")
    
    try:
        logits, Y_prob, Y_hat, _, results_dict = model(h, label=label, instance_eval=True)
        print("Success!")
        print("Instance Loss:", results_dict['instance_loss'])
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
