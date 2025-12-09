import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WSI_Dataset, get_simple_loader
from models.clam import CLAM_SB, CLAM_MB
from utils import calculate_error
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def evaluate(model, loader, n_classes, device):
    model.eval()
    val_loss = 0.
    val_error = 0.
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

            loss = loss_fn(logits, label)
            instance_loss = instance_dict['instance_loss']
            
            val_loss += loss.item() # usually we monitor bag loss for eval
            val_error += calculate_error(Y_hat, label)
            
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.item())
            all_preds.append(Y_hat.item())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} slides...")

    val_loss /= len(loader)
    val_error /= len(loader)
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    acc = 1 - val_error
    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = 0.0
        print("Warning: Could not calculate AUC (single class present?)")
        
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nEvaluation Results:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained CLAM Model')
    parser.add_argument('--data_root_dir', type=str, required=True, help='data directory')
    parser.add_argument('--csv_path', type=str, required=True, help='path to csv file with list of slides to test')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to trained model checkpoint .pt file')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], default='clam_sb', help='type of model')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout rate used in training')
    parser.add_argument('--k_sample', type=int, default=8, help='k_sample used in training')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    print(f"Loading data from {args.csv_path}")
    dataset = WSI_Dataset(csv_path=args.csv_path, data_dir=args.data_root_dir, shuffle=False)
    loader = get_simple_loader(dataset, batch_size=1)
    
    # Load Model
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'k_sample': args.k_sample}
    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict, instance_loss_fn=nn.CrossEntropyLoss())
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict, instance_loss_fn=nn.CrossEntropyLoss())
    else:
        raise NotImplementedError
        
    print(f"Loading checkpoint: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    
    evaluate(model, loader, args.n_classes, device)

if __name__ == "__main__":
    main()
