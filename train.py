import os
import csv
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score

from dataset import MILFeatureDataset
from models.clam import CLAM_SB, CLAM_MB

# ======================================================
# SEED
# ======================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================================================
# METRICHE
# ======================================================
def compute_metrics(labels, preds):
    """
    Calcola accuracy, f1 weighted e confusion matrix
    """
    labels = np.array(labels)
    preds = np.array(preds)

    acc = np.sum(labels == preds) / len(labels)
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)
    return acc, f1, cm

# ======================================================
# TRAINING CLAM CON CV
# ======================================================
def train_clam_cv(
    csv_path,
    features_dir,
    model_type="clam_sb",
    n_classes=3,
    epochs=20,
    batch_size=1,
    lr=1e-4,
    weight_decay=1e-5,
    save_dir="checkpoints",
    device=None,
    seed=42,
    n_splits=5,
    cv_mode="kfold",
    drop_out=0.25,
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] CV mode: {cv_mode}")

    dataset = MILFeatureDataset(csv_path=csv_path, features_dir=features_dir)
    labels_all = [label for _, label in dataset]

    base_dir = os.path.join(save_dir, features_dir, f"seed_{seed}")
    os.makedirs(base_dir, exist_ok=True)

    # ============================
    # CV SPLITTER
    # ============================
    if cv_mode == "kfold":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.arange(len(dataset)), labels_all)
        cv_dir = base_dir
        print(f"[INFO] Using {n_splits}-Fold Stratified CV")
    elif cv_mode == "loo":
        splitter = LeaveOneOut()
        split_iter = splitter.split(np.arange(len(dataset)))
        cv_dir = os.path.join(base_dir, "loo")
        os.makedirs(cv_dir, exist_ok=True)
        print("[INFO] Using Leave-One-Out CV")
    else:
        raise ValueError("cv_mode must be 'kfold' or 'loo'")

    # ============================
    # GLOBAL STORAGE LOO
    # ============================
    global_val_labels = []
    global_val_preds = []
    fold_best_f1 = []

    # ==================================================
    # FOLDS
    # ==================================================
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n========== Fold {fold + 1} ==========")
        fold_dir = (
            os.path.join(cv_dir, f"fold_{fold}")
            if cv_mode == "kfold"
            else os.path.join(cv_dir, f"sample_{val_idx[0]}")
        )
        os.makedirs(fold_dir, exist_ok=True)

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # ============================
        # CREAZIONE MODELLO DIRETTO
        # ============================
        if model_type == "clam_sb":
            model = CLAM_SB(
                gate=True,
                size_arg="small",
                dropout=drop_out,
                k_sample=8,
                n_classes=n_classes,
                instance_loss_fn=nn.CrossEntropyLoss(),
                subtyping=False,
                embed_dim=1024
            ).to(device)
        elif model_type == "clam_mb":
            model = CLAM_MB(
                gate=True,
                size_arg="small",
                dropout=drop_out,
                k_sample=8,
                n_classes=n_classes,
                instance_loss_fn=nn.CrossEntropyLoss(),
                subtyping=False,
                embed_dim=1024
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # CSV per metriche
        csv_path_fold = os.path.join(fold_dir, "metrics.csv")
        with open(csv_path_fold, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc", "train_f1",
                "val_loss", "val_acc", "val_f1",
            ])

        best_f1 = 0.0

        # ============================
        # EPOCHS
        # ============================
        for epoch in range(1, epochs + 1):
            # ---------------- TRAIN ----------------
            model.train()
            train_loss = 0.0
            train_labels, train_preds = [], []

            for bags, labels in tqdm(train_loader, desc=f"Train | Epoch {epoch}"):
                bags = bags.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits, _, Y_hat, _, results = model(bags[0], label=labels, instance_eval=True)
                inst_loss = results.get("instance_loss", 0.0)
                loss = criterion(logits, labels) + inst_loss if isinstance(inst_loss, torch.Tensor) else criterion(logits, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_labels.extend(labels.cpu().tolist())
                train_preds.extend(Y_hat.cpu().squeeze(1).tolist())

            train_loss /= len(train_loader)
            train_acc, train_f1, train_cm = compute_metrics(train_labels, train_preds)

            # ---------------- VALIDATION ----------------
            model.eval()
            val_loss = 0.0
            val_labels, val_preds = [], []

            with torch.no_grad():
                for bags, labels in tqdm(val_loader, desc=f"Val   | Epoch {epoch}"):
                    bags = bags.to(device)
                    labels = labels.to(device)

                    logits, _, Y_hat, _, _ = model(bags[0], label=labels, instance_eval=False)
                    val_loss += criterion(logits, labels).item()
                    val_labels.extend(labels.cpu().tolist())
                    val_preds.extend(Y_hat.cpu().squeeze(1).tolist())

            val_loss /= len(val_loader)
            val_acc, val_f1, val_cm = compute_metrics(val_labels, val_preds)

            print(f"[Epoch {epoch}] Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

            # SALVA MODELLO MIGLIORE
            if cv_mode == "kfold" and val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))

            # SALVA METRICHE CSV
            with open(csv_path_fold, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1])

        # ============================
        # POST-FOLD
        # ============================
        if cv_mode == "kfold":
            fold_best_f1.append(best_f1)
            print(f"[Fold {fold}] Best Val F1: {best_f1:.4f}")
        else:  # LOO
            global_val_labels.extend(val_labels)
            global_val_preds.extend(val_preds)

    # ============================
    # SUMMARY
    # ============================
    if cv_mode == "kfold":
        summary_path = os.path.join(base_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "best_val_f1"])
            for i, f1_val in enumerate(fold_best_f1):
                writer.writerow([i, f1_val])
            writer.writerow(["mean", np.mean(fold_best_f1)])
            writer.writerow(["std", np.std(fold_best_f1)])
    else:
        val_acc, val_f1, val_cm = compute_metrics(global_val_labels, global_val_preds)
        summary_path = os.path.join(cv_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["ACC", val_acc])
            writer.writerow(["F1", val_f1])
        print(f"\n[LOO SUMMARY] F1: {val_f1:.4f}, ACC: {val_acc:.4f}")
        print(f"Confusion matrix:\n{val_cm}")

    print("\n[INFO] Training completed.")

# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, help="Path to dataset CSV")
    parser.add_argument("--features-dir", required=True, help="Directory with .h5 features")
    parser.add_argument("--model-type", choices=["clam_sb", "clam_mb"], default="clam_sb")
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--cv", choices=["kfold", "loo"], default="kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    train_clam_cv(
        csv_path=args.csv_path,
        features_dir=args.features_dir,
        model_type=args.model_type,
        n_classes=args.n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        seed=args.seed,
        cv_mode=args.cv,
        n_splits=args.n_splits,
        drop_out=args.drop_out
    )
