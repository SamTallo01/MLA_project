import os
import csv
import argparse
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import MILFeatureDataset
from models.clam import CLAM_SB, CLAM_MB

# SEED
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# BINARY MERGE UTILITY
def merge_labels(label, binary_merge=False):
    """
    Se binary_merge=True, unisce classe 0 e classe 2 in classe 0.
    Classe 1 rimane classe 1.
    
    Mapping: 0->0, 1->1, 2->0
    """
    if not binary_merge:
        return label
    
    if label == 2:
        return 0
    return label

# METRICS
def compute_metrics(labels, preds):
    """
    Calcola accuracy, f1 weighted e confusion matrix
    """
    labels = np.array(labels)
    preds = np.array(preds)

    acc = np.sum(labels == preds) / len(labels)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds)
    return acc, f1, cm

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

def save_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    """
    Save confusion matrix as an image
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Aggiungi etichette
    n_classes = cm.shape[0]
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, [f'Class {i}' for i in range(n_classes)])
    plt.yticks(tick_marks, [f'Class {i}' for i in range(n_classes)])
    
    # Aggiungi valori nelle celle
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Confusion matrix saved to: {save_path}")

# PCA VISUALIZATION
def extract_slide_features(model, loader, device, binary_merge=False):
    """
    Estrae le feature slide-level (dopo attention pooling) per ogni sample
    """
    model.eval()
    features_list = []
    labels_list = []
    preds_list = []
    
    with torch.inference_mode():
        for data, label in loader:
            data = data.squeeze(0).to(device)
            
            # Applica binary merge se necessario
            original_label = label.item()
            merged_label = merge_labels(original_label, binary_merge)
            label = torch.tensor([merged_label], dtype=torch.long).to(device)
            
            # Forward pass
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=False)
            
            # Estrai feature dopo attention pooling
            with torch.no_grad():
                h = data  # [N, D] features delle istanze
                
                # Passa attraverso attention network
                if hasattr(model, 'attention_net'):
                    A, h = model.attention_net(h)  # attention scores and transformed features
                    A = torch.transpose(A, 1, 0)  # [n_classes, N] or [1, N]
                    
                    if hasattr(model, 'classifiers') and isinstance(model.classifiers, nn.ModuleList):
                        # Multi-branch (CLAM-MB)
                        # Usa la prima branch per semplicità
                        A_raw = A[0]
                        A = torch.softmax(A_raw, dim=0)
                        slide_feature = torch.mm(A.unsqueeze(0), h).squeeze(0)
                    else:
                        # Single-branch (CLAM-SB)
                        A = torch.softmax(A, dim=1)
                        slide_feature = torch.mm(A, h).squeeze(0)
                else:
                    # Fallback: usa mean pooling
                    slide_feature = torch.mean(h, dim=0)
                
                features_list.append(slide_feature.cpu().numpy())
                labels_list.append(label.item())
                preds_list.append(Y_hat.item())
    
    return np.array(features_list), np.array(labels_list), np.array(preds_list)

def plot_pca_visualization(features, labels, preds, save_path, title="PCA Visualization", n_classes=3):
    """
    Visualizza le feature slide-level nello spazio PCA
    """
    # Applica PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Crea plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by true labels
    ax1 = axes[0]
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i in range(n_classes):
        mask = labels == i
        ax1.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                   c=[colors(i)], label=f'Class {i}', alpha=0.7, s=100, edgecolors='k')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title(f'{title} - True Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Colored by predictions (with markers for errors)
    ax2 = axes[1]
    for i in range(n_classes):
        mask = preds == i
        ax2.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                   c=[colors(i)], label=f'Pred {i}', alpha=0.7, s=100, edgecolors='k')
    
    # Marca gli errori con una X rossa
    errors = labels != preds
    if np.any(errors):
        ax2.scatter(features_pca[errors, 0], features_pca[errors, 1], 
                   marker='x', c='red', s=200, linewidths=3, label='Misclassified')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title(f'{title} - Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] PCA visualization saved to: {save_path}")

# TRAINING CLAM
def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, loss_fn, device, binary_merge=False):

    model.train()
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    
    train_labels = []
    train_preds = []

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data = data.squeeze(0).to(device)
        
        # Optional merge
        original_label = label.item()
        merged_label = merge_labels(original_label, binary_merge)
        label = torch.tensor([merged_label], dtype=torch.long).to(device)
        
        
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        
        # Bag-level loss
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        # Instance-level loss
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        # Combined loss con bag_weight
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss
        
        train_loss += loss_value
        
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(
                batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {} (orig: {}), bag_size: {}'.format(label.item(), original_label, data.size(0)))
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Save metrics
        train_labels.append(label.item())
        train_preds.append(Y_hat.item())
    
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
    
    train_acc, train_f1, _ = compute_metrics(train_labels, train_preds)
    
    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss: {:.4f}, train_error: {:.4f}, train_acc: {:.4f}, train_f1: {:.4f}'.format(
        epoch, train_loss, train_inst_loss, train_error, train_acc, train_f1))
    
    return train_loss, train_error, train_acc, train_f1

# VALIDATION CLAM
def validate_clam(epoch, model, loader, n_classes, loss_fn, device, binary_merge=False):

    model.eval()
    
    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count = 0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    
    val_labels = []
    val_preds = []
    
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):

            data = data.squeeze(0).to(device)
            
            # Optional merge
            original_label = label.item()
            merged_label = merge_labels(original_label, binary_merge)
            label = torch.tensor([merged_label], dtype=torch.long).to(device)
            
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            
            # Instance loss
            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error
            
            val_labels.append(label.item())
            val_preds.append(Y_hat.item())
    
    val_error /= len(loader)
    val_loss /= len(loader)
    
    if inst_count > 0:
        val_inst_loss /= inst_count
    
    # Calcola AUC
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        aucs = []
        for class_idx in range(n_classes):
            if class_idx in labels:
                auc_class = roc_auc_score(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(auc_class)
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))
    
    val_acc, val_f1, val_cm = compute_metrics(val_labels, val_preds)
    
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}'.format(
        val_loss, val_error, auc, val_acc, val_f1))
    
    return val_loss, val_error, auc, val_acc, val_f1, val_cm

# TRAINING CLAM with Cross-Validation
def train_clam_cv(
    csv_path,
    features_dir,
    model_type="clam_sb",
    n_classes=3,
    epochs=20,
    batch_size=1,
    lr=1e-4,
    weight_decay=1e-5,
    bag_weight=0.7,
    save_dir="checkpoints",
    device=None,
    seed=42,
    n_splits=5,
    cv_mode="kfold",
    drop_out=0.25,
    opt='adam',
    binary_merge=False,
    embed_dim=1024,
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] CV mode: {cv_mode}")
    print(f"[INFO] Bag weight: {bag_weight}")
    print(f"[INFO] Optimizer: {opt}")
    print(f"[INFO] Embedding dimension: {embed_dim}")
    print(f"[INFO] Binary merge: {binary_merge}")
    
    if binary_merge:
        print("[INFO] Classes 0 and 2 will be merged into class 0")
        print("[INFO] Effective number of classes: 2 (instead of 3)")
        n_classes = 2  # Override n_classes quando binary_merge è attivo

    dataset = MILFeatureDataset(csv_path=csv_path, features_dir=features_dir)
    
    if binary_merge:
        labels_all = [merge_labels(label, binary_merge) for _, label in dataset]
    else:
        labels_all = [label for _, label in dataset]

    base_dir = os.path.join(save_dir, features_dir, f"seed_{seed}")
    if binary_merge:
        base_dir = os.path.join(base_dir, "binary_merge")
    os.makedirs(base_dir, exist_ok=True)

    # CV SPLITTER
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

    # GLOBAL STORAGE
    global_val_labels = []
    global_val_preds = []
    fold_best_f1 = []
    fold_best_val_loss = []
    fold_best_auc = []
    
    # Per confusion matrix aggregata (K-Fold)
    all_folds_val_labels = []
    all_folds_val_preds = []

    # FOLDS
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        fold_dir = (
            os.path.join(cv_dir, f"fold_{fold}")
            if cv_mode == "kfold"
            else os.path.join(cv_dir, f"sample_{val_idx[0]}")
        )
        os.makedirs(fold_dir, exist_ok=True)

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                               num_workers=0, pin_memory=torch.cuda.is_available())

        # LOSS FUNCTIONS
        criterion = nn.CrossEntropyLoss()
        instance_loss_fn = nn.CrossEntropyLoss()

        # CREAZIONE MODELLO
        if model_type == "clam_sb":
            model = CLAM_SB(
                gate=True,
                size_arg="small",
                dropout=drop_out,
                k_sample=8,
                n_classes=n_classes,
                instance_loss_fn=instance_loss_fn,
                subtyping=False,
                embed_dim=embed_dim
            ).to(device)
        elif model_type == "clam_mb":
            model = CLAM_MB(
                gate=True,
                size_arg="small",
                dropout=drop_out,
                k_sample=8,
                n_classes=n_classes,
                instance_loss_fn=instance_loss_fn,
                subtyping=False,
                embed_dim=embed_dim
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # OPTIMIZER
        if opt == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        # CSV per metriche
        csv_path_fold = os.path.join(fold_dir, "metrics.csv")
        with open(csv_path_fold, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_error", "train_acc", "train_f1",
                "val_loss", "val_error", "val_auc", "val_acc", "val_f1",
            ])

        best_val_loss = float('inf')
        best_f1 = 0.0
        best_auc = 0.0
        best_epoch = 0

        # EPOCHS
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_error, train_acc, train_f1 = train_loop_clam(
                epoch, model, train_loader, optimizer, n_classes, bag_weight, criterion, device, binary_merge
            )
            
            # Validation
            val_loss, val_error, val_auc, val_acc, val_f1, val_cm = validate_clam(
                epoch, model, val_loader, n_classes, criterion, device, binary_merge
            )

            # SAVE BEST MODEL (based on MINIMUM VALIDATION LOSS)
            if cv_mode == "kfold" and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_f1 = val_f1
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
                print(f"[SAVED] New best model with Val Loss: {best_val_loss:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
                
                # Save confusion matrix of the best model
                save_confusion_matrix(
                    val_cm, 
                    os.path.join(fold_dir, "best_confusion_matrix.png"),
                    title=f"Best Model Confusion Matrix (Epoch {epoch}, Val Loss: {best_val_loss:.4f})"
                )

            # SAVE METRICS CSV
            with open(csv_path_fold, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    train_loss, train_error, train_acc, train_f1,
                    val_loss, val_error, val_auc, val_acc, val_f1
                ])

        # Confusion Matrix e PCA
        if cv_mode == "kfold":
            fold_best_f1.append(best_f1)
            fold_best_val_loss.append(best_val_loss)
            fold_best_auc.append(best_auc)
            
            # Carica il miglior modello
            model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pt")))
            model.eval()
            
            # Final metrics on validation set using best model
            final_val_labels = []
            final_val_preds = []
            
            with torch.inference_mode():
                for data, label in val_loader:
                    data = data.squeeze(0).to(device)
                    original_label = label.item()
                    merged_label = merge_labels(original_label, binary_merge)
                    label = torch.tensor([merged_label], dtype=torch.long).to(device)
                    
                    _, _, Y_hat, _, _ = model(data, label=label, instance_eval=False)
                    final_val_labels.append(label.item())
                    final_val_preds.append(Y_hat.item())
            
            # Confusion matrix finale
            _, _, final_cm = compute_metrics(final_val_labels, final_val_preds)
            
            # Salva confusion matrix come CSV
            cm_csv_path = os.path.join(fold_dir, "confusion_matrix.csv")
            np.savetxt(cm_csv_path, final_cm, delimiter=',', fmt='%d')
            print(f"[SAVED] Confusion matrix CSV: {cm_csv_path}")
            
            # Save confusion matrix
            save_confusion_matrix(
                final_cm, 
                os.path.join(fold_dir, "confusion_matrix_final.png"),
                title=f"Fold {fold+1} - Final Confusion Matrix"
            )
            
            # PCA VISUALIZATION
            print("[INFO] Extracting slide-level features for PCA visualization...")
            
            # Extract slide-level features for train and validation sets
            train_features, train_labels_pca, train_preds_pca = extract_slide_features(model, train_loader, device, binary_merge)
            val_features, val_labels_pca, val_preds_pca = extract_slide_features(model, val_loader, device, binary_merge)
            
            # Plot PCA training
            plot_pca_visualization(
                train_features, 
                train_labels_pca, 
                train_preds_pca,
                os.path.join(fold_dir, "pca_train.png"),
                title=f"Fold {fold+1} - Training Set",
                n_classes=n_classes
            )
            
            # Plot PCA validation
            plot_pca_visualization(
                val_features, 
                val_labels_pca, 
                val_preds_pca,
                os.path.join(fold_dir, "pca_val.png"),
                title=f"Fold {fold+1} - Validation Set",
                n_classes=n_classes
            )
            
            # Plot PCA combined (train + val)
            combined_features = np.vstack([train_features, val_features])
            combined_labels = np.concatenate([train_labels_pca, val_labels_pca])
            combined_preds = np.concatenate([train_preds_pca, val_preds_pca])
            
            plot_pca_visualization(
                combined_features, 
                combined_labels, 
                combined_preds,
                os.path.join(fold_dir, "pca_combined.png"),
                title=f"Fold {fold+1} - Combined (Train + Val)",
                n_classes=n_classes
            )
            
            print(f"\n[Fold {fold + 1}] Best Val Loss: {best_val_loss:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f} (Epoch {best_epoch})")
            
            # Salva le predizioni di questo fold per la confusion matrix globale
            all_folds_val_labels.extend(final_val_labels)
            all_folds_val_preds.extend(final_val_preds)
            
        else:  # LOO
            # Final predictions for LOO
            model.eval()
            with torch.inference_mode():
                for data, label in val_loader:
                    data = data.squeeze(0).to(device)
                    original_label = label.item()
                    merged_label = merge_labels(original_label, binary_merge)
                    label = torch.tensor([merged_label], dtype=torch.long).to(device)
                    
                    _, _, Y_hat, _, _ = model(data, label=label, instance_eval=False)
                    global_val_labels.append(label.item())
                    global_val_preds.append(Y_hat.item())

    # SUMMARY
    if cv_mode == "kfold":
        summary_path = os.path.join(base_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "best_val_loss", "best_val_f1", "best_val_auc"])
            for i in range(len(fold_best_f1)):
                writer.writerow([i, fold_best_val_loss[i], fold_best_f1[i], fold_best_auc[i]])
            writer.writerow(["mean", np.mean(fold_best_val_loss), np.mean(fold_best_f1), np.mean(fold_best_auc)])
            writer.writerow(["std", np.std(fold_best_val_loss), np.std(fold_best_f1), np.std(fold_best_auc)])
        

        if len(all_folds_val_labels) > 0:
            global_acc, global_f1, global_cm = compute_metrics(all_folds_val_labels, all_folds_val_preds)
            
            # Salva confusion matrix globale come CSV
            global_cm_csv_path = os.path.join(base_dir, "confusion_matrix_all_folds.csv")
            np.savetxt(global_cm_csv_path, global_cm, delimiter=',', fmt='%d')
            print(f"\n[SAVED] Global confusion matrix CSV: {global_cm_csv_path}")
            
            # Global confusion matrix
            save_confusion_matrix(
                global_cm, 
                os.path.join(base_dir, "confusion_matrix_all_folds.png"),
                title=f"Global Confusion Matrix - All {n_splits} Folds"
            )
            
            print(f"\n{'='*50}")
            print(f"GLOBAL METRICS (All Folds Combined)")
            print(f"{'='*50}")
            print(f"Total samples: {len(all_folds_val_labels)}")
            print(f"Global Accuracy: {global_acc:.4f}")
            print(f"Global F1: {global_f1:.4f}")
            print(f"Global Confusion Matrix:")
            print(global_cm)
        
        print(f"\n{'='*50}")
        print(f"SUMMARY - K-Fold CV")
        print(f"{'='*50}")
        print(f"Mean Val Loss: {np.mean(fold_best_val_loss):.4f} ± {np.std(fold_best_val_loss):.4f}")
        print(f"Mean F1: {np.mean(fold_best_f1):.4f} ± {np.std(fold_best_f1):.4f}")
        print(f"Mean AUC: {np.mean(fold_best_auc):.4f} ± {np.std(fold_best_auc):.4f}")
        print(f"Summary saved to: {summary_path}")
    else:
        val_acc, val_f1, val_cm = compute_metrics(global_val_labels, global_val_preds)
        
        # Confusion matrix for LOO
        save_confusion_matrix(
            val_cm, 
            os.path.join(cv_dir, "confusion_matrix_loo.png"),
            title="Leave-One-Out - Confusion Matrix"
        )
        
        cm_csv_path = os.path.join(cv_dir, "confusion_matrix.csv")
        np.savetxt(cm_csv_path, val_cm, delimiter=',', fmt='%d')
        
        summary_path = os.path.join(cv_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["ACC", val_acc])
            writer.writerow(["F1", val_f1])
        
        print(f"\n{'='*50}")
        print(f"SUMMARY - Leave-One-Out CV")
        print(f"{'='*50}")
        print(f"F1: {val_f1:.4f}, ACC: {val_acc:.4f}")
        print(f"Confusion matrix:\n{val_cm}")
        print(f"Summary saved to: {summary_path}")

    print("\n[INFO] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, help="Path to dataset CSV")
    parser.add_argument("--features-dir", required=True, help="Directory with .h5 features")
    parser.add_argument("--model-type", choices=["clam_sb", "clam_mb"], default="clam_sb")
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=1024,
                       help="Feature dimension from feature extractor (default: 1024 for ResNet/ImageNet)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bag-weight", type=float, default=0.7, 
                       help="Weight for bag loss vs instance loss (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--cv", choices=["kfold", "loo"], default="kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--binary-merge", action="store_true", 
                       help="Merge classes 0 and 2 into class 0 (binary classification: 0 vs 1)")
    args = parser.parse_args()

    train_clam_cv(
        csv_path=args.csv_path,
        features_dir=args.features_dir,
        model_type=args.model_type,
        n_classes=args.n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bag_weight=args.bag_weight,
        save_dir=args.save_dir,
        seed=args.seed,
        cv_mode=args.cv,
        n_splits=args.n_splits,
        drop_out=args.drop_out,
        opt=args.opt,
        binary_merge=args.binary_merge,
        embed_dim=args.embed_dim
    )