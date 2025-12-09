from __future__ import print_function

import argparse
import pdb
import os
import math
import sys

# Internal imports
from utils import EarlyStopping, calculate_error
from dataset import WSI_Dataset, get_simple_loader, collate_features
from models.clam import CLAM_SB, CLAM_MB

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import numpy as np

def train(datasets, cur, args):
    """   
    train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    
    # Define splits for this fold
    train_split = datasets['train']
    val_split = datasets['val']
    
    # Loaders
    train_loader = get_simple_loader(train_split, batch_size=1)
    val_loader = get_simple_loader(val_split, batch_size=1)

    # Model
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict, instance_loss_fn=nn.CrossEntropyLoss())
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict, instance_loss_fn=nn.CrossEntropyLoss())
    else:
        raise NotImplementedError
    
    model.relocate() # Not implemented in my copy, need to handle device
    model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    loss_fn = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose = True, path=os.path.join(args.results_dir, f"checkpoint_fold_{cur}.pt"))

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.
        train_error = 0.
        train_inst_loss = 0.
        
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(args.device), label.to(args.device)
            
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            
            loss = loss_fn(logits, label)
            instance_loss = instance_dict['instance_loss']
            
            total_loss = args.bag_weight * loss + (1-args.bag_weight) * instance_loss 
            
            train_loss += total_loss.item()
            train_error += calculate_error(Y_hat, label)
            train_inst_loss += instance_loss.item()
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss /= len(train_loader)
        train_error /= len(train_loader)
        
        if (epoch + 1) % 5 == 0:
             print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))

        # Validation
        val_loss = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, loss_fn, args)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

def validate(cur, epoch, model, loader, n_classes, early_stopping, loss_fn, args):
    model.eval()
    val_loss = 0.
    val_error = 0.
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(args.device), label.to(args.device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

            loss = loss_fn(logits, label)
            instance_loss = instance_dict['instance_loss']
            total_loss = args.bag_weight * loss + (1-args.bag_weight) * instance_loss 
            
            val_loss += total_loss.item()
            val_error += calculate_error(Y_hat, label)

    val_loss /= len(loader)
    val_error /= len(loader)

    if (epoch + 1) % 5 == 0:
        print('Val loss: {:.4f}, Val error: {:.4f}'.format(val_loss, val_error))

    early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(args.results_dir, f"checkpoint_fold_{cur}.pt"))
    return val_loss

def main():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--csv_path', type=str, default=None, help='path to csv file')
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], default='clam_sb', help='type of model (default: clam_sb)')
    parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout rate')
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"Loading data from {args.csv_path}")
    dataset = WSI_Dataset(csv_path=args.csv_path, data_dir=args.data_root_dir, shuffle=True, seed=args.seed)
    
    # Simple random split for now, 80-20 train-val
    # In a real scenario, use cross validation based on args.k
    print("Splitting dataset...")
    size = len(dataset)
    train_size = int(0.8 * size)
    val_size = size - train_size
    train_dset, val_dset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    datasets = {'train': train_dset, 'val': val_dset}
    
    train(datasets, 0, args)

if __name__ == "__main__":
    main()
