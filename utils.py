import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.checkpoint_path = path

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [0.] * dataset.n_classes                                      
    for c in range(dataset.n_classes):                                                   
        count = 0                                           
        for i in range(len(dataset)):
            if dataset.slide_data['label'][i] == c:
                count += 1
        weight_per_class[c] = N/float(count)                                      
    weight = [0] * len(dataset)                                              
    for i in range(len(dataset)):                                          
        weight[i] = weight_per_class[dataset.slide_data['label'][i]]                                  
    return weight

def get_cam_1d(classifier, features):
    tweights = classifier.weight
    cam_maps = torch.einsum('bg,cg->bc', features, tweights)
    return cam_maps
