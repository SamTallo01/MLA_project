import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py

class WSI_Dataset(Dataset):
    def __init__(self, csv_path, data_dir, shuffle=False, seed=7, transform=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            data_dir (string): Directory with features (contains 'pt_files' or 'h5_files').
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.slide_data = pd.read_csv(csv_path)
        
        if shuffle:
            np.random.seed(seed)
            self.slide_data = self.slide_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            
        # Check files existence
        # self._check_files()

    def _check_files(self):
        missing = []
        for idx in range(len(self.slide_data)):
            slide_id = self.slide_data.iloc[idx]['slide_id']
            full_path = os.path.join(self.data_dir, f"{slide_id}.pt")
            if not os.path.exists(full_path):
                missing.append(slide_id)
        if len(missing) > 0:
            print(f"Warning: {len(missing)} feature files missing.")

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id = str(self.slide_data.iloc[idx]['slide_id'])
        label = int(self.slide_data.iloc[idx]['label'])
        
        # Depending on how features are saved. 
        # Based on extract_fv.py, features might be directly in output_dir as .pt or .h5
        # The user seems to have features directly. Let's assume .pt for easy loading
        
        full_path = os.path.join(self.data_dir, f"{slide_id}.pt")
        
        if not os.path.exists(full_path):
             # Try h5 if pt doesn't exist
            full_path_h5 = os.path.join(self.data_dir, f"{slide_id}.h5")
            if os.path.exists(full_path_h5):
                with h5py.File(full_path_h5, 'r') as hdf5_file:
                    features = hdf5_file['features'][:]
                features = torch.from_numpy(features)
            else:
                raise FileNotFoundError(f"Feature file not found for {slide_id} at {full_path}")
        else:
            features = torch.load(full_path)

        if self.transform:
            features = self.transform(features)

        return features, label
    
    def get_split(self, split_key, all_splits):
        """
        Returns a new dataset based on the split_key ('train', 'val', 'test')
        from a splits dataframe or dictionary.
        """
        split_ids = all_splits[split_key].dropna().tolist()
        mask = self.slide_data['slide_id'].isin(split_ids)
        df_slice = self.slide_data[mask].reset_index(drop=True)
        
        # Create a new instance
        new_dset = WSI_Dataset.__new__(WSI_Dataset)
        new_dset.data_dir = self.data_dir
        new_dset.transform = self.transform
        new_dset.slide_data = df_slice
        return new_dset

def get_simple_loader(dataset, batch_size=1, num_workers=0):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_features, **kwargs)
    return loader

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]
