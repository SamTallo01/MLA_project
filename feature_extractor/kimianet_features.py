from __future__ import print_function, division

import os
import time
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import openslide


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to NDPI file and saved KimiaNet weights
ndpi_path = 'M-6.ndpi'
kimianet_weights_path = 'KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth'
save_features_path = './extracted_features/'

os.makedirs(save_features_path, exist_ok=True)

# Patch extraction parameters
patch_size = 1000  # size of patches to extract (1000x1000)
stride = 1000      # patch stride (non-overlapping)

# Transformation for KimiaNet input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize patches to model input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                        [0.229, 0.224, 0.225])  # ImageNet std
])

# Custom Dataset for NDPI patches
class NDPI_Patch_Dataset(Dataset):
    def __init__(self, slide_path, patch_size, stride, transform=None):
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # Get slide dimensions using temporary OpenSlide object
        with openslide.OpenSlide(self.slide_path) as slide:
            width, height = slide.dimensions

        self.coordinates = []
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                self.coordinates.append((x, y))

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        with openslide.OpenSlide(self.slide_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert('RGB')
        if self.transform:
            patch = self.transform(patch)
        return patch, f'patch_x{x}_y{y}'


# Define KimiaNet model (DenseNet121 backbone with custom head)
class FullyConnected(nn.Module):
    def __init__(self, base_model, num_ftrs, num_classes=30):
        super(FullyConnected, self).__init__()
        self.model = base_model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        features = x
        out = self.fc_4(x)
        return features, out

def main():
    # Load pretrained DenseNet121 and adapt
    base_model = torchvision.models.densenet121(pretrained=True)
    for param in base_model.parameters():
        param.requires_grad = False

    base_model.features = nn.Sequential(base_model.features, nn.AdaptiveAvgPool2d((1, 1)))
    num_ftrs = base_model.classifier.in_features
    model = FullyConnected(base_model.features, num_ftrs)
    model = model.to(device)
    model = nn.DataParallel(model)

    # Load KimiaNet weights
    model.load_state_dict(torch.load(kimianet_weights_path))

    # Set model to eval mode
    model.eval()

    # Create dataset and dataloader
    dataset = NDPI_Patch_Dataset(ndpi_path, patch_size, stride, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Extract features and save as pickle dictionary
    features_dict = {}

    patch_save_folder = "./patch_images"
    os.makedirs(patch_save_folder, exist_ok=True)

    with torch.no_grad():
        for inputs, patch_names in dataloader:
            inputs = inputs.to(device)
            features, outputs = model(inputs)
            features = features.cpu().numpy()
            for i, patch_name in enumerate(patch_names):
                features_dict[patch_name] = features[i]
                # Save the patch PIL image - need to inverse transform
                img_tensor = inputs[i].cpu()
                # Undo normalization and convert to PIL Image
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                img_tensor = inv_normalize(img_tensor).clamp(0,1)
                pil_img = transforms.ToPILImage()(img_tensor)
                pil_img.save(os.path.join(patch_save_folder, f"{patch_name}.png"))


    # Save features dictionary
    with open(os.path.join(save_features_path, 'KimiaNet_NDPI_Features.pickle'), 'wb') as f:
        pickle.dump(features_dict, f)

    print(f'Extracted features for {len(features_dict)} patches saved in {save_features_path}')
    
if __name__ == '__main__':
    main()
