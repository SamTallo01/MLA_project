# kimianet.py
import torch
import torch.nn as nn
import torchvision.models as models

# KimiaNet is based on DenseNet-121, which typically outputs 1024 features
OUTPUT_DIM = 1024 


def get_feature_extractor(device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the full DenseNet-121 model (we need this to correctly load the weights)
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # Load the KimiaNet state dict (which may contain keys for the classifier)
    state_dict = torch.load('feature_extractor/models/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth')
    
    # Load the weights into the full model (strict=False ignores missing classifier keys)
    # Note: If the checkpoint is from a DataParallel wrapped model, use: 
    # {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False) 

    # --- Now, create the feature extractor by truncating the loaded model ---
    # 1. Truncate the classifier:
    feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
    
    # 2. Add the Global Average Pool (GAP) layer which is implicitly needed for a feature vector:
    # This is critical because the default DenseNet-121 model has the GAP *before* the classifier, 
    # but the sequence truncation above removes it. 
    # To get the final 1024 vector, we need the GAP.
    feature_extractor = nn.Sequential(feature_extractor, nn.AdaptiveAvgPool2d((1, 1)))
    
    # Squeeze the output to be a 1D vector (Batch, 1024)
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)
            
    feature_extractor.add_module('flatten', Flatten())

    feature_extractor.to(device).eval()
    return feature_extractor

def get_output_dim():
    return OUTPUT_DIM