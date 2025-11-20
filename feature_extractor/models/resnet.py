import torch.nn as nn
from torchvision import models


OUTPUT_DIM = 1024 


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Load a pretrained ResNet50 model using ImageNet1K V2 weights
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final classification layer (fc) by taking all children except the last one
        # This keeps the convolutional backbone + pooling
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))

        # Flatten layer converts output from shape (B, 2048, 1, 1) to (B, 2048)
        self.flatten = nn.Flatten()

        # Linear projection layer:
        # Maps ResNet's 2048-dimensional output to a 1024-dimensional feature vector
        self.project = nn.Linear(2048, OUTPUT_DIM)

    def forward(self, x):
        # Pass input through all ResNet convolutional layers
        x = self.features(x)

        # Flatten from (B, 2048, 1, 1) → (B, 2048)
        x = self.flatten(x)

        # Apply projection layer to reduce dimension to 1024
        x = self.project(x)

        # Return the final feature embedding
        return x


def get_feature_extractor(device="cuda"):
    # Create the feature extractor, move it to the chosen device,
    # and set it to evaluation mode
    model = ResNetFeatureExtractor().to(device).eval()
    return model
