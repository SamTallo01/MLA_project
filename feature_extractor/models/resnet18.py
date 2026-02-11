import torch.nn as nn
from torchvision import models


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Load a pretrained ResNet18 model using ImageNet1K V1 weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final classification layer (fc) by taking all children except the last one
        # This keeps the convolutional backbone + pooling. Output shape: (B, 512, 1, 1)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))

        # Flatten layer converts output from shape (B, 512, 1, 1) to (B, 512)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Pass input through all ResNet convolutional layers
        x = self.features(x)

        # Flatten from (B, 512, 1, 1) → (B, 512)
        x = self.flatten(x)

        # Return the feature embedding (512 dimensions)
        return x


def get_feature_extractor(device="cuda"):
    # Create the feature extractor, move it to the chosen device,
    # and set it to evaluation mode
    model = ResNet18FeatureExtractor().to(device).eval()
    return model
