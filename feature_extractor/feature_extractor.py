import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from settings import Settings as S

class FeatureExtractor(nn.Module):
    def __init__(self, backbone=S.backbone, embed_dim=S.embed_dim, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            modules = list(resnet.children())[:-1]  # remove the last fc layer
            self.feature_extractor = nn.Sequential(*modules)
            self.out_dim = resnet.fc.in_features
        elif backbone == 'kimia':
            # here custom kimia net model
            self.feature_extractor = nn.Identity()
            self.out_dim = embed_dim
        else:
            raise ValueError(f"Backbone {backbone} non supportato")
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # flatten
        return features
