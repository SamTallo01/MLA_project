import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from settings import Settings as S
from feature_extractor.feature_extractor import FeatureExtractor
from attention.attention import Attention

class CLAM(nn.Module):
    def __init__(self, backbone='resnet50', embed_dim=1024, hidden_dim=256, n_classes=2, gated=True):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone, embed_dim)
        self.attention = Attention(self.feature_extractor.out_dim, hidden_dim, gated=gated)
        self.classifier = nn.Linear(self.feature_extractor.out_dim, n_classes)
    
    def forward(self, x):
        # x: batch of patches (N_instances, C, H, W)
        h = self.feature_extractor(x)  # N x D
        M, A = self.attention(h)       # 1 x D, N x 1
        logits = self.classifier(M)    # 1 x n_classes
        probs = torch.softmax(logits, dim=1)
        return logits, probs, A
