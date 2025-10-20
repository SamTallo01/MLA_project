import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from settings import Settings as S
from model.feature_extractor import FeatureExtractor
from model.attention import Attention

class CLAM(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=3, gated=True, k_sample=8, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.k_sample = k_sample

        # Feature extractor
        self.feature_extractor = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.embed_dim = self.feature_extractor.embed_dim

        # Attention module
        self.attention = Attention(input_dim=self.embed_dim, hidden_dim=256, gated=gated)

        # Bag-level classifier
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Instance-level classifiers (one per class)
        self.instance_classifiers = nn.ModuleList([nn.Linear(self.embed_dim, 2) for _ in range(num_classes)])
        self.instance_loss_fn = nn.CrossEntropyLoss()

    # Helper functions for instance-level supervision
    def create_positive_targets(self, length, device):
        return torch.ones(length, dtype=torch.long, device=device)

    def create_negative_targets(self, length, device):
        return torch.zeros(length, dtype=torch.long, device=device)

    def instance_eval(self, A, H, classifier):
        """
        Evaluate top-k positive and negative patches for a single class.
        """
        device = H.device
        if len(A.shape) == 1:
            A = A.view(-1)
        top_p_ids = torch.topk(A, self.k_sample)[1]
        top_n_ids = torch.topk(-A, self.k_sample)[1]

        top_p = H[top_p_ids]
        top_n = H[top_n_ids]

        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([p_targets, n_targets], dim=0)

        logits = classifier(all_instances)
        preds = torch.argmax(logits, dim=1)
        loss = self.instance_loss_fn(logits, all_targets)

        return loss, preds, all_targets

    # Forward pass
    def forward(self, x, label=None, instance_eval=False):
        """
        x: [num_patches, 3, H, W] tensor of patches
        label: int (bag-level label)
        instance_eval: whether to compute instance-level loss
        """
        # Extract features for all patches 
        H = self.feature_extractor(x)

        # --- Attention pooling ---
        M, A = self.attention(H)

        # Bag-level classification 
        logits = self.classifier(M)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        results = {'logits': logits, 'probs': probs, 'pred': pred, 'attention': A}

        # Instance-level supervision
        if instance_eval and label is not None:
            total_instance_loss = 0
            all_preds = []
            all_targets = []
            for c in range(self.num_classes):
                if label == c:
                    loss, preds, targets = self.instance_eval(A.view(-1), H, self.instance_classifiers[c])
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    total_instance_loss += loss
            results['instance_loss'] = total_instance_loss

        return results