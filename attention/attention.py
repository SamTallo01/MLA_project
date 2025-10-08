import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from settings import Settings as S

class Attention(nn.Module):
    def __init__(self, input_dim=S.embed_dim, hidden_dim=256, gated=S.gated):
        super().__init__()
        self.gated = gated
        self.fc_a = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        if gated:
            self.fc_b = nn.Linear(input_dim, hidden_dim)
            self.sigmoid = nn.Sigmoid()
        self.fc_c = nn.Linear(hidden_dim, 1)
    
    def forward(self, h):
        a = self.tanh(self.fc_a(h))
        if self.gated:
            b = self.sigmoid(self.fc_b(h))
            a = a * b
        A = self.fc_c(a)      # N x 1
        A = torch.softmax(A, dim=0)  # normalize over instances
        M = torch.sum(A * h, dim=0, keepdim=True)  # 1 x D
        return M, A
