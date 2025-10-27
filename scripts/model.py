# scripts/model.py

import torch
import torch.nn as nn
from collections import OrderedDict

# 외부 g-mlp 모듈 import
from g_mlp import gMLP 

# -------------------- [A. 모델] --------------------
class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict, d_model=512, d_ffn=1024, depth=4, dropout=0.2, use_gated_pool=True):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.mod_dims = [mod_dims[n] for n in self.mod_names]
        self.in_features = sum(self.mod_dims)
        self.seq_len = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({name: nn.Linear(in_dim, d_model) for name, in_dim in zip(self.mod_names, self.mod_dims)})
        self.backbone = gMLP(seq_len=self.seq_len, d_model=d_model, d_ffn=d_ffn, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk) for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)
        X = self.backbone(X)
        if self.use_gated_pool:
            w = torch.softmax(self.alpha, dim=0)
            Xp = (X * w.view(1, -1, 1)).sum(dim=1)
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        logits = self.head(Xp).squeeze(-1)
        return logits
        
# -------------------- [B. 유틸] --------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
