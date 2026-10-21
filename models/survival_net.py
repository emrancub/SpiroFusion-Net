# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSurvHead(nn.Module):
    """
    Nonlinear Cox Proportional Hazards (DeepSurv) log-risk prediction head.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim // 2)
        self.ln1 = nn.LayerNorm(in_dim // 2)
        self.drop = nn.Dropout(0.20)
        self.fc_out = nn.Linear(in_dim // 2, 1)

    def forward(self, z):
        h = self.drop(F.gelu(self.ln1(self.fc1(z))))
        return self.fc_out(h).squeeze(-1)
