# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, T)
        se = x.mean(dim=-1)
        se = F.gelu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).unsqueeze(-1)
        return x * se

class SEResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=2, dropout=0.15):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcitation1D(out_ch, reduction=8)
        self.drop = nn.Dropout(dropout)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.gelu(out + identity)
        return out

class MultiScaleSEResNet1D(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem (k = 5, 11, 21) + 3-Stage SE-ResNet-1D
    + Multi-Head Temporal Self-Attention Pooling.
    """
    def __init__(self, in_channels=Config.NUM_WAVEFORM_CHANNELS, embed_dim=Config.CNN_EMBED_DIM):
        super().__init__()
        self.branch5 = nn.Conv1d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False)
        self.branch11 = nn.Conv1d(in_channels, 24, kernel_size=11, stride=2, padding=5, bias=False)
        self.branch21 = nn.Conv1d(in_channels, 16, kernel_size=21, stride=2, padding=10, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.stage1 = SEResBlock1D(64, 96, kernel_size=5, stride=2)
        self.stage2 = SEResBlock1D(96, 144, kernel_size=5, stride=2)
        self.stage3 = SEResBlock1D(144, embed_dim, kernel_size=3, stride=2)

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.15, batch_first=True)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, 4, 1000)
        s5 = self.branch5(x)
        s11 = self.branch11(x)
        s21 = self.branch21(x)
        h = torch.cat([s5, s11, s21], dim=1)
        h = self.stem_pool(F.gelu(self.stem_bn(h)))

        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)  # (B, D, T')

        tokens = h.transpose(1, 2)  # (B, T', D)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm_attn(tokens + attn_out)

        weights = torch.softmax(self.attn_pool(tokens), dim=1)  # (B, T', 1)
        pooled = torch.sum(tokens * weights, dim=1)  # (B, D)
        return pooled, tokens

class PatchTST1DTransformer(nn.Module):
    """
    Pure 1D Patch-Transformer (PatchTST-Spirogram) baseline operating on 4-channel waveforms.
    Segments L=1000 into 40 patches of length 25 (250 ms per patch).
    """
    def __init__(self, in_channels=Config.NUM_WAVEFORM_CHANNELS, patch_len=25, d_model=128):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = Config.TARGET_WAVEFORM_LENGTH // patch_len
        self.patch_proj = nn.Linear(in_channels * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.2, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, 4, 1000)
        b, c, l = x.shape
        x_p = x.view(b, c, self.num_patches, self.patch_len).permute(0, 2, 1, 3).reshape(b, self.num_patches, c * self.patch_len)
        tokens = self.patch_proj(x_p) + self.pos_embed
        tokens = self.norm(self.transformer(tokens))
        pooled = tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)
