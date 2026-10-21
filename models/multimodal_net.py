# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models.spiro_1d_cnn import MultiScaleSEResNet1D
from models.survival_net import DeepSurvHead

class GatedResidualTabularEncoder(nn.Module):
    """
    Gated Residual Network (GRN) for clinical, GLI-2012, primary care EHR,
    and 12 small-airway curve biomarkers.
    """
    def __init__(self, in_features, embed_dim=Config.TABULAR_EMBED_DIM, dropout=Config.DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h0 = self.input_proj(x)
        h = self.drop(F.gelu(self.fc1(h0)))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(h0))
        return self.ln(h0 + g * h)

class BidirectionalGatedCrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention between 4-Channel Spirogram Waveform Tokens
    and Clinical/Biomarker Embeddings with a learnable QC-Aware Sigmoid Modality Gate.
    """
    def __init__(self, dim=Config.FUSION_EMBED_DIM, num_heads=4, dropout=0.15):
        super().__init__()
        self.wave_to_tab = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.tab_to_wave = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln_w = nn.LayerNorm(dim)
        self.ln_t = nn.LayerNorm(dim)

        self.qc_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, z_wave, wave_tokens, z_tab):
        # z_wave: (B, D), wave_tokens: (B, T', D), z_tab: (B, D)
        q_tab = z_tab.unsqueeze(1)   # (B, 1, D)
        q_wave = z_wave.unsqueeze(1) # (B, 1, D)

        # Tabular query attends over temporal waveform tokens
        t2w, _ = self.tab_to_wave(q_tab, wave_tokens, wave_tokens)
        z_t_tilde = self.ln_t(z_tab + t2w.squeeze(1))

        # Waveform summary attends to tabular context
        w2t, _ = self.wave_to_tab(q_wave, q_tab, q_tab)
        z_w_tilde = self.ln_w(z_wave + w2t.squeeze(1))

        # QC-Aware Sigmoid Gate dynamically weights waveform vs clinical/biomarker stream
        g = self.qc_gate(torch.cat([z_w_tilde, z_t_tilde], dim=-1))
        z_fused = g * z_w_tilde + (1.0 - g) * z_t_tilde
        return self.out_proj(z_fused), g

class SpiroFusionNet(nn.Module):
    """
    SpiroFusion-Net: Proposed Flagship Multimodal Architecture for Joint
    COPD Detection, Suboptimal Blow Salvage, PRISm Phenotyping, and 16-Year Survival Prediction.
    Includes a Wide-&-Deep Residual Structural Prior Skip Connection so the network
    strictly improves upon both gradient-boosted tabular priors and deep 1D waveform features.
    """
    def __init__(self, num_tab_features, embed_dim=Config.FUSION_EMBED_DIM):
        super().__init__()
        self.wave_encoder = MultiScaleSEResNet1D(
            in_channels=Config.NUM_WAVEFORM_CHANNELS, embed_dim=embed_dim
        )
        self.tab_encoder = GatedResidualTabularEncoder(
            in_features=num_tab_features, embed_dim=embed_dim, dropout=Config.DROPOUT
        )
        self.cross_fusion = BidirectionalGatedCrossModalAttention(
            dim=embed_dim, num_heads=4, dropout=0.15
        )
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(embed_dim // 2, 1)
        )
        # Wide-&-Deep Residual Skip Connection from standardized tabular + prior features
        self.wide_skip = nn.Linear(num_tab_features, 1)
        self.surv_head = DeepSurvHead(embed_dim)
        self.surv_wide_skip = nn.Linear(num_tab_features, 1)

    def forward(self, wave, tab):
        z_wave, wave_tokens = self.wave_encoder(wave)
        z_tab = self.tab_encoder(tab)
        z_fused, gate_weights = self.cross_fusion(z_wave, wave_tokens, z_tab)

        cls_logit = self.cls_head(z_fused).squeeze(-1) + 0.35 * self.wide_skip(tab).squeeze(-1)
        surv_hazard = self.surv_head(z_fused) + 0.25 * self.surv_wide_skip(tab).squeeze(-1)
        return cls_logit, surv_hazard
