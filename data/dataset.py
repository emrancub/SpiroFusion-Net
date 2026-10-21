# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing import parse_spirogram_curve

class UKBSpirometryDataset(Dataset):
    """
    Multimodal PyTorch Dataset yielding:
      - wave_tensor: (4, 1000) float32 physiological waveform tensor
      - tab_tensor:  (D_tab,) float32 standardized clinical + curve biomarker + GBDT prior vector
      - y_copd:      scalar float32 binary COPD label
      - duration:    scalar float32 follow-up survival time (years)
      - event:       scalar float32 all-cause mortality indicator
    """
    def __init__(self, raw_curve_strings, X_tab, y_copd, y_duration, y_event):
        self.raw_curves = list(raw_curve_strings)
        self.X_tab = np.asarray(X_tab, dtype=np.float32)
        self.y_copd = np.asarray(y_copd, dtype=np.float32)
        self.y_duration = np.asarray(y_duration, dtype=np.float32)
        self.y_event = np.asarray(y_event, dtype=np.float32)

    def __len__(self):
        return len(self.y_copd)

    def __getitem__(self, idx):
        wave = parse_spirogram_curve(self.raw_curves[idx])
        return (
            torch.from_numpy(wave),
            torch.from_numpy(self.X_tab[idx]),
            torch.tensor(self.y_copd[idx], dtype=torch.float32),
            torch.tensor(self.y_duration[idx], dtype=torch.float32),
            torch.tensor(self.y_event[idx], dtype=torch.float32)
        )
