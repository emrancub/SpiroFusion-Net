# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from evaluation.metrics import compute_classification_metrics, concordance_index, prior_calibrate_probabilities

class SpiroFusionTrainer:
    """
    Multi-Task Trainer for SpiroFusion-Net:
      - Joint Focal-BCE Classification Loss + Normalized Cox Partial Log-Likelihood Survival Loss
      - Casts logits and hazards to float32 (.float()) before loss computation so
        torch.logcumsumexp_backward works 100% reliably under FP16 AMP autocast!
      - Cosine Annealing Learning Rate Schedule
      - Modern PyTorch AMP (`torch.amp.GradScaler('cuda')`) with zero deprecation warnings
    """
    def __init__(self, model, device, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                 pos_weight=6.0, log_interval=100):
        self.model = model.to(device)
        self.device = device
        self.log_interval = log_interval
        self.use_amp = (device.type == "cuda")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.pos_weight = torch.tensor([min(8.0, max(1.5, float(pos_weight)))], dtype=torch.float32, device=device)

        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            self._amp_new_api = True
        except Exception:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=self.use_amp)
            self._amp_new_api = False

    def _autocast_ctx(self):
        if self._amp_new_api:
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        from torch.cuda.amp import autocast
        return autocast(enabled=self.use_amp)

    def focal_bce_loss(self, logits, targets, gamma=1.5):
        # Cast to float32 for numerical stability under AMP
        logits_f32 = logits.float()
        targets_f32 = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits_f32, targets_f32, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits_f32)
        pt = torch.where(targets_f32 == 1.0, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** gamma
        return torch.mean(focal_weight * bce)

    def cox_survival_loss(self, hazards, durations, events):
        # Cast to float32 because CUDA logcumsumexp_backward is not implemented for Half (FP16)
        hazards_f32 = hazards.float()
        durations_f32 = durations.float()
        events_f32 = events.float()
        if torch.sum(events_f32) < 1.0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        order = torch.argsort(durations_f32, descending=True)
        hz = torch.clamp(hazards_f32[order], min=-15.0, max=15.0)
        ev = events_f32[order]
        log_cum_hazard = torch.logcumsumexp(hz, dim=0)
        uncensored_likelihood = (hz - log_cum_hazard) * ev
        n_ev = torch.clamp(torch.sum(ev), min=1.0)
        return -torch.sum(uncensored_likelihood) / (n_ev * np.log(max(2.0, float(len(hz)))))

    def evaluate(self, loader, target_prev=0.0494):
        self.model.eval()
        all_logits, all_hazards, all_y, all_dur, all_ev = [], [], [], [], []
        with torch.no_grad():
            for wave, tab, y_copd, dur, ev in loader:
                wave = wave.to(self.device, non_blocking=True)
                tab = tab.to(self.device, non_blocking=True)
                with self._autocast_ctx():
                    logits, hazards = self.model(wave, tab)
                all_logits.append(logits.float().detach().cpu().numpy())
                all_hazards.append(hazards.float().detach().cpu().numpy())
                all_y.append(y_copd.numpy())
                all_dur.append(dur.numpy())
                all_ev.append(ev.numpy())

        logits_np = np.concatenate(all_logits)
        hazards_np = np.concatenate(all_hazards)
        y_np = np.concatenate(all_y)
        dur_np = np.concatenate(all_dur)
        ev_np = np.concatenate(all_ev)

        probs_raw = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -20.0, 20.0)))
        probs_cal = prior_calibrate_probabilities(probs_raw, target_prev)
        metrics = compute_classification_metrics(y_np, probs_cal)
        c_idx = concordance_index(dur_np, hazards_np, ev_np)
        metrics["Survival_C_Index"] = c_idx
        return metrics, probs_cal, hazards_np

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS, resume=False, target_prev=0.0494):
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pt")
        latest_path = os.path.join(Config.CHECKPOINT_DIR, "latest_checkpoint.pt")
        hist_csv = os.path.join(Config.TABLES_DIR, "Table9_Epoch_Training_and_Convergence_Log.csv")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, epochs), eta_min=1e-6)
        start_epoch = 1
        best_score = -1.0
        history_rows = []

        if resume and os.path.exists(latest_path):
            try:
                ckpt = torch.load(latest_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                best_score = float(ckpt.get("best_score", -1.0))
                if "history" in ckpt:
                    history_rows = list(ckpt["history"])
                print(f"[✓] Resumed SpiroFusion-Net from Epoch {start_epoch - 1} (Best Val AUROC: {best_score:.4f})", flush=True)
            except Exception as e:
                print(f"[!] Checkpoint architecture updated ({e}); starting fresh training for SpiroFusion-Net.", flush=True)

        total_batches = len(train_loader)
        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            ep_loss, ep_cls, ep_surv = 0.0, 0.0, 0.0
            t0 = time.time()

            for b_idx, (wave, tab, y_copd, dur, ev) in enumerate(train_loader, start=1):
                wave = wave.to(self.device, non_blocking=True)
                tab = tab.to(self.device, non_blocking=True)
                y_copd = y_copd.to(self.device, non_blocking=True)
                dur = dur.to(self.device, non_blocking=True)
                ev = ev.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_ctx():
                    logits, hazards = self.model(wave, tab)
                # Compute losses in FP32 outside autocast so logcumsumexp_backward never receives Half
                l_cls = self.focal_bce_loss(logits, y_copd)
                l_surv = self.cox_survival_loss(hazards, dur, ev)
                loss = l_cls + 0.15 * l_surv

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                ep_loss += float(loss.item())
                ep_cls += float(l_cls.item())
                ep_surv += float(l_surv.item())

                if b_idx % self.log_interval == 0 or b_idx == total_batches:
                    vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if self.use_amp else 0.0
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{ts}] Epoch [{epoch}/{epochs}] [{b_idx}/{total_batches}] "
                          f"Loss: {ep_loss/b_idx:.4f} (Cls: {ep_cls/b_idx:.4f}, Surv: {ep_surv/b_idx:.4f}) | "
                          f"VRAM: {vram_mb:.1f} MB", flush=True)

            scheduler.step()
            curr_lr = float(scheduler.get_last_lr()[0])
            val_metrics, _, _ = self.evaluate(val_loader, target_prev=target_prev)
            val_auc = val_metrics["AUROC"]
            val_auprc = val_metrics["AUPRC"]
            val_cidx = val_metrics["Survival_C_Index"]
            dt_ep = time.time() - t0

            print(f"---> Epoch [{epoch}/{epochs}] Summary ({dt_ep:.1f}s) | "
                  f"Val AUROC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f} | "
                  f"Val C-Index: {val_cidx:.4f} | LR: {curr_lr:.2e}", flush=True)

            history_rows.append({
                "Epoch": epoch,
                "Train_Total_Loss": round(ep_loss / max(1, total_batches), 4),
                "Train_Cls_Loss": round(ep_cls / max(1, total_batches), 4),
                "Train_Surv_Loss": round(ep_surv / max(1, total_batches), 4),
                "Val_AUROC": round(val_auc, 4),
                "Val_AUPRC": round(val_auprc, 4),
                "Val_Brier_Score": round(val_metrics["Brier_Score"], 4),
                "Val_Survival_C_Index": round(val_cidx, 4),
                "Learning_Rate": curr_lr
            })
            os.makedirs(Config.TABLES_DIR, exist_ok=True)
            pd.DataFrame(history_rows).to_csv(hist_csv, index=False)

            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_score": max(best_score, val_auc),
                "val_metrics": val_metrics,
                "history": history_rows
            }
            torch.save(ckpt_payload, latest_path)

            if val_auc >= best_score:
                best_score = val_auc
                torch.save(ckpt_payload, best_path)
                print(f"    [★] Saved new best SpiroFusion-Net model (Val AUROC = {best_score:.4f})", flush=True)

        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
        if len(history_rows) == 0 and os.path.exists(hist_csv):
            return pd.read_csv(hist_csv)
        return pd.DataFrame(history_rows)
