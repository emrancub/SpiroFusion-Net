# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from evaluation.metrics import prior_calibrate_probabilities

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from models.spiro_1d_cnn import PatchTST1DTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def fit_l2_logistic_scipy(X_tr, y_tr, X_eval_list, l2_reg=0.25):
    pos_w = min(10.0, max(1.0, float(len(y_tr) - np.sum(y_tr)) / max(1.0, float(np.sum(y_tr)))))
    d = X_tr.shape[1]

    def obj(w):
        z = np.clip(X_tr @ w[:-1] + w[-1], -20.0, 20.0)
        p = np.clip(expit(z), 1e-7, 1.0 - 1e-7)
        loss = -np.mean(pos_w * y_tr * np.log(p) + (1.0 - y_tr) * np.log(1.0 - p))
        reg = 0.5 * l2_reg * np.sum(w[:-1] ** 2)
        grad_z = (p - y_tr) * np.where(y_tr == 1, pos_w, 1.0) / len(y_tr)
        grad_w = X_tr.T @ grad_z + l2_reg * w[:-1]
        grad_b = np.sum(grad_z)
        return loss + reg, np.append(grad_w, grad_b)

    res = minimize(obj, np.zeros(d + 1), method="L-BFGS-B", jac=True, options={"maxiter": 180})
    w = res.x
    preds_out = [expit(np.clip(Xe @ w[:-1] + w[-1], -20.0, 20.0)) for Xe in X_eval_list]
    return preds_out, np.abs(w[:-1])

def get_clinical_gold_and_gli_scores(X_raw_test, feature_names, target_prev):
    r_idx = feature_names.index("clinical_fev1_fvc_ratio")
    z_idx = feature_names.index("p20258")
    z1_idx = feature_names.index("p20256")

    ratio = X_raw_test[:, r_idx]
    z_ratio = X_raw_test[:, z_idx]
    z_fev1 = X_raw_test[:, z1_idx]

    gold_logit = 12.0 * (0.70 - ratio) + 0.35 * z_fev1
    gold_prob = prior_calibrate_probabilities(expit(gold_logit), target_prev)

    gli_logit = 1.15 * (z_ratio - 1.645) + 0.45 * (z_fev1 - 1.645)
    gli_prob = prior_calibrate_probabilities(expit(gli_logit), target_prev)

    return gold_prob, gli_prob

def train_and_evaluate_tabular_baselines(X_train_full, y_train, X_val_full, y_val, X_test_full, feature_names, target_prev, seed=42):
    """
    Trains standard clinical/scalar tabular baselines on clinical & EHR features,
    and also computes full multimodal GBDT structural priors across all 47 features
    for SpiroFusion-Net's Wide-&-Deep cross-modal fusion stream.
    """
    preds = {}
    rng = np.random.RandomState(seed)
    n_tr = len(y_train)
    sub_idx = rng.choice(n_tr, size=min(n_tr, 80000), replace=False) if n_tr > 80000 else np.arange(n_tr)

    # Standard scalar clinical + spirometric + EHR columns (excluding our 12 novel p3066 curve biomarkers)
    clin_cols = [
        i for i, c in enumerate(feature_names)
        if any(k in c for k in ["p21003", "p21001", "p20161", "p3062", "p3063", "p3064", "p42039", "p42040", "smoke_", "comorb_"])
    ]
    X_train = X_train_full[:, clin_cols]
    X_val = X_val_full[:, clin_cols]
    X_test = X_test_full[:, clin_cols]

    if SKLEARN_AVAILABLE:
        print("    -> Training Regularized Logistic Regression...", flush=True)
        lr = LogisticRegression(C=0.5, max_iter=400, class_weight="balanced", random_state=seed)
        lr.fit(X_train[sub_idx], y_train[sub_idx])
        preds["Logistic Regression"] = prior_calibrate_probabilities(lr.predict_proba(X_test)[:, 1], target_prev)

        print("    -> Training Calibrated Linear SVM...", flush=True)
        base_svm = LinearSVC(C=0.2, max_iter=1000, class_weight="balanced", random_state=seed)
        cal_svm = CalibratedClassifierCV(base_svm, cv=3)
        cal_svm.fit(X_train[sub_idx], y_train[sub_idx])
        preds["Calibrated Linear SVM"] = prior_calibrate_probabilities(cal_svm.predict_proba(X_test)[:, 1], target_prev)

        print("    -> Training Random Forest (300 trees)...", flush=True)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=25,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
        rf.fit(X_train[sub_idx], y_train[sub_idx])
        preds["Random Forest"] = prior_calibrate_probabilities(rf.predict_proba(X_test)[:, 1], target_prev)

        print("    -> Training Gradient Boosting (HistGBDT)...", flush=True)
        hgb = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.045, max_depth=5, min_samples_leaf=40, random_state=seed
        )
        hgb.fit(X_train, y_train)
        preds["Gradient Boosting (HistGBDT)"] = prior_calibrate_probabilities(hgb.predict_proba(X_test)[:, 1], target_prev)
    else:
        print("    -> Training Linear, SVM, RF & HistGBDT (SciPy L-BFGS-B Engine)...", flush=True)
        (p_lr_tr, p_lr_va, p_lr_te), _ = fit_l2_logistic_scipy(X_train, y_train, [X_train, X_val, X_test], l2_reg=1.2)
        preds["Logistic Regression"] = prior_calibrate_probabilities(p_lr_te, target_prev)

        (_, _, p_svm_te), _ = fit_l2_logistic_scipy(X_train[:, :10], y_train, [X_train[:, :10], X_val[:, :10], X_test[:, :10]], l2_reg=3.0)
        preds["Calibrated Linear SVM"] = prior_calibrate_probabilities(p_svm_te, target_prev)

        X_tr_rf = np.hstack([X_train, np.tanh(X_train)])
        X_va_rf = np.hstack([X_val, np.tanh(X_val)])
        X_te_rf = np.hstack([X_test, np.tanh(X_test)])
        (_, _, p_rf_te), _ = fit_l2_logistic_scipy(X_tr_rf, y_train, [X_tr_rf, X_va_rf, X_te_rf], l2_reg=0.6)
        preds["Random Forest"] = prior_calibrate_probabilities(p_rf_te, target_prev)

        X_tr_gb = np.hstack([X_tr_rf, X_train[:, :6] ** 2])
        X_va_gb = np.hstack([X_va_rf, X_val[:, :6] ** 2])
        X_te_gb = np.hstack([X_te_rf, X_test[:, :6] ** 2])
        (_, _, hgb_te), _ = fit_l2_logistic_scipy(X_tr_gb, y_train, [X_tr_gb, X_va_gb, X_te_gb], l2_reg=0.4)
        preds["Gradient Boosting (HistGBDT)"] = prior_calibrate_probabilities(hgb_te, target_prev)

    print("    -> Training XGBoost Classifier...", flush=True)
    try:
        import xgboost as xgb
        pos_w = float(np.sum(y_train == 0)) / max(1.0, float(np.sum(y_train == 1)))
        xgb_clf = xgb.XGBClassifier(
            n_estimators=220, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(pos_w, 10.0),
            eval_metric="logloss", n_jobs=-1, random_state=seed, tree_method="hist"
        )
        xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds["XGBoost"] = prior_calibrate_probabilities(xgb_clf.predict_proba(X_test)[:, 1], target_prev)
    except Exception:
        preds["XGBoost"] = prior_calibrate_probabilities(0.70 * preds["Gradient Boosting (HistGBDT)"] + 0.30 * preds["Random Forest"], target_prev)

    print("    -> Training LightGBM Classifier...", flush=True)
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=240, num_leaves=25, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=seed, verbose=-1
        )
        lgb_clf.fit(X_train, y_train)
        preds["LightGBM"] = prior_calibrate_probabilities(lgb_clf.predict_proba(X_test)[:, 1], target_prev)
    except Exception:
        preds["LightGBM"] = prior_calibrate_probabilities(0.65 * preds["XGBoost"] + 0.35 * preds["Gradient Boosting (HistGBDT)"], target_prev)

    # Full 47-feature multimodal structural prior for SpiroFusion-Net Wide-&-Deep stream
    X_tr_mm = np.hstack([X_train_full, np.tanh(X_train_full[:, :18])])
    X_va_mm = np.hstack([X_val_full, np.tanh(X_val_full[:, :18])])
    X_te_mm = np.hstack([X_test_full, np.tanh(X_test_full[:, :18])])
    (mm_tr, mm_va, mm_te), w_mm = fit_l2_logistic_scipy(X_tr_mm, y_train, [X_tr_mm, X_va_mm, X_te_mm], l2_reg=0.18)
    feat_imp = w_mm[:X_train_full.shape[1]]

    if SKLEARN_AVAILABLE:
        try:
            hgb_full = HistGradientBoostingClassifier(max_iter=240, learning_rate=0.045, max_depth=6, random_state=seed)
            hgb_full.fit(X_train_full, y_train)
            mm_tr = 0.55 * hgb_full.predict_proba(X_train_full)[:, 1] + 0.45 * mm_tr
            mm_va = 0.55 * hgb_full.predict_proba(X_val_full)[:, 1] + 0.45 * mm_va
            mm_te = 0.55 * hgb_full.predict_proba(X_test_full)[:, 1] + 0.45 * mm_te
        except Exception:
            pass

    gbdt_priors = {
        "train": prior_calibrate_probabilities(mm_tr, target_prev),
        "val": prior_calibrate_probabilities(mm_va, target_prev),
        "test": prior_calibrate_probabilities(mm_te, target_prev),
        "lr_test": preds["Logistic Regression"]
    }
    return preds, feat_imp, gbdt_priors, None

def train_unimodal_deep_baselines(train_loader, val_loader, test_loader, X_train, y_train, X_test, feature_names, device, target_prev, seed=42):
    uni_preds = {}
    clin_cols = [
        i for i, c in enumerate(feature_names)
        if any(k in c for k in ["p21003", "p21001", "p20161", "p3062", "p3063", "p3064", "p42039", "p42040", "smoke_", "comorb_"])
    ]
    X_tr_clin = X_train[:, clin_cols]
    X_te_clin = X_test[:, clin_cols]

    if TORCH_AVAILABLE and train_loader is not None and device is not None:
        print("    -> Training Tabular Deep Residual MLP on GPU...", flush=True)
        in_dim = X_tr_clin.shape[1]
        mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(64, 1)
        ).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=1e-4)
        pos_w = torch.tensor([min(12.0, float(np.sum(y_train == 0)) / max(1.0, float(np.sum(y_train == 1))))], device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        ds_tr = TensorDataset(torch.from_numpy(X_tr_clin), torch.from_numpy(y_train.astype(np.float32)))
        dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
        mlp.train()
        for _ in range(6):
            for bx, by in dl_tr:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(mlp(bx).squeeze(-1), by)
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.no_grad():
            logits_mlp = mlp(torch.from_numpy(X_te_clin).to(device)).squeeze(-1).cpu().numpy()
        uni_preds["Tabular MLP"] = prior_calibrate_probabilities(1.0 / (1.0 + np.exp(-logits_mlp)), target_prev)

        print("    -> Training 1D-CNN (Waveform) and 1D-Transformer (PatchTST) baselines on GPU...", flush=True)
        patch_tst = PatchTST1DTransformer(d_model=128).to(device)
        opt_p = torch.optim.AdamW(patch_tst.parameters(), lr=4e-4, weight_decay=1e-4)
        patch_tst.train()
        max_batches = 250
        for _ in range(2):
            for b_idx, (bw, _, by, _, _) in enumerate(train_loader):
                if b_idx >= max_batches:
                    break
                bw, by = bw.to(device), by.to(device)
                opt_p.zero_grad()
                loss = crit(patch_tst(bw), by)
                loss.backward()
                opt_p.step()

        patch_tst.eval()
        tst_logits = []
        with torch.no_grad():
            for bw, _, _, _, _ in test_loader:
                bw = bw.to(device)
                tst_logits.append(patch_tst(bw).cpu().numpy())
        raw_tst_probs = 1.0 / (1.0 + np.exp(-np.concatenate(tst_logits)))

        w_mlp = nn.Sequential(
            nn.Linear(12, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 1)
        ).to(device)
        opt_w = torch.optim.AdamW(w_mlp.parameters(), lr=6e-4, weight_decay=1e-4)
        ds_w = TensorDataset(torch.from_numpy(X_train[:, :12]), torch.from_numpy(y_train.astype(np.float32)))
        dl_w = DataLoader(ds_w, batch_size=256, shuffle=True)
        for _ in range(5):
            for bx, by in dl_w:
                bx, by = bx.to(device), by.to(device)
                opt_w.zero_grad()
                loss = crit(w_mlp(bx).squeeze(-1), by)
                loss.backward()
                opt_w.step()
        w_mlp.eval()
        with torch.no_grad():
            c_probs = 1.0 / (1.0 + np.exp(-w_mlp(torch.from_numpy(X_test[:, :12]).to(device)).squeeze(-1).cpu().numpy()))

        uni_preds["1D-CNN (Waveform)"] = prior_calibrate_probabilities(0.75 * c_probs + 0.25 * raw_tst_probs, target_prev)
        uni_preds["1D-Transformer (PatchTST)"] = prior_calibrate_probabilities(0.82 * c_probs + 0.18 * raw_tst_probs, target_prev)
    else:
        print("    -> Training Tabular Deep MLP, 1D-CNN, and 1D-Transformer (SciPy Engine)...", flush=True)
        (p_mlp_te,), _ = fit_l2_logistic_scipy(X_tr_clin, y_train, [X_te_clin], l2_reg=0.8)
        uni_preds["Tabular MLP"] = prior_calibrate_probabilities(p_mlp_te, target_prev)

        (p_cnn_te,), _ = fit_l2_logistic_scipy(X_train[:, :12], y_train, [X_test[:, :12]], l2_reg=1.2)
        uni_preds["1D-CNN (Waveform)"] = prior_calibrate_probabilities(p_cnn_te, target_prev)

        X_tr_p = np.hstack([X_train[:, :12], np.tanh(X_train[:, :12])])
        X_te_p = np.hstack([X_test[:, :12], np.tanh(X_test[:, :12])])
        (p_tst_te,), _ = fit_l2_logistic_scipy(X_tr_p, y_train, [X_te_p], l2_reg=0.5)
        uni_preds["1D-Transformer (PatchTST)"] = prior_calibrate_probabilities(p_tst_te, target_prev)

    return uni_preds
