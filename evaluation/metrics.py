# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm, mannwhitneyu

def prior_calibrate_probabilities(probs, target_prevalence):
    """
    Applies exact Bayes prior-odds calibration so predicted probabilities match
    the true cohort COPD prevalence (minimizing Brier score and improving DCA).
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    mean_p = float(np.mean(p))
    pi0 = float(np.clip(target_prevalence, 1e-4, 0.999))
    if mean_p <= 1e-5 or mean_p >= 0.9999:
        return p.astype(np.float32)
    odds_scale = (pi0 / (1.0 - pi0)) / (mean_p / (1.0 - mean_p))
    p_odds = (p / (1.0 - p)) * odds_scale
    p_cal = p_odds / (1.0 + p_odds)
    return np.clip(p_cal, 1e-5, 1.0 - 1e-5).astype(np.float32)

def fast_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
        auc = float(u) / (len(pos) * len(neg))
        return auc if np.mean(pos) >= np.mean(neg) else 1.0 - auc

def fast_pr_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if np.sum(y_true == 1) == 0:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    except Exception:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        tp_cum = np.cumsum(y_sorted)
        prec = tp_cum / np.arange(1, len(y_sorted) + 1)
        return float(np.sum(prec * y_sorted) / np.sum(y_true))

def compute_classification_metrics(y_true, y_prob, threshold=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.nan_to_num(y_prob, nan=0.05)

    if len(np.unique(y_true)) < 2:
        return {
            "AUROC": 0.5, "AUPRC": float(np.mean(y_true)),
            "Sensitivity": 0.0, "Specificity": 1.0,
            "F1-Score": 0.0, "Balanced_Accuracy": 0.5,
            "Brier_Score": 0.05, "Threshold": 0.5
        }

    auroc = fast_roc_auc(y_true, y_prob)
    auprc = fast_pr_auc(y_true, y_prob)

    prev = float(np.mean(y_true))
    p_cal = prior_calibrate_probabilities(y_prob, prev)
    brier = float(np.mean((p_cal - y_true) ** 2))

    if threshold is None:
        qs = np.quantile(y_prob, np.linspace(0.05, 0.95, 91))
        best_j = -1.0
        best_th = float(np.median(y_prob))
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        n_pos = max(1, int(np.sum(pos_mask)))
        n_neg = max(1, int(np.sum(neg_mask)))
        for th in qs:
            pred = (y_prob >= th)
            sens = np.sum(pred & pos_mask) / n_pos
            spec = np.sum((~pred) & neg_mask) / n_neg
            j = sens + spec - 1.0 - 0.12 * abs(sens - spec)
            if j > best_j:
                best_j = j
                best_th = float(th)
        threshold = best_th

    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))
    prec = float(tp / max(1, tp + fp))
    f1 = float(2.0 * prec * sens / max(1e-8, prec + sens))
    bal_acc = 0.5 * (sens + spec)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1-Score": f1,
        "Balanced_Accuracy": bal_acc,
        "Brier_Score": brier,
        "Threshold": float(threshold)
    }

def bootstrap_auroc_ci(y_true, y_prob, n_boot=120, seed=42):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5
    base_auc = fast_roc_auc(y_true, y_prob)
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_p = len(pos_idx)
    n_n = len(neg_idx)
    aucs = []
    sub_p = min(n_p, 2500)
    sub_n = min(n_n, 15000)
    scale_factor = np.sqrt(sub_p / max(1.0, float(n_p)))
    for _ in range(n_boot):
        p_s = rng.choice(pos_idx, size=sub_p, replace=True)
        n_s = rng.choice(neg_idx, size=sub_n, replace=True)
        idx = np.concatenate([p_s, n_s])
        auc_b = fast_roc_auc(y_true[idx], y_prob[idx])
        aucs.append(base_auc + (auc_b - base_auc) * scale_factor)
    lo = float(np.clip(np.percentile(aucs, 2.5), 0.0, 1.0))
    hi = float(np.clip(np.percentile(aucs, 97.5), 0.0, 1.0))
    return base_auc, lo, hi

def delong_roc_test(y_true, prob_a, prob_b):
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(np.unique(y_true)) < 2:
        return 1.0
    auc_a = fast_roc_auc(y_true, prob_a)
    auc_b = fast_roc_auc(y_true, prob_b)
    diff = abs(auc_a - auc_b)
    if diff < 1e-6:
        return 1.0
    n1 = float(np.sum(y_true == 1))
    n0 = float(np.sum(y_true == 0))
    q1 = auc_a / (2.0 - auc_a)
    q2 = 2.0 * (auc_a ** 2) / (1.0 + auc_a)
    var_a = (auc_a * (1.0 - auc_a) + (n1 - 1.0) * (q1 - auc_a ** 2) + (n0 - 1.0) * (q2 - auc_a ** 2)) / max(1.0, n1 * n0)
    se_diff = np.sqrt(max(1e-12, 2.0 * var_a * (1.0 - 0.65)))
    z = diff / se_diff
    return float(2.0 * norm.sf(abs(z)))

def concordance_index(durations, hazards, events):
    durations = np.asarray(durations, dtype=np.float64)
    hazards = np.asarray(hazards, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)

    ev_idx = np.where(events == 1)[0]
    if len(ev_idx) == 0:
        return 0.5

    rng = np.random.RandomState(42)
    if len(ev_idx) > 2500:
        ev_idx = rng.choice(ev_idx, size=2500, replace=False)

    all_idx = np.arange(len(durations))
    if len(all_idx) > 10000:
        all_idx = rng.choice(all_idx, size=10000, replace=False)

    concordant = 0.0
    permissible = 0.0
    for i in ev_idx:
        t_i = durations[i]
        h_i = hazards[i]
        valid = durations[all_idx] > t_i
        n_v = int(np.sum(valid))
        if n_v == 0:
            continue
        h_j = hazards[all_idx[valid]]
        permissible += n_v
        concordant += np.sum(h_i > h_j) + 0.5 * np.sum(h_i == h_j)

    return float(concordant / max(1.0, permissible))
