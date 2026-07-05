import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import wilcoxon

def calc_rmse(x_true, x_est):
    return np.sqrt(np.mean((x_true - x_est)**2))

def calc_nrmse(x_true, x_est):
    rmse = calc_rmse(x_true, x_est)
    rng = np.max(x_true) - np.min(x_true)
    if rng == 0:
        return 0.0
    return rmse / rng

def calc_max_err(x_true, x_est):
    return np.max(np.linalg.norm(x_true - x_est, axis=1))

def calc_detection_metrics(y_true, y_pred, y_score=None):
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = roc_auc_score(y_true, y_score) if y_score is not None else 0.0
    
    # Delay: compute first index where y_true=1 and y_pred=1
    attack_starts = np.where(np.diff(np.concatenate(([0], y_true))) == 1)[0]
    delays = []
    for start in attack_starts:
        # Find first detection after start
        det = np.where(y_pred[start:] == 1)[0]
        if len(det) > 0:
            delays.append(det[0]) # sample delay
    avg_delay = np.mean(delays) if delays else 0.0
    
    return precision, recall, f1, auc, avg_delay

def run_wilcoxon(dist_a, dist_b):
    if np.allclose(dist_a, dist_b):
        return 0.0, 1.0 # P-value 1.0 if identical
    try:
        res = wilcoxon(dist_a, dist_b)
        return res.statistic, res.pvalue
    except ValueError:
        return 0.0, 1.0
