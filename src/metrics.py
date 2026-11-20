"""Evaluation utilities: AUC with DeLong variance, KS, Brier, ECE, bootstrap, recall metrics."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, recall_score


# ---------- DeLong (fast, Sun & Xu 2014 algorithm) ----------
def _midrank(x):
    J = np.argsort(x, kind='mergesort')
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_auc_ci(y, p, alpha=0.95):
    y = np.asarray(y); order = np.argsort(-y, kind='mergesort'); m = int(y.sum())
    aucs, cov = _fast_delong(np.asarray(p)[order][None, :], m)
    se = np.sqrt(cov[0, 0]); z = stats.norm.ppf(1 - (1 - alpha) / 2)
    return aucs[0], aucs[0] - z * se, aucs[0] + z * se


def delong_test(y, p1, p2):
    """Two-sided DeLong test for correlated ROC AUCs. Returns (auc1, auc2, z, pvalue)."""
    y = np.asarray(y); order = np.argsort(-y, kind='mergesort'); m = int(y.sum())
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1, -1]])
    var = (l @ cov @ l.T).item()
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


# ---------- other metrics ----------
def ks_stat(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return stats.ks_2samp(p[y == 1], p[y == 0]).statistic


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-mass bins."""
    y = np.asarray(y); p = np.asarray(p)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    idx = np.digitize(p, qs[1:-1])
    e = 0.0
    for b in range(n_bins):
        mk = idx == b
        if mk.sum() == 0: continue
        e += mk.mean() * abs(p[mk].mean() - y[mk].mean())
    return e


def recall_pair(y, p, thr):
    yhat = (np.asarray(p) >= thr).astype(int)
    r1 = recall_score(y, yhat, pos_label=1); r0 = recall_score(y, yhat, pos_label=0)
    return r1, r0, (r1 + r0) / 2


def best_macro_threshold(y, p):
    """Threshold maximising recall macro (balanced accuracy = Youden) on a validation set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return thr[j]


def full_report(y, p, thr, calibrated=True):
    auc, lo, hi = delong_auc_ci(y, p)
    r1, r0, rm = recall_pair(y, p, thr)
    out = dict(AUC=auc, AUC_lo=lo, AUC_hi=hi, Gini=2 * auc - 1, KS=ks_stat(y, p),
               PR_AUC=average_precision_score(y, p),
               recall_default=r1, recall_paid=r0, recall_macro=rm, threshold=thr)
    if calibrated:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out.update(Brier=brier_score_loss(y, pc), LogLoss=log_loss(y, pc), ECE=ece(y, pc))
    return out
