"""
referral_lib.py — skin cancer referral with a malignant miss-rate target and label-free site recalibration.
Project: Label-free recalibration of skin cancer referral thresholds under site shift
Author: Arshad Saeed, Bay Atlantic University (asaeed@stu.bau.edu)
License: MIT

Policy: a case is AUTO-CLEARED (not referred) when its malignancy score s < tau, otherwise REFERRED.
Safety target: P(s < tau | malignant) <= alpha  (malignant miss rate on the deployment site).

Threshold methods (all except ORACLE use NO target labels):
  SRC      split-conformal class-conditional threshold fitted on source calibration data only
  WCP      weighted conformal (Tibshirani et al., 2019): calibration malignant scores re-weighted by
           estimated density ratio p_target(x)/p_source(x) from a cross-fitted domain classifier on embeddings.
           Per-test-case threshold; if the test point carries too much weight the case is referred
           ("certify or refuse").
  QTC      quantile threshold matching (Yilmaz & Heckel): keep the source auto-clear fraction on the target
  QTC_LS   QTC with the auto-clear fraction rescaled by the EM-estimated target benign prevalence
  ORACLE_n uses n labeled target malignant cases (few-label reference, not label-free)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
MAL_IDX = [0, 2, 3]   # MEL, BCC, AKIEC


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y, grid=np.linspace(0.5, 5, 46)):
    best, bt = np.inf, 1.0
    for t in grid:
        p = softmax(logits / t)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best:
            best, bt = nll, t
    return bt


def mal_score(probs):
    return probs[:, MAL_IDX].sum(1)


# ---------------------------------------------------------------- thresholds
def conformal_lower(scores_mal, alpha):
    """Largest tau with P(s_new < tau) <= alpha under exchangeability (finite-sample)."""
    s = np.sort(scores_mal)
    k = int(np.floor(alpha * (len(s) + 1)))
    return -np.inf if k < 1 else s[k - 1]


def weighted_lower(scores_mal, w_cal, w_test, alpha):
    """Per-test weighted conformal lower threshold (Tibshirani et al. 2019, lower tail).
    The test point's own mass sits at -inf, so tau = -inf (refer) when that mass alone exceeds alpha."""
    order = np.argsort(scores_mal)
    s, w = scores_mal[order], w_cal[order]
    W = w.sum()
    taus = np.empty(len(w_test))
    cw = np.cumsum(w)
    for i, wt in enumerate(w_test):
        tot = W + wt
        # mass strictly below candidate s_j (exclusive) = wt + cw[j-1]
        below = (wt + np.concatenate([[0.0], cw[:-1]])) / tot
        ok = np.where(below <= alpha)[0]
        taus[i] = -np.inf if (wt / tot > alpha or len(ok) == 0) else s[ok[-1]]
    return taus


def density_ratio(emb_src, emb_tgt, n_pca=64, seed=0, clip=50.0):
    """Cross-fitted domain classifier -> w(x) = p_t(x)/p_s(x) for source AND target points.
    Returns (w_src, w_tgt, domain_auc)."""
    X = np.vstack([emb_src, emb_tgt]).astype(np.float32)
    d = np.r_[np.zeros(len(emb_src)), np.ones(len(emb_tgt))]
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=min(n_pca, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    p = np.zeros(len(X))
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, d):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], d[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(d, p)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = (p / (1 - p)) * (len(emb_src) / len(emb_tgt))
    w = np.clip(w, 0, clip)
    return w[:len(emb_src)], w[len(emb_src):], auc


def em_prior(probs_t, prior_s, iters=200, tol=1e-6):
    """Saerens et al. (2002) EM estimate of the target class prior (label shift), no target labels."""
    prior_s = np.clip(np.asarray(prior_s, float), 1e-6, None); prior_s = prior_s / prior_s.sum()
    pi = prior_s.copy()
    for _ in range(iters):
        q = probs_t * (pi / prior_s)
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


# ---------------------------------------------------------------- one site
def evaluate_site(src, tgt, alphas=(0.05, 0.10), oracle_ns=(20, 50), seed=0):
    """
    src: dict with  s_cal (all cal scores), y_cal_mal (0/1), emb_cal, probs_cal, prior_s
    tgt: dict with  s, y_mal (0/1, used ONLY for evaluation and ORACLE), is_mel (0/1), emb, probs, lesion (ids)
    Returns list of result rows (one per alpha x method).
    """
    rng = np.random.default_rng(seed)
    s_mal_src = src['s_cal'][src['y_cal_mal'] == 1]
    w_src, w_tgt, dom_auc = density_ratio(src['emb_cal'], tgt['emb'], seed=seed)
    w_mal = w_src[src['y_cal_mal'] == 1]
    n_eff = w_mal.sum() ** 2 / (w_mal ** 2).sum()
    pi_t = em_prior(tgt['probs'], src['prior_s'])
    mal_prev_s = src['prior_s'][MAL_IDX].sum()
    mal_prev_t_hat = pi_t[MAL_IDX].sum()
    rows = []
    for a in alphas:
        th = {}
        tau_src = conformal_lower(s_mal_src, a)
        th['SRC'] = np.full(len(tgt['s']), tau_src)
        th['WCP'] = weighted_lower(s_mal_src, w_mal, w_tgt, a)
        clear_frac_src = np.mean(src['s_cal'] < tau_src)
        th['QTC'] = np.full(len(tgt['s']), np.quantile(tgt['s'], clear_frac_src) if clear_frac_src > 0 else -np.inf)
        cf = np.clip(clear_frac_src * (1 - mal_prev_t_hat) / (1 - mal_prev_s), 0, 1)
        th['QTC_LS'] = np.full(len(tgt['s']), np.quantile(tgt['s'], cf) if cf > 0 else -np.inf)
        mal_idx = np.where(tgt['y_mal'] == 1)[0]
        for n in oracle_ns:
            if len(mal_idx) > n:
                pick = rng.choice(mal_idx, n, replace=False)
                t = conformal_lower(tgt['s'][pick], a)
                th[f'ORACLE_{n}'] = np.full(len(tgt['s']), t)
        for m, tau in th.items():
            cleared = tgt['s'] < tau
            r = summarize(cleared, tgt)
            r.update(dict(alpha=a, method=m, domain_auc=dom_auc, n_eff_mal=n_eff,
                          est_mal_prev=mal_prev_t_hat, true_mal_prev=tgt['y_mal'].mean()))
            rows.append(r)
    return rows


def summarize(cleared, tgt):
    y, mel = tgt['y_mal'] == 1, tgt['is_mel'] == 1
    return dict(
        n=len(y), n_mal=int(y.sum()), n_mel=int(mel.sum()),
        mal_miss=float(cleared[y].mean()) if y.any() else np.nan,
        mel_miss=float(cleared[mel].mean()) if mel.any() else np.nan,
        referral_rate=float((~cleared).mean()),
        auto_clear_npv=float((~y[cleared]).mean()) if cleared.any() else np.nan,
    )


def lesion_bootstrap(cleared, tgt, B=1000, seed=0):
    """95% CI for mal_miss and referral_rate, resampling lesions."""
    rng = np.random.default_rng(seed)
    les = np.asarray(tgt['lesion'])
    uniq, inv = np.unique(les, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    mm, rr = [], []
    for _ in range(B):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        y = tgt['y_mal'][idx] == 1
        c = cleared[idx]
        if y.any():
            mm.append(c[y].mean())
        rr.append((~c).mean())
    return np.percentile(mm, [2.5, 97.5]), np.percentile(rr, [2.5, 97.5])
