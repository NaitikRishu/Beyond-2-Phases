"""Phase 2 (default prediction): replication, leakage audit, improved models, economics, XAI, robustness."""
import os, sys, json, time, warnings
import numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

warnings.filterwarnings('ignore')
from metrics import full_report, delong_test, delong_auc_ci, best_macro_threshold, recall_pair
from nn import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, xgboost as xgb

OUT = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
DATA_DIR = os.environ.get('P2P_DATA', os.path.join(ROOT, 'data'))
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

NUM = ['loan_amnt', 'term', 'installment', 'emp_length', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
       'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'log_annual_inc', 'fico_score', 'log_revol_bal']
CAT = ['home_ownership', 'verification_status', 'purpose', 'application_type']
Y = 'charged_off'

raw = pd.read_parquet(os.path.join(DATA_DIR, 'phase2.parquet'))
CATS = {c: sorted(raw[c].unique()) for c in CAT}


def implied_monthly_rate(P, A, n):
    """Solve A = P r / (1 - (1+r)^-n) for r by vectorised bisection."""
    lo = np.full(len(P), 1e-7); hi = np.full(len(P), 0.1)
    for _ in range(80):
        mid = (lo + hi) / 2
        pay = P * mid / (1 - (1 + mid) ** (-n))
        hi = np.where(pay > A, mid, hi); lo = np.where(pay > A, lo, mid)
    return (lo + hi) / 2


def engineer(d):
    d = d.copy()
    inc = 10 ** d['log_annual_inc']
    d['annual_inc'] = inc
    d['credit_hist_yrs'] = d['issue_d'].dt.year - d['earliest_cr_line']
    d['loan_to_income'] = d['loan_amnt'] / inc
    d['inst_to_income'] = 12 * d['installment'] / inc
    d['dti_incl_new'] = d['dti'] + 100 * d['inst_to_income']
    d['revol_to_income'] = 10 ** d['log_revol_bal'] / inc
    d['implied_apr'] = 12 * implied_monthly_rate(d['loan_amnt'].values, d['installment'].values, d['term'].values)
    return d


df = engineer(raw)
log('implied APR summary', df.implied_apr.describe().round(4).to_dict())

# -------------------------------------------------- authors' protocol split
d = df[df.issue_d < '2016-01-01'].copy()
cut = d.issue_d.quantile(0.9)
tr = d[d.issue_d < cut].copy(); te = d[d.issue_d >= cut].copy()
vcut = tr.issue_d.quantile(0.9)
fit = tr[tr.issue_d < vcut].copy(); val = tr[tr.issue_d >= vcut].copy()
R = {'n_total_lt2016': len(d), 'cut': str(cut.date()), 'val_cut': str(vcut.date()),
     'n_fit': len(fit), 'n_val': len(val), 'n_test': len(te),
     'default_rate': {'fit': fit[Y].mean(), 'val': val[Y].mean(), 'test': te[Y].mean()}}
log(R)

# target encoding of purpose (paper's own suggested future work, §3.3.2) — fit on 'fit' only
prior = fit[Y].mean(); m = 200
st = fit.groupby('purpose')[Y].agg(['sum', 'count'])
te_map = ((st['sum'] + m * prior) / (st['count'] + m)).to_dict()
for x in (fit, val, te, tr, d, df):
    x['purpose_default_rate'] = x['purpose'].map(te_map).fillna(prior)

ENG = ['credit_hist_yrs', 'loan_to_income', 'inst_to_income', 'dti_incl_new', 'revol_to_income', 'purpose_default_rate']
F_PAPER = NUM + CAT
F_ENG = NUM + CAT + ENG
F_ENG_STRICT = [f for f in F_ENG if f not in ('installment', 'inst_to_income', 'dti_incl_new')]


# -------------------------------------------------- design helpers
class LinDesign:
    def __init__(self, nums, cats): self.nums, self.cats = nums, cats
    def fit(self, X):
        self.mean = X[self.nums].mean(); self.sc = StandardScaler().fit(X[self.nums].fillna(self.mean)); return self
    def transform(self, X):
        A = self.sc.transform(X[self.nums].fillna(self.mean))
        D = [np.stack([(X[c].values == v).astype(float) for v in CATS[c]], 1) for c in self.cats]
        return np.hstack([A] + D) if D else A


def downsample(x, seed=0):
    pos = x[x[Y] == 1]; neg = x[x[Y] == 0].sample(n=len(pos), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


def lgb_frame(x, feats):
    X = x[feats].copy()
    for c in CAT:
        if c in X: X[c] = pd.Categorical(X[c], categories=CATS[c])
    return X


def train_lgb(fitx, valx, feats, weight=None, params=None, seed=0):
    p = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_child_samples=300,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             cat_smooth=50, verbose=-1, num_threads=2, seed=seed)
    if params: p.update(params)
    dtr = lgb.Dataset(lgb_frame(fitx, feats), fitx[Y].values, weight=weight)
    dva = lgb.Dataset(lgb_frame(valx, feats), valx[Y].values, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])


def balanced_test_metrics(y, p, thr, n_rep=20):
    """Authors' protocol: test set downsampled to balanced classes (averaged over repeats)."""
    y = np.asarray(y); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    aucs, r1s, r0s = [], [], []
    rng = np.random.default_rng(1)
    for _ in range(n_rep):
        j = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        aucs.append(roc_auc_score(y[j], p[j])); r1, r0, _ = recall_pair(y[j], p[j], thr); r1s.append(r1); r0s.append(r0)
    return dict(AUC_bal=float(np.mean(aucs)), recall_default_bal=float(np.mean(r1s)), recall_paid_bal=float(np.mean(r0s)))


SC = {}   # test-set scores per model
VS = {}   # validation scores per model
MODELS = {}
yte, yva = te[Y].values, val[Y].values


def record(name, p_val, p_test, thr=None, calibrated=True, extra=None):
    if thr is None: thr = best_macro_threshold(yva, p_val)
    rep = full_report(yte, p_test, thr, calibrated=calibrated)
    rep.update(balanced_test_metrics(yte, p_test, thr))
    if extra: rep.update(extra)
    MODELS[name] = rep; SC[name] = p_test; VS[name] = p_val
    log(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})


# ================================================== 1. BASELINES (paper re-implementation)
fit_bal = downsample(fit)
ld = LinDesign(NUM, CAT).fit(fit)
best = None
for C in [1e-3, 1e-2, 1e-1, 1, 10]:
    m_ = LogisticRegression(C=C, max_iter=2000).fit(ld.transform(fit_bal), fit_bal[Y].values)
    a = roc_auc_score(yva, m_.predict_proba(ld.transform(val))[:, 1])
    if best is None or a > best[0]: best = (a, C, m_)
lr_paper = best[2]
record('LR (paper features)', lr_paper.predict_proba(ld.transform(val))[:, 1], lr_paper.predict_proba(ld.transform(te))[:, 1],
       thr=0.5, calibrated=False, extra={'C': best[1]})

ldn = LinDesign(NUM, []).fit(fit)
dnn_res = {}
for hidden in [(5, 5), (30, 1), (20, 5)]:
    mlp = MLP(len(NUM), hidden=hidden, dropout=0.2, lr=1e-3, seed=0).fit(
        ldn.transform(fit_bal), fit_bal[Y].values.astype(float), ldn.transform(val), yva, epochs=40, batch=256, patience=4)
    dnn_res[hidden] = mlp
    log('DNN', hidden, 'val AUC', round(mlp.best_val_auc, 4), 'epochs', mlp.epochs_run)
for hidden, name in [((5, 5), 'DNN [5,5] (paper)'), ((30, 1), 'DNN [30,1] (paper)')]:
    mlp = dnn_res[hidden]
    record(name, mlp.predict_proba(ldn.transform(val)), mlp.predict_proba(ldn.transform(te)), thr=0.5, calibrated=False)

# ================================================== 2. LEAKAGE AUDIT: instalment encodes the analyst-set interest rate
leak = {}
leak['corr_implied_apr_vs_default_test_auc'] = float(roc_auc_score(yte, te.implied_apr))
leak['implied_apr_val_auc'] = float(roc_auc_score(yva, val.implied_apr))
# LR with only (loan_amnt, term, installment) vs implied APR alone
for nm, feats in [('LR_loan_term_instalment', ['loan_amnt', 'term', 'installment']), ('LR_strict_no_instalment', [f for f in NUM if f != 'installment'])]:
    l2 = LinDesign(feats, CAT if 'strict' in nm else []).fit(fit)
    mm = LogisticRegression(C=best[1], max_iter=2000).fit(l2.transform(fit_bal), fit_bal[Y].values)
    leak[nm + '_test_auc'] = float(roc_auc_score(yte, mm.predict_proba(l2.transform(te))[:, 1]))
    if 'strict' in nm:
        record('LR (strict ex-ante, no instalment)', mm.predict_proba(l2.transform(val))[:, 1], mm.predict_proba(l2.transform(te))[:, 1], thr=0.5, calibrated=False)
g_apr = train_lgb(fit, val, ['loan_amnt', 'term', 'installment'])
leak['LGBM_loan_term_instalment_test_auc'] = float(roc_auc_score(yte, g_apr.predict(lgb_frame(te, ['loan_amnt', 'term', 'installment']))))
log('LEAKAGE', leak)
R['leakage'] = leak

# ================================================== 3. IMPROVED MODELS
# small tuning grid on temporal validation
import os
if os.environ.get('GRID_JSON'):
    grid_res = json.load(open(os.environ['GRID_JSON']))
    bg = max(grid_res, key=lambda r: r['val_auc']); best_g = (bg['val_auc'], dict(num_leaves=bg['num_leaves'], min_child_samples=bg['min_child_samples']))
else:
    grid_res = []
    best_g = None
    for nl in [15, 31, 63]:
        for mcs in [100, 500]:
            g = train_lgb(fit, val, F_ENG, params=dict(num_leaves=nl, min_child_samples=mcs))
            a = roc_auc_score(yva, g.predict(lgb_frame(val, F_ENG)))
            grid_res.append(dict(num_leaves=nl, min_child_samples=mcs, val_auc=a, iters=g.best_iteration))
            if best_g is None or a > best_g[0]: best_g = (a, dict(num_leaves=nl, min_child_samples=mcs))
log('LGBM grid', grid_res)
R['lgbm_grid'] = grid_res; R['lgbm_best_params'] = best_g[1]
BP = best_g[1]

g_paper = train_lgb(fit, val, F_PAPER, params=BP)
record('LGBM (paper features)', g_paper.predict(lgb_frame(val, F_PAPER)), g_paper.predict(lgb_frame(te, F_PAPER)))
g_eng = train_lgb(fit, val, F_ENG, params=BP)
pv_eng, pt_eng = g_eng.predict(lgb_frame(val, F_ENG)), g_eng.predict(lgb_frame(te, F_ENG))
record('LGBM + engineered (proposed)', pv_eng, pt_eng, extra={'iters': g_eng.best_iteration})
g_strict = train_lgb(fit, val, F_ENG_STRICT, params=BP)
record('LGBM strict ex-ante (no instalment)', g_strict.predict(lgb_frame(val, F_ENG_STRICT)), g_strict.predict(lgb_frame(te, F_ENG_STRICT)))

# XGBoost robustness (one-hot not needed: native categorical)
def xgb_frame(x, feats): return lgb_frame(x, feats)
xm = xgb.XGBClassifier(n_estimators=4000, learning_rate=0.03, max_depth=5, min_child_weight=50, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=5, tree_method='hist', enable_categorical=True,
                       early_stopping_rounds=150, eval_metric='auc', n_jobs=2, random_state=0)
xm.fit(xgb_frame(fit, F_ENG), fit[Y].values, eval_set=[(xgb_frame(val, F_ENG), yva)], verbose=False)
record('XGBoost + engineered', xm.predict_proba(xgb_frame(val, F_ENG))[:, 1], xm.predict_proba(xgb_frame(te, F_ENG))[:, 1])

# Isotonic calibration on validation (for PD use)
iso = IsotonicRegression(out_of_bounds='clip').fit(pv_eng, yva)
pt_cal = iso.predict(pt_eng); pv_cal = iso.predict(pv_eng)
record('LGBM + engineered, isotonic-calibrated', pv_cal, pt_cal)

# Conservative combination LR + DNN + GBDT (paper's own future-work suggestion) — rank average
def rank01(a): return pd.Series(a).rank(pct=True).values
ens_t = (rank01(SC['LR (paper features)']) + rank01(SC['DNN [5,5] (paper)']) + 2 * rank01(pt_eng)) / 4
ens_v = (rank01(VS['LR (paper features)']) + rank01(VS['DNN [5,5] (paper)']) + 2 * rank01(pv_eng)) / 4
record('Rank ensemble (LR+DNN+LGBM)', ens_v, ens_t, calibrated=False)

# Calibration of the uncalibrated baselines when used as PDs (balanced training inflates PDs)
R['calibration_note'] = {k: dict(mean_pred=float(np.mean(SC[k])), actual=float(yte.mean())) for k in SC}

# DeLong tests vs paper baselines
dl = {}
for a in ['LGBM + engineered (proposed)', 'LGBM (paper features)', 'XGBoost + engineered', 'Rank ensemble (LR+DNN+LGBM)']:
    for b in ['LR (paper features)', 'DNN [5,5] (paper)', 'DNN [30,1] (paper)']:
        a1, a2, z, p = delong_test(yte, SC[a], SC[b]); dl[f'{a} vs {b}'] = dict(dAUC=a1 - a2, z=z, p=p)
R['delong'] = dl; log('DeLong', dl)

# ================================================== 4. ECONOMIC EVALUATION
inst, P, n = te.installment.values, te.loan_amnt.values, te.term.values
gain_good = inst * n - P            # interest earned on a fully paid loan (upper bound; ignores prepayment)


def strategy(p, lgd=0.65):
    o = np.argsort(p); rows = []
    yb = yte[o]; gg = gain_good[o]; pp = P[o]
    cum_bad = np.cumsum(yb); cum_n = np.arange(1, len(o) + 1)
    profit = np.cumsum(np.where(yb == 1, -lgd * pp, gg))
    for ar in np.arange(0.05, 1.0001, 0.05):
        k = int(round(ar * len(o))) - 1
        rows.append(dict(approval_rate=round(ar, 2), bad_rate=cum_bad[k] / cum_n[k], profit_musd=profit[k] / 1e6,
                         defaults_avoided_share=1 - cum_bad[k] / yb.sum()))
    # approval rate attainable at bad-rate <= 10%
    br = cum_bad / cum_n
    ok = np.where(br <= 0.10)[0]
    ar10 = (ok.max() + 1) / len(o) if len(ok) else 0.0
    best_k = int(np.argmax(profit))
    return rows, dict(approval_at_10pct_bad=ar10, max_profit_musd=profit[best_k] / 1e6,
                      approval_at_max_profit=(best_k + 1) / len(o), bad_rate_at_max_profit=br[best_k],
                      status_quo_profit_musd=profit[-1] / 1e6, status_quo_bad_rate=br[-1])

econ = {}; curves = []
for k in ['LR (paper features)', 'DNN [5,5] (paper)', 'LGBM + engineered (proposed)', 'LGBM strict ex-ante (no instalment)']:
    rows, summ = strategy(SC[k]); econ[k] = summ
    for r in rows: r['model'] = k; curves.append(r)
    summ['by_lgd'] = {lgd: strategy(SC[k], lgd)[1]['max_profit_musd'] for lgd in [0.4, 0.5, 0.65, 0.8, 0.9]}
pd.DataFrame(curves).to_csv(OUT + 'phase2_strategy_curves.csv', index=False)
R['economics'] = econ; log('ECON', econ)

# Expected-profit-optimal threshold from calibrated PD: approve if (1-PD)*gain > PD*LGD*P
ev_accept = (1 - pt_cal) * gain_good > pt_cal * 0.65 * P
R['ev_rule'] = dict(approval_rate=float(ev_accept.mean()), bad_rate=float(yte[ev_accept].mean()),
                    profit_musd=float(np.where(yte[ev_accept] == 1, -0.65 * P[ev_accept], gain_good[ev_accept]).sum() / 1e6))
log('EV rule', R['ev_rule'])

# ================================================== 5. THREE-BAND DECISIONS + CONFORMAL
bands_rows = []   # band thresholds fixed on validation quantiles, applied to test
for acc_share, ref_share in [(0.6, 0.2), (0.5, 0.3), (0.7, 0.1)]:
    t_lo = np.quantile(pv_cal, acc_share); t_hi = np.quantile(pv_cal, acc_share + ref_share)
    acc = pt_cal < t_lo; rej = pt_cal >= t_hi; ref = ~acc & ~rej
    bands_rows.append(dict(accept_share=acc.mean(), refer_share=ref.mean(), reject_share=rej.mean(),
                           bad_rate_accept=yte[acc].mean(), bad_rate_refer=yte[ref].mean(), bad_rate_reject=yte[rej].mean(),
                           defaults_in_reject=yte[rej].sum() / yte.sum(), defaults_in_accept=yte[acc].sum() / yte.sum()))
R['three_band'] = bands_rows; log('BANDS', bands_rows)

# class-conditional (Mondrian) split conformal on calibration = validation set
def conformal(alpha):
    s_val_1 = 1 - pv_cal[yva == 1]; s_val_0 = pv_cal[yva == 0]          # nonconformity scores
    q1 = np.quantile(s_val_1, np.ceil((len(s_val_1) + 1) * (1 - alpha)) / len(s_val_1))
    q0 = np.quantile(s_val_0, np.ceil((len(s_val_0) + 1) * (1 - alpha)) / len(s_val_0))
    in1 = (1 - pt_cal) <= q1; in0 = pt_cal <= q0
    cov1 = in1[yte == 1].mean(); cov0 = in0[yte == 0].mean()
    single = (in1 ^ in0); only_def = in1 & ~in0; only_paid = in0 & ~in1
    return dict(alpha=alpha, coverage_default=cov1, coverage_paid=cov0, singleton_share=single.mean(),
                only_default_share=only_def.mean(), precision_only_default=yte[only_def].mean() if only_def.any() else np.nan,
                only_paid_share=only_paid.mean(), bad_rate_only_paid=yte[only_paid].mean() if only_paid.any() else np.nan)
R['conformal'] = [conformal(a) for a in [0.1, 0.2, 0.3]]; log('CONFORMAL', R['conformal'])

import pickle as _pk
_pk.dump(dict(R=R, MODELS=MODELS, SC=SC, VS=VS), open(OUT + 'phase2_checkpoint.pkl', 'wb'))
pickle_meta = dict(BP=BP, te_map=te_map, prior=prior, cut=cut, vcut=vcut)
_pk.dump(pickle_meta, open(OUT + 'phase2_meta.pkl', 'wb'))
# ================================================== 6. SELECTION-BIAS (reject inference) via inverse propensity weighting
p1 = pd.read_parquet(os.path.join(DATA_DIR, 'phase1.parquet'), columns=['dti', 'emp_length', 'loan_amnt', 'purpose', 'issue_d', 'rejected'])
p1 = p1[(p1.issue_d < vcut) & p1.purpose.notna()]
p1 = p1.sample(3_000_000, random_state=0)
p1['purpose'] = p1['purpose'].replace({'Debt consolidation': 'debt_consolidation', 'Home improvement': 'home_improvement',
    'Credit card refinancing': 'credit_card', 'Other': 'other', 'Vacation': 'vacation', 'Medical expenses': 'medical',
    'Car financing': 'car', 'Major purchase': 'major_purchase', 'Moving and relocation': 'moving', 'Home buying': 'house',
    'Business': 'small_business', 'Wedding expenses': 'wedding', 'Green loan': 'renewable_energy'})
pc = sorted(p1.purpose.unique())
def f1(x):
    X = x[['dti', 'emp_length', 'loan_amnt']].copy(); X['purpose'] = pd.Categorical(x['purpose'], categories=pc); return X
acc_model = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=2),
                      lgb.Dataset(f1(p1), 1 - p1.rejected.values), num_boost_round=300)
del p1
p_acc_fit = np.clip(acc_model.predict(f1(fit)), 0.02, 1)
w = 1 / p_acc_fit; w = w / w.mean()
R['ipw_weight_summary'] = dict(pd.Series(w).describe().round(3))
g_ipw = train_lgb(fit, val, F_ENG, weight=w, params=BP)
record('LGBM + engineered, IPW reject-inference', g_ipw.predict(lgb_frame(val, F_ENG)), g_ipw.predict(lgb_frame(te, F_ENG)))
# Does the IPW model behave differently on borderline-acceptance applicants (low propensity)?
pa_te = acc_model.predict(f1(te)); low = pa_te < np.quantile(pa_te, 0.2)
R['ipw_low_propensity'] = dict(n=int(low.sum()), default_rate=float(yte[low].mean()),
    auc_plain=float(roc_auc_score(yte[low], pt_eng[low])), auc_ipw=float(roc_auc_score(yte[low], SC['LGBM + engineered, IPW reject-inference'][low])))
log('IPW', R['ipw_low_propensity'])

# ================================================== 7. EXPLAINABILITY (TreeSHAP)
import shap
samp = te.sample(20000, random_state=0)
ex = shap.TreeExplainer(g_eng)
sv = ex.shap_values(lgb_frame(samp, F_ENG))
if isinstance(sv, list): sv = sv[1]
imp = pd.Series(np.abs(sv).mean(0), index=F_ENG).sort_values(ascending=False)
imp.to_csv(OUT + 'phase2_shap_importance.csv')
dep = pd.DataFrame({f: samp[f].values for f in ['fico_score', 'term', 'dti', 'inst_to_income', 'log_annual_inc', 'loan_to_income']})
for f in ['fico_score', 'term', 'dti', 'inst_to_income', 'log_annual_inc', 'loan_to_income']:
    dep['shap_' + f] = sv[:, F_ENG.index(f)]
dep.to_csv(OUT + 'phase2_shap_dependence.csv', index=False)
log('SHAP top', imp.head(10).round(4).to_dict())

# calibration curve data
cal_rows = []
for k in ['LR (paper features)', 'DNN [5,5] (paper)', 'LGBM + engineered (proposed)', 'LGBM + engineered, isotonic-calibrated']:
    p = SC[k]; qs = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(p, qs[1:-1]), 0, 9)
    for b in range(10):
        mk = idx == b; cal_rows.append(dict(model=k, bin=b, mean_pred=p[mk].mean(), obs=yte[mk].mean()))
pd.DataFrame(cal_rows).to_csv(OUT + 'phase2_calibration.csv', index=False)

# ================================================== 8. SMALL-BUSINESS SUBSET (paper §3.3.3)
sb = {}
mask_te = (te.purpose == 'small_business').values
for nm, trainset in [('all', fit), ('sb_only', fit[fit.purpose == 'small_business'])]:
    vset = val if nm == 'all' else val[val.purpose == 'small_business']
    tb = downsample(trainset)
    l3 = LinDesign(NUM, CAT).fit(trainset)
    mm = LogisticRegression(C=best[1], max_iter=2000).fit(l3.transform(tb), tb[Y].values)
    p_lr_sb = mm.predict_proba(l3.transform(te[mask_te]))[:, 1]
    g = train_lgb(trainset, vset, F_ENG, params=BP if nm == 'all' else dict(num_leaves=7, min_child_samples=50))
    p_g = g.predict(lgb_frame(te[mask_te], F_ENG))
    thr_g = best_macro_threshold(vset[Y].values, g.predict(lgb_frame(vset, F_ENG)))
    for mn, p, thr in [('LR', p_lr_sb, 0.5), ('LGBM', p_g, thr_g)]:
        auc, lo, hi = delong_auc_ci(yte[mask_te], p)
        r1, r0, _ = recall_pair(yte[mask_te], p, thr)
        sb[f'{mn} trained on {nm}'] = dict(AUC=auc, lo=lo, hi=hi, recall_default=r1, recall_paid=r0)
sb['n_test_sb'] = int(mask_te.sum()); sb['default_rate_sb_test'] = float(yte[mask_te].mean())
sb['n_fit_sb'] = int((fit.purpose == 'small_business').sum())
R['small_business'] = sb; log('SB', sb)

# ================================================== 9. TEMPORAL ROBUSTNESS (rolling origin) + 36-month-only
roll = []
for year in [2012, 2013, 2014, 2015]:
    trn = df[(df.issue_d.dt.year < year)]; tst = df[df.issue_d.dt.year == year]
    vc = trn.issue_d.quantile(0.9); f_, v_ = trn[trn.issue_d < vc], trn[trn.issue_d >= vc]
    for term_filter in ['all', '36m']:
        if term_filter == '36m':
            f2, v2, t2 = f_[f_.term == 36], v_[v_.term == 36], tst[tst.term == 36]
        else:
            f2, v2, t2 = f_, v_, tst
        fb = downsample(f2); l4 = LinDesign(NUM, CAT).fit(f2)
        mm = LogisticRegression(C=best[1], max_iter=2000).fit(l4.transform(fb), fb[Y].values)
        a_lr = roc_auc_score(t2[Y], mm.predict_proba(l4.transform(t2))[:, 1])
        g = train_lgb(f2, v2, F_ENG, params=BP)
        a_g = roc_auc_score(t2[Y], g.predict(lgb_frame(t2, F_ENG)))
        roll.append(dict(test_year=year, terms=term_filter, n_train=len(f2), n_test=len(t2), default_rate=t2[Y].mean(), AUC_LR=a_lr, AUC_LGBM=a_g))
        log('ROLL', roll[-1])
pd.DataFrame(roll).to_csv(OUT + 'phase2_rolling.csv', index=False)
R['rolling'] = roll

# ================================================== save
R['models'] = MODELS
json.dump(R, open(OUT + 'phase2_results.json', 'w'), indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o))
np.savez_compressed(OUT + 'phase2_scores.npz', y=yte, **{k.replace(' ', '_'): v for k, v in SC.items()})
# save artefacts used by India extension
import pickle
pickle.dump(dict(BP=BP, te_map=te_map, prior=prior, cut=cut, vcut=vcut), open(OUT + 'phase2_meta.pkl', 'wb'))
log('ALL DONE')
