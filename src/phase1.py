"""Phase 1 (loan acceptance): faithful replication of Turiel & Aste (2020) + improved model."""
import os, sys, json, time
import numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from metrics import full_report, delong_test, best_macro_threshold
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

OUT = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
DATA_DIR = os.environ.get('P2P_DATA', os.path.join(ROOT, 'data'))
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
df = pd.read_parquet(os.path.join(DATA_DIR, 'phase1.parquet'))
# ---- authors' exact cleaning (code_1_acceptance.py) ----
df = df[df['purpose'].notna()]
counts = df['purpose'].value_counts()
df = df[df['purpose'].isin(counts[counts > 19000].index)]
to_replace = {'Debt consolidation': 'debt_consolidation', 'Home improvement': 'home_improvement',
              'Credit card refinancing': 'credit_card', 'Other': 'other', 'Vacation': 'vacation',
              'Medical expenses': 'medical', 'Car financing': 'car', 'Major purchase': 'major_purchase',
              'Moving and relocation': 'moving', 'Home buying': 'house'}
df['purpose'] = df['purpose'].replace(to_replace)
print('rows after cleaning', len(df), 'rejected share', df.rejected.mean(), flush=True)
results = {'n_rows': int(len(df)), 'rejected_share': float(df.rejected.mean())}

NUM = ['dti', 'emp_length', 'loan_amnt']
purposes = sorted(df.purpose.unique())


def design_lr(d, mean=None, scaler=None):
    X = d[NUM].astype('float64').copy()
    if mean is None: mean = X.mean()
    X = X.fillna(mean)
    if scaler is None: scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    dummies = np.stack([(d['purpose'].values == p).astype(np.float64) for p in purposes], axis=1)
    return np.hstack([Xs, dummies]), mean, scaler


for split_q in [0.9, 0.75]:
    cut = df['issue_d'].quantile(split_q)
    tr = df[df.issue_d < cut]; te = df[df.issue_d >= cut]
    key = f'split_{int(split_q*100)}'
    res = {'cut_date': str(cut.date()), 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'test_rejected_share': float(te.rejected.mean())}
    # ---- Baseline: SGD logistic regression, balanced weights, alpha=1e-3 (paper) ----
    Xtr, mu, sc = design_lr(tr); Xte, _, _ = design_lr(te, mu, sc)
    lr = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, class_weight='balanced',
                       max_iter=1000, tol=1e-3, random_state=1).fit(Xtr, tr.rejected.values)
    p_lr = lr.predict_proba(Xte)[:, 1]
    res['LR_baseline'] = full_report(te.rejected.values, p_lr, 0.5, calibrated=False)
    print(key, 'LR', res['LR_baseline'], time.time() - t0, flush=True)

    # ---- Improved: LightGBM with native categorical + missing handling ----
    # temporal validation = last 10% of training period
    vcut = tr['issue_d'].quantile(0.9)
    fit = tr[tr.issue_d < vcut]; val = tr[tr.issue_d >= vcut]
    rng = np.random.default_rng(0)
    fit = fit.iloc[rng.choice(len(fit), size=min(4_000_000, len(fit)), replace=False)]

    def lgb_X(d):
        X = d[NUM].copy()
        X['purpose'] = pd.Categorical(d['purpose'], categories=purposes)
        X['emp_missing'] = d['emp_length'].isna().astype('int8')
        X['loan_round_1000'] = (d['loan_amnt'] % 1000 == 0).astype('int8')
        return X
    params = dict(objective='binary', learning_rate=0.1, num_leaves=127, min_child_samples=200,
                  feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=2, seed=0)
    dtr = lgb.Dataset(lgb_X(fit), fit.rejected.values)
    dva = lgb.Dataset(lgb_X(val), val.rejected.values, reference=dtr)
    gbm = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
    p_val = gbm.predict(lgb_X(val)); thr = best_macro_threshold(val.rejected.values, p_val)
    p_gb = gbm.predict(lgb_X(te))
    res['LGBM'] = full_report(te.rejected.values, p_gb, thr, calibrated=True)
    res['LGBM']['best_iter'] = int(gbm.best_iteration)
    a1, a2, z, pv = delong_test(te.rejected.values, p_gb, p_lr)
    res['DeLong_LGBM_vs_LR'] = dict(z=float(z), p=float(pv))
    imp = dict(zip(gbm.feature_name(), gbm.feature_importance('gain').round(1).tolist()))
    res['LGBM_gain_importance'] = imp
    print(key, 'LGBM', res['LGBM'], res['DeLong_LGBM_vs_LR'], imp, time.time() - t0, flush=True)

    if split_q == 0.9:
        # Empirical evidence of rule-based screening: rejection rate by DTI and by loan amount (test period)
        bins = [-1e9, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 1e9]
        g = te.groupby(pd.cut(te.dti, bins), observed=True).rejected.agg(['mean', 'size'])
        g.index = g.index.astype(str); g.to_csv(OUT + 'phase1_reject_rate_by_dti.csv')
        ge = te.groupby(te.emp_length.fillna(-1)).rejected.agg(['mean', 'size'])
        ge.to_csv(OUT + 'phase1_reject_rate_by_emp.csv')
        # partial dependence of LGBM on dti (others at test medians/mode)
        grid = np.linspace(0, 60, 61)
        base = lgb_X(te.sample(20000, random_state=0))
        pdp = []
        for gv in grid:
            b = base.copy(); b['dti'] = gv; pdp.append(gbm.predict(b).mean())
        pd.DataFrame({'dti': grid, 'pdp_reject': pdp}).to_csv(OUT + 'phase1_pdp_dti.csv', index=False)
        np.save(OUT + 'phase1_scores.npy', np.vstack([te.rejected.values, p_lr, p_gb]))
    results[key] = res

json.dump(results, open(OUT + 'phase1_results.json', 'w'), indent=2, default=float)
print('done', time.time() - t0)
