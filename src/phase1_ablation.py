"""Phase 1 ablation: how much of the GBDT gain is employment-length coding vs genuine nonlinearity?"""
import os, sys, json, numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from metrics import delong_auc_ci, recall_pair, best_macro_threshold
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb

OUT = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
DATA_DIR = os.environ.get('P2P_DATA', os.path.join(ROOT, 'data'))
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(os.path.join(DATA_DIR, 'phase1.parquet'))
df = df[df['purpose'].notna()]
c = df['purpose'].value_counts(); df = df[df['purpose'].isin(c[c > 19000].index)]
df['purpose'] = df['purpose'].replace({'Debt consolidation':'debt_consolidation','Home improvement':'home_improvement','Credit card refinancing':'credit_card','Other':'other','Vacation':'vacation','Medical expenses':'medical','Car financing':'car','Major purchase':'major_purchase','Moving and relocation':'moving','Home buying':'house'})
P = sorted(df.purpose.unique())
cut = df.issue_d.quantile(0.9); tr = df[df.issue_d < cut]; te = df[df.issue_d >= cut]; del df
vcut = tr.issue_d.quantile(0.9); fit = tr[tr.issue_d < vcut].sample(3_000_000, random_state=0); val = tr[tr.issue_d >= vcut]; del tr
out = {}
yte = te.rejected.values
def lgbX(d, use_emp, use_other=True):
    X = pd.DataFrame(index=d.index)
    if use_other:
        X['dti'] = d.dti; X['loan_amnt'] = d.loan_amnt; X['purpose'] = pd.Categorical(d.purpose, categories=P)
    if use_emp: X['emp_length'] = d.emp_length
    return X
for name, ue, uo in [('LGBM_no_emp_length', False, True), ('LGBM_emp_length_only', True, False)]:
    g = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=127, min_child_samples=200, verbose=-1, num_threads=1),
                  lgb.Dataset(lgbX(fit, ue, uo), fit.rejected.values), 600, valid_sets=[lgb.Dataset(lgbX(val, ue, uo), val.rejected.values)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    p = g.predict(lgbX(te, ue, uo)); thr = best_macro_threshold(val.rejected.values, g.predict(lgbX(val, ue, uo)))
    a, lo, hi = delong_auc_ci(yte, p); r1, r0, rm = recall_pair(yte, p, thr)
    out[name] = dict(AUC=a, lo=lo, hi=hi, recall_rejected=r1, recall_accepted=r0, recall_macro=rm); print(name, out[name], flush=True)
# LR with employment length as one-hot categories (lets a linear model use the same coding)
emp_levels = sorted(fit.emp_length.fillna(-1).unique())
def lrX(d, mu, sd):
    A = ((d[['dti','loan_amnt']] - mu) / sd).to_numpy(np.float32)
    Pm = (d.purpose.to_numpy()[:, None] == np.array(P)[None, :]).astype(np.float32)
    E = (d.emp_length.fillna(-1).to_numpy()[:, None] == np.array(emp_levels)[None, :]).astype(np.float32)
    return np.hstack([A, Pm, E])
mu = fit[['dti','loan_amnt']].mean(); sd = fit[['dti','loan_amnt']].std()
m = SGDClassifier(loss='log_loss', alpha=1e-5, class_weight='balanced', max_iter=1000, tol=1e-4, random_state=1).fit(lrX(fit, mu, sd), fit.rejected.values)
p = m.predict_proba(lrX(te, mu, sd))[:, 1]
a, lo, hi = delong_auc_ci(yte, p); r1, r0, rm = recall_pair(yte, p, 0.5)
out['LR_emp_length_onehot'] = dict(AUC=a, lo=lo, hi=hi, recall_rejected=r1, recall_accepted=r0, recall_macro=rm); print(out['LR_emp_length_onehot'])
json.dump(out, open(os.path.join(OUT, 'phase1_ablation.json'), 'w'), indent=2, default=float)
