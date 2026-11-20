"""Phase 1 secondary run: memory-lean LR for the 75/25 split (paper text) + split statistics; merges logged 90/10 results."""
import os, sys, json, ast, re, numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from metrics import full_report
from sklearn.linear_model import SGDClassifier

OUT = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
DATA_DIR = os.environ.get('P2P_DATA', os.path.join(ROOT, 'data'))
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(os.path.join(DATA_DIR, 'phase1.parquet'))
df = df[df['purpose'].notna()]
c = df['purpose'].value_counts(); df = df[df['purpose'].isin(c[c > 19000].index)]
df['purpose'] = df['purpose'].replace({'Debt consolidation':'debt_consolidation','Home improvement':'home_improvement','Credit card refinancing':'credit_card','Other':'other','Vacation':'vacation','Medical expenses':'medical','Car financing':'car','Major purchase':'major_purchase','Moving and relocation':'moving','Home buying':'house'})
purposes = sorted(df.purpose.unique())
res = {'n_rows': int(len(df)), 'rejected_share': float(df.rejected.mean()), 'n_purposes': len(purposes)}
for q in [0.9, 0.75]:
    cut = df.issue_d.quantile(q); tr = df[df.issue_d < cut]; te = df[df.issue_d >= cut]
    r = dict(cut=str(cut.date()), n_train=int(len(tr)), n_test=int(len(te)), test_accepted=int((te.rejected==0).sum()),
             train_accepted=int((tr.rejected==0).sum()), test_rejected_share=float(te.rejected.mean()))
    if q == 0.75:
        mu = tr[['dti','emp_length','loan_amnt']].mean(); sd = tr[['dti','emp_length','loan_amnt']].fillna(mu).std()
        def X(d):
            A = ((d[['dti','emp_length','loan_amnt']].fillna(mu) - mu) / sd).to_numpy(np.float32)
            P = (d['purpose'].to_numpy()[:, None] == np.array(purposes)[None, :]).astype(np.float32)
            return np.hstack([A, P])
        m = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, class_weight='balanced', max_iter=1000, tol=1e-3, random_state=1).fit(X(tr), tr.rejected.values)
        p = m.predict_proba(X(te))[:, 1]
        r['LR_baseline'] = full_report(te.rejected.values, p, 0.5, calibrated=False)
    res[f'split_{int(q*100)}'] = r
    print(q, r, flush=True)
# merge logged split_90 model results
txt = open(OUT + 'phase1.log').read().replace('np.float64(', '(')
for line in txt.splitlines():
    if line.startswith('split_90 LR '):
        res['split_90']['LR_baseline'] = ast.literal_eval(line[len('split_90 LR '):line.rindex('}')+1])
    if line.startswith('split_90 LGBM '):
        body = line[len('split_90 LGBM '):]
        parts = re.findall(r'\{[^{}]*\}', body)
        res['split_90']['LGBM'] = ast.literal_eval(parts[0]); res['split_90']['DeLong_LGBM_vs_LR'] = ast.literal_eval(parts[1]); res['split_90']['LGBM_gain_importance'] = ast.literal_eval(parts[2])
json.dump(res, open(OUT + 'phase1_results.json', 'w'), indent=2, default=float)
print(json.dumps(res, indent=1, default=float)[:3000])
