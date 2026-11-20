"""Rolling-origin robustness + assembly of the final phase2_results.json (reuses phase2.py's data prep verbatim)."""
import os, sys, ast, re, json, pickle, numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

OUT = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
DATA_DIR = os.environ.get('P2P_DATA', os.path.join(ROOT, 'data'))
os.makedirs(OUT, exist_ok=True)

src = open(os.path.join(SRC_DIR, 'phase2.py')).read()
header = src[:src.index('# ================================================== 1. BASELINES')]
header = header.replace("self.mean = X[self.nums].mean();", "self.mean = X[self.nums].mean().fillna(0);")
exec(header)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
BP = json.load(open(OUT + 'phase2_grid.json')); BP = max(BP, key=lambda r: r['val_auc']); BP = dict(num_leaves=BP['num_leaves'], min_child_samples=BP['min_child_samples'])
roll = []
for year in [2012, 2013, 2014, 2015]:
    trn = df[(df.issue_d.dt.year < year)]; tst = df[df.issue_d.dt.year == year]
    vc = trn.issue_d.quantile(0.9); f_, v_ = trn[trn.issue_d < vc], trn[trn.issue_d >= vc]
    for tf in ['all', '36m']:
        f2, v2, t2 = (f_, v_, tst) if tf == 'all' else (f_[f_.term == 36], v_[v_.term == 36], tst[tst.term == 36])
        fb = downsample(f2); l4 = LinDesign(NUM, CAT).fit(f2)
        mm = LogisticRegression(C=10, max_iter=3000).fit(l4.transform(fb), fb[Y].values)
        a_lr = roc_auc_score(t2[Y], mm.predict_proba(l4.transform(t2))[:, 1])
        g = train_lgb(f2, v2, F_ENG, params=BP)
        a_g = roc_auc_score(t2[Y], g.predict(lgb_frame(t2, F_ENG)))
        roll.append(dict(test_year=year, terms=tf, n_train=len(f2), n_test=len(t2), default_rate=t2[Y].mean(), AUC_LR=a_lr, AUC_LGBM=a_g))
        log('ROLL', roll[-1])
pd.DataFrame(roll).to_csv(OUT + 'phase2_rolling.csv', index=False)
# ---- assemble
ck = pickle.load(open(OUT + 'phase2_checkpoint.pkl', 'rb'))
Rf = ck['R']; Rf['models'] = ck['MODELS']; Rf['rolling'] = roll
txt = open(OUT + 'phase2.log').read().replace('np.float64(', '(').replace('np.float32(', '(')
def grab(tag):
    for line in txt.splitlines():
        if f'] {tag} ' in line:
            s = line.split(f'] {tag} ', 1)[1]
            if tag.startswith('LGBM + engineered, IPW'): return ast.literal_eval(s)
            return ast.literal_eval(s)
Rf['models']['LGBM + engineered, IPW reject-inference'] = grab('LGBM + engineered, IPW reject-inference')
Rf['ipw_low_propensity'] = grab('IPW'); Rf['small_business'] = grab('SB'); Rf['shap_top10'] = grab('SHAP top')
json.dump(Rf, open(OUT + 'phase2_results.json', 'w'), indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o))
np.savez_compressed(OUT + 'phase2_scores.npz', y=np.load(OUT + 'phase2_ytest.npy'), **{k.replace(' ', '_'): v for k, v in ck['SC'].items()})
log('assembled', list(Rf.keys()))
