"""India / CIBIL extension: a documented semi-synthetic 'CIBIL-world' built on real Lending Club outcomes.

What is REAL: borrowers, loan terms, incomes, bureau attributes and default outcomes (Lending Club, pre-2016, same split as phase2).
What is TRANSFORMED (deterministic, documented): FICO -> CIBIL scale, DTI -> FOIR at a reference rate.
What is SIMULATED (assumption-driven, swept in sensitivity analysis):
  * which 15% of borrowers are New-to-Credit (NTC) -> their bureau fields are masked (CIBIL 'NH/NA')
  * Account Aggregator / UPI / NACH cash-flow features, generated from a latent stress factor whose
    dependence on bureau risk (beta) and on the realised outcome (gamma) is explicit.
"""
import os, sys, json, time, pickle, warnings
import numpy as np, pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

warnings.filterwarnings('ignore')
from metrics import delong_auc_ci, delong_test
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUT = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
DATA_DIR = os.environ.get('P2P_DATA', os.path.join(ROOT, 'data'))
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)
Y = 'charged_off'
meta = pickle.load(open(OUT + 'phase2_meta.pkl', 'rb'))
BP, cut, vcut = meta['BP'], meta['cut'], meta['vcut']
rng = np.random.default_rng(2026)

raw = pd.read_parquet(os.path.join(DATA_DIR, 'phase2.parquet'))
d = raw[raw.issue_d < '2016-01-01'].copy().reset_index(drop=True)
inc = 10 ** d.log_annual_inc
# ---------------------------------------------------------------- deterministic transforms
d['cibil'] = 300 + (d.fico_score - 300) * 600 / 550                      # linear FICO(300-850) -> CIBIL(300-900)
r_ref = 0.14 / 12                                                          # EMI at a flat 14% p.a. reference rate (no risk-based price leakage)
d['emi_ref'] = d.loan_amnt * r_ref / (1 - (1 + r_ref) ** (-d.term))
d['existing_oblig_m'] = d.dti / 100 * inc / 12
d['foir'] = (d.existing_oblig_m + d.emi_ref) / (inc / 12)                  # fixed-obligation-to-income ratio
d['credit_hist_yrs'] = d.issue_d.dt.year - d.earliest_cr_line
d['loan_to_income'] = d.loan_amnt / inc
d['income_verified'] = (d.verification_status != 'Not Verified').astype(int)   # analogue: ITR / AA-verified income
d['split'] = np.where(d.issue_d < vcut, 'fit', np.where(d.issue_d < cut, 'val', 'test'))
y = d[Y].values

tiers = pd.cut(d.cibil, [299, 680, 730, 770, 790, 900], labels=['Subprime 300-680', 'Near prime 681-730', 'Prime 731-770', 'Prime plus 771-790', 'Super prime 791-900'])
tier_tab = d.groupby(tiers, observed=True)[Y].agg(['mean', 'size']).rename(columns={'mean': 'default_rate', 'size': 'n'})
tier_tab['share'] = tier_tab.n / tier_tab.n.sum()
tier_tab.to_csv(OUT + 'india_tier_table.csv'); log(tier_tab)

# ---------------------------------------------------------------- NTC assignment (15% share, thin files more likely)
NTC_SHARE = 0.15   # TransUnion CIBIL CMI June 2026: NTC = 15% of originations, 3M ended Mar-26
z = -(d.credit_hist_yrs - d.credit_hist_yrs.median()) / d.credit_hist_yrs.std() - (d.total_acc - d.total_acc.median()) / d.total_acc.std()
logit_ntc = z.values
# choose intercept so expected share = 15%
lo, hi = -10, 10
for _ in range(60):
    mid = (lo + hi) / 2
    if (1 / (1 + np.exp(-(logit_ntc + mid)))).mean() > NTC_SHARE: hi = mid
    else: lo = mid
p_ntc = 1 / (1 + np.exp(-(logit_ntc + mid)))
d['ntc'] = (rng.random(len(d)) < p_ntc).astype(int)
log('NTC share', d.ntc.mean(), 'default rate NTC vs ETC', d.loc[d.ntc == 1, Y].mean(), d.loc[d.ntc == 0, Y].mean())

BUREAU = ['cibil', 'credit_hist_yrs', 'open_acc', 'total_acc', 'revol_util', 'log_revol_bal', 'mort_acc', 'pub_rec', 'pub_rec_bankruptcies']
for c in BUREAU:
    d.loc[d.ntc == 1, c] = np.nan
# NTC: no bureau record of existing obligations -> FOIR from the new EMI only (what a lender would observe)
d.loc[d.ntc == 1, 'foir'] = d.loc[d.ntc == 1, 'emi_ref'] / (inc[d.ntc == 1] / 12)

CAT = ['home_ownership', 'purpose']
CATS = {c: sorted(d[c].unique()) for c in CAT}
APP = ['loan_amnt', 'term', 'emp_length', 'log_annual_inc', 'loan_to_income', 'foir', 'income_verified', 'ntc'] + CAT
F_BUREAU = APP + BUREAU


def frame(x, feats):
    X = x[feats].copy()
    for c in CAT:
        if c in X: X[c] = pd.Categorical(X[c], categories=CATS[c])
    return X


def fit_lgb(feats, data):
    f, v = data[data.split == 'fit'], data[data.split == 'val']
    p = dict(objective='binary', learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=5.0, verbose=-1, num_threads=2, seed=0); p.update(BP)
    return lgb.train(p, lgb.Dataset(frame(f, feats), f[Y].values), 4000,
                     valid_sets=[lgb.Dataset(frame(v, feats), v[Y].values)], callbacks=[lgb.early_stopping(150, verbose=False)])


# ---------------------------------------------------------------- latent bureau-risk factor (for alt-data dependence on bureau risk)
# R = standardised out-of-fold log-odds from a model using the UNMASKED bureau view (what cash-flow behaviour co-moves with)
full = raw[raw.issue_d < '2016-01-01'].reset_index(drop=True)
full_feats = ['loan_amnt', 'term', 'emp_length', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_util', 'total_acc',
              'mort_acc', 'pub_rec_bankruptcies', 'log_annual_inc', 'fico_score', 'log_revol_bal']
folds = np.arange(len(full)) % 3
R = np.zeros(len(full))
for k in range(3):
    m = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=31, verbose=-1, num_threads=2),
                  lgb.Dataset(full.loc[folds != k, full_feats], full.loc[folds != k, Y]), 200)
    pk = np.clip(m.predict(full.loc[folds == k, full_feats]), 1e-4, 1 - 1e-4); R[folds == k] = np.log(pk / (1 - pk))
R = (R - R.mean()) / R.std()
Ys = (y - y.mean()) / y.std()


def make_alt(gamma, beta=0.5, seed=0):
    """Alternative-data features from latent stress S = beta*R + gamma*Ystd + eps (higher S = more stress)."""
    g = np.random.default_rng(seed)
    S = beta * R + gamma * Ys + g.normal(size=len(d))
    S = S / S.std()
    alt = pd.DataFrame(index=d.index)
    alt['aa_avg_bal_to_emi'] = np.exp(1.2 - 0.45 * S + 0.6 * g.normal(size=len(d)))           # Account Aggregator bank statements
    alt['nach_bounces_12m'] = g.poisson(np.exp(-1.6 + 0.55 * S))                               # NACH/ECS mandate returns
    alt['salary_credit_regularity'] = 1 / (1 + np.exp(-(1.8 - 0.6 * S + 0.8 * g.normal(size=len(d)))))
    alt['upi_txn_per_month'] = np.exp(3.2 - 0.25 * S + 0.7 * g.normal(size=len(d)))
    gst = 1 / (1 + np.exp(-(1.5 - 0.6 * S + 0.8 * g.normal(size=len(d)))))
    alt['gst_filing_regularity'] = np.where(d.purpose.values == 'small_business', gst, np.nan)  # MSME borrowers only
    return alt


ALT = ['aa_avg_bal_to_emi', 'nach_bounces_12m', 'salary_credit_regularity', 'upi_txn_per_month', 'gst_filing_regularity']
te_mask = (d.split == 'test').values
yte = y[te_mask]; ntc_te = d.ntc.values[te_mask] == 1
P_te = d.loan_amnt.values[te_mask]
LGD = 0.65

# ---------------------------------------------------------------- policies
def rule_policy(cibil_min=700, foir_max=0.5, ntc_rule='reject'):
    x = d[te_mask]
    ok = (x.cibil >= cibil_min) & (x.foir <= foir_max)
    if ntc_rule == 'reject':
        ok = ok & (x.ntc == 0)
    return ok.values


def summarise(approve, label):
    a = approve
    return dict(policy=label, approval_rate=a.mean(), bad_rate=yte[a].mean() if a.any() else np.nan,
                ntc_approval_rate=a[ntc_te].mean(), ntc_bad_rate=yte[a & ntc_te].mean() if (a & ntc_te).any() else np.nan,
                etc_approval_rate=a[~ntc_te].mean(),
                loss_per_100cr=100 * LGD * P_te[a & (yte == 1)].sum() / P_te[a].sum())   # INR crore lost per INR 100 crore disbursed


def model_policy_at_bad_rate(p, target_bad):
    o = np.argsort(p); cb = np.cumsum(yte[o]) / np.arange(1, len(o) + 1)
    ok = np.where(cb <= target_bad)[0]; k = ok.max() + 1 if len(ok) else 0
    a = np.zeros(len(p), bool); a[o[:k]] = True
    return a


results = {'tier_table': tier_tab.reset_index().astype({'cibil': str}).to_dict('records'),
           'ntc_share': float(d.ntc.mean()),
           'default_rate_ntc': float(d.loc[d.ntc == 1, Y].mean()), 'default_rate_etc': float(d.loc[d.ntc == 0, Y].mean())}

# M0: bureau-only CIBIL-world model
m0 = fit_lgb(F_BUREAU, d); p0 = m0.predict(frame(d[te_mask], F_BUREAU))
def seg_auc(p):
    out = {}
    for nm, mk in [('all', np.ones(len(p), bool)), ('NTC', ntc_te), ('ETC', ~ntc_te)]:
        a, lo_, hi_ = delong_auc_ci(yte[mk], p[mk]); out[nm] = dict(AUC=a, lo=lo_, hi=hi_)
    return out
results['M0_bureau_only'] = seg_auc(p0); log('M0', results['M0_bureau_only'])

# Policies at baseline: a typical prime-only bank rule is the benchmark
rules = [summarise(rule_policy(750, 0.5, 'reject'), 'Rule A: CIBIL>=750 & FOIR<=50%, NTC rejected'),
         summarise(rule_policy(700, 0.5, 'reject'), 'Rule B: CIBIL>=700 & FOIR<=50%, NTC rejected')]
target_bad = rules[0]['bad_rate']
pol = rules + [summarise(model_policy_at_bad_rate(p0, target_bad), 'Model M0 (bureau only) at Rule A bad rate')]


def ntc_inclusion(p, target):
    """Approve lowest-PD NTC applicants while the NTC approved bad rate stays <= target."""
    pn, yn = p[ntc_te], yte[ntc_te]
    o = np.argsort(pn); cb = np.cumsum(yn[o]) / np.arange(1, len(o) + 1)
    ok = np.where(cb <= target)[0]; k = ok.max() + 1 if len(ok) else 0
    good_total = (yn == 0).sum(); good_appr = (yn[o[:k]] == 0).sum()
    return dict(ntc_approved=int(k), ntc_approval_rate=k / len(pn), share_of_good_ntc_included=good_appr / good_total)

results['ntc_inclusion_M0'] = ntc_inclusion(p0, target_bad); log('NTC incl M0', results['ntc_inclusion_M0'])
store = {'y': yte, 'ntc': ntc_te, 'p0': p0}
sweep = []
for gamma in [0.0, 0.15, 0.3, 0.45, 0.6]:
    alt = make_alt(gamma)
    dd = pd.concat([d, alt], axis=1)
    alt_only_auc = roc_auc_score(yte, -alt.loc[te_mask, 'aa_avg_bal_to_emi'].values)
    m1 = fit_lgb(F_BUREAU + ALT, dd); p1 = m1.predict(frame(dd[te_mask], F_BUREAU + ALT))
    m_alt = fit_lgb(APP + ALT, dd); p_alt = m_alt.predict(frame(dd[te_mask], APP + ALT))
    store[f'p1_g{gamma}'] = p1
    s1 = seg_auc(p1)
    _, _, z, pv = delong_test(yte[ntc_te], p1[ntc_te], p0[ntc_te])
    pm = summarise(model_policy_at_bad_rate(p1, target_bad), f'Model M1 (bureau + AA/UPI), gamma={gamma}, at Rule A bad rate')
    inc = ntc_inclusion(p1, target_bad)
    row = dict(gamma=gamma, auc_all=s1['all']['AUC'], auc_ntc=s1['NTC']['AUC'], auc_ntc_lo=s1['NTC']['lo'], auc_ntc_hi=s1['NTC']['hi'],
               auc_etc=s1['ETC']['AUC'], auc_alt_only_model=roc_auc_score(yte, p_alt), auc_single_aa_feature=alt_only_auc,
               delong_p_ntc_vs_M0=pv, approval_rate=pm['approval_rate'], ntc_approval_rate_portfolio=pm['ntc_approval_rate'],
               bad_rate=pm['bad_rate'], loss_per_100cr=pm['loss_per_100cr'],
               ntc_inclusion_rate=inc['ntc_approval_rate'], ntc_good_included=inc['share_of_good_ntc_included'], ntc_approved_n=inc['ntc_approved'])
    sweep.append(row); log('SWEEP', {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    if gamma == 0.3:
        pol.append(pm)
        imp = pd.Series(m1.feature_importance('gain'), index=m1.feature_name()).sort_values(ascending=False)
        results['M1_gain_importance_gamma0.3'] = (imp / imp.sum()).round(4).to_dict()
        results['M1_gamma0.3'] = s1
np.savez_compressed(OUT + 'india_scores.npz', **store)

results['policies'] = pol
results['sweep'] = sweep
results['assumptions'] = dict(NTC_SHARE=NTC_SHARE, beta=0.5, LGD=LGD, ref_rate=0.14, cibil_map='300+(FICO-300)*600/550',
                              target_bad_rate=target_bad)
pd.DataFrame(sweep).to_csv(OUT + 'india_sweep.csv', index=False)
pd.DataFrame(pol).to_csv(OUT + 'india_policies.csv', index=False)
json.dump(results, open(OUT + 'india_results.json', 'w'), indent=2, default=float)
log('policies'); print(pd.DataFrame(pol).round(4).to_string())
log('DONE')
