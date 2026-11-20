"""Publication figures (PNG, 200 dpi). Palette: validated reference categorical palette (light mode)."""
import json, pickle, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import os, sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
R = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
F = os.environ.get('P2P_FIGURES', os.path.join(ROOT, 'figures') + '/')
os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
GREY, INK, INK2, GRID = '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'figure.facecolor': 'white'})

try:
    ck = pickle.load(open(R + 'phase2_checkpoint.pkl', 'rb'))
    SC, MODELS = ck.get('SC', {}), ck.get('MODELS', {})
except Exception:
    ck, SC, MODELS = {}, {}, {}

y = None
try:
    y = np.load(R + 'phase2_scores.npz')['y']
except Exception:
    y = ck.get('R', {}).get('y_test', None)

# ---------------- Fig 2: AUC forest plot with DeLong 95% CIs (ROC curves overlap too closely to compare by eye)
order = [('DNN [5,5] (paper)', 'DNN [5,5], paper [1]', GREY), ('LR (strict ex-ante, no instalment)', 'LR, strict ex-ante', GREY),
         ('LR (paper features)', 'LR, paper [1]', GREY), ('DNN [30,1] (paper)', 'DNN [30,1], paper [1]', GREY),
         ('LGBM + engineered, IPW reject-inference', 'LightGBM + IPW reweighting', GREY),
         ('LGBM strict ex-ante (no instalment)', 'LightGBM, strict ex-ante', AQUA), ('LGBM (paper features)', 'LightGBM, paper features', GREY),
         ('Rank ensemble (LR+DNN+LGBM)', 'Rank ensemble LR+DNN+LGBM', GREY),
         ('LGBM + engineered (proposed)', 'LightGBM + engineered (proposed)', BLUE), ('XGBoost + engineered', 'XGBoost + engineered', GREY)]
import json as _j
M = _j.load(open(R + 'phase2_results.json'))['models']
order = sorted(order, key=lambda o: M[o[0]]['AUC'])
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, (k, lab, c) in enumerate(order):
    m = M[k]; ax.errorbar(m['AUC'], i, xerr=[[m['AUC'] - m['AUC_lo']], [m['AUC_hi'] - m['AUC']]], fmt='o', color=c, ms=7, capsize=3, lw=1.6)
    ax.text(m['AUC_hi'] + 0.0008, i, f"{m['AUC']:.3f}", va='center', fontsize=8.5, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order])
ax.axvline(M['LR (paper features)']['AUC'], color=INK2, lw=1, ls=':'); ax.text(M['LR (paper features)']['AUC'] + 0.0005, len(order) - 0.4, 'paper LR 0.702', fontsize=8, color=INK2)
ax.axvline(0.6938, color=RED, lw=1, ls='--'); ax.text(0.6941, -0.9, 'implied APR only', fontsize=8, color=RED)
ax.set_xlabel('Test AUC with DeLong 95% CI (n = 71,912)'); ax.set_ylim(-1.2, len(order))
ax.set_title('Fig. 2  Phase 2 discrimination by model'); ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig(F + 'fig2_auc_forest.png', dpi=200); plt.close()

# ---------------- Fig 3: strategy curves (bad rate vs approval) and profit vs approval -- two panels, one y each
cur = pd.read_csv(R + 'phase2_strategy_curves.csv')
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
cols = {'LR (paper features)': (GREY, 'LR, paper [1]'), 'DNN [5,5] (paper)': (VIOLET, 'DNN [5,5], paper [1]'),
        'LGBM strict ex-ante (no instalment)': (AQUA, 'LightGBM strict ex-ante'), 'LGBM + engineered (proposed)': (BLUE, 'LightGBM + engineered')}
for k, (c, lab) in cols.items():
    s = cur[cur.model == k]
    axs[0].plot(100 * s.approval_rate, 100 * s.bad_rate, color=c, lw=2, label=lab)
    axs[1].plot(100 * s.approval_rate, s.profit_musd, color=c, lw=2, label=lab)
axs[0].axhline(10, color=RED, lw=1, ls=':'); axs[0].text(55, 8.6, '10% bad-rate target', color=INK2, fontsize=8.5)
axs[0].set_xlabel('Approval rate among Lending Club-accepted loans (%)'); axs[0].set_ylabel('Default rate of approved loans (%)')
axs[0].set_title('(a) Bad rate vs approval rate')
axs[1].set_xlabel('Approval rate (%)'); axs[1].set_ylabel('Portfolio profit, USD million (LGD 0.65)')
axs[1].set_title('(b) Profit vs approval rate')
axs[1].legend(fontsize=8.5, loc='lower center')
fig.suptitle('Fig. 3  Strategy curves on the test set (n = 71,912)', x=0.01, ha='left', fontweight='bold', fontsize=11)
fig.tight_layout(); fig.savefig(F + 'fig3_strategy.png', dpi=200); plt.close()

# ---------------- Fig 4: calibration
cal = pd.read_csv(R + 'phase2_calibration.csv')
fig, ax = plt.subplots(figsize=(5.6, 5))
for k, c, lab in [('LR (paper features)', GREY, 'LR, paper (balanced training)'), ('DNN [5,5] (paper)', VIOLET, 'DNN [5,5], paper'),
                  ('LGBM + engineered (proposed)', ORANGE, 'LightGBM, raw'), ('LGBM + engineered, isotonic-calibrated', BLUE, 'LightGBM, isotonic-calibrated')]:
    s = cal[cal.model == k]; ax.plot(s.mean_pred, s.obs, marker='o', ms=5, color=c, lw=2, label=f"{lab}")
ax.plot([0, 0.8], [0, 0.8], color=INK2, lw=1, ls='--'); ax.text(0.55, 0.5, 'perfect calibration', color=INK2, fontsize=8.5, rotation=40)
ax.set_xlabel('Mean predicted PD (decile)'); ax.set_ylabel('Observed default rate')
ax.set_title('Fig. 4  Reliability of predicted PDs')
ax.legend(fontsize=8.5, loc='upper left'); fig.tight_layout(); fig.savefig(F + 'fig4_calibration.png', dpi=200); plt.close()

# ---------------- Fig 5: SHAP importance
imp = pd.read_csv(R + 'phase2_shap_importance.csv', index_col=0).iloc[:, 0].head(14)[::-1]
fig, ax = plt.subplots(figsize=(6.2, 5))
colors = [BLUE if i == len(imp) - 1 else GREY for i in range(len(imp))]
ax.barh(imp.index, imp.values, color=colors, height=0.6)
ax.set_xlabel('Mean |SHAP value| (log-odds)'); ax.set_title('Fig. 5  TreeSHAP importance, 20,000 test loans')
ax.grid(axis='y', visible=False); fig.tight_layout(); fig.savefig(F + 'fig5_shap.png', dpi=200); plt.close()

# ---------------- Fig 6: rolling-origin AUC
roll = pd.read_csv(R + 'phase2_rolling.csv')
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, t, ttl in [(axs[0], 'all', '(a) All terms'), (axs[1], '36m', '(b) 36-month loans only')]:
    s = roll[roll.terms == t]
    ax.plot(s.test_year, s.AUC_LR, marker='o', color=GREY, lw=2, label='LR, paper features')
    ax.plot(s.test_year, s.AUC_LGBM, marker='o', color=BLUE, lw=2, label='LightGBM + engineered')
    ax.set_xticks(s.test_year); ax.set_title(ttl); ax.set_xlabel('Test year (trained on all earlier years)')
axs[0].set_ylabel('Test AUC'); axs[0].legend(fontsize=8.5)
fig.suptitle('Fig. 6  Rolling-origin validation (train on all earlier years)', x=0.01, ha='left', fontweight='bold', fontsize=11)
fig.tight_layout(); fig.savefig(F + 'fig6_rolling.png', dpi=200); plt.close()

# ---------------- Fig 1: Phase 1 empirical rejection rate by employment length (the artefact)
emp = pd.read_csv(R + 'phase1_reject_rate_by_emp.csv')
fig, ax = plt.subplots(figsize=(6.4, 3.8))
lab = ['missing' if v < 0 else ('<1' if v == 0 else ('10+' if v == 10 else str(int(v)))) for v in emp.emp_length]
cols_ = [BLUE if v in (0, 5) else GREY for v in emp.emp_length]
ax.scatter(lab, 100 * emp['mean'], color=cols_, s=60, zorder=3)   # dots, not bars: axis does not start at zero
ax.set_ylim(92.5, 100.4); ax.set_ylabel('Rejection rate in test window (%)'); ax.set_xlabel('Employment length (years)')
for i, (v, n) in enumerate(zip(emp['mean'], emp['size'])):
    if emp.emp_length.iloc[i] in (0, 5): ax.text(i + 0.18, 100 * v - 0.35, f'n = {n/1e6:.2f}M' if n > 1e6 else f'n = {n/1e3:.0f}k', ha='left', fontsize=8.5, color=INK2)
ax.set_title('Fig. 1  Two employment-length codes mark rejections')
ax.grid(axis='x', visible=False); fig.tight_layout(); fig.savefig(F + 'fig1_phase1_emp.png', dpi=200); plt.close()
print('phase figs done')
