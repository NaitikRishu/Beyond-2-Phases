"""India extension figures."""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
R = os.environ.get('P2P_RESULTS', os.path.join(ROOT, 'results') + '/')
F = os.environ.get('P2P_FIGURES', os.path.join(ROOT, 'figures') + '/')
os.makedirs(F, exist_ok=True)
BLUE, AQUA, RED, GREY, INK2, GRID = '#2a78d6', '#1baf7a', '#e34948', '#8a8984', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
r = json.load(open(R + 'india_results.json')); sw = pd.DataFrame(r['sweep'])
m0 = r['M0_bureau_only']['NTC']

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axs[0]
ax.axhspan(m0['lo'], m0['hi'], color=GREY, alpha=0.15, lw=0)
ax.axhline(m0['AUC'], color=GREY, lw=2); ax.text(0.36, m0['AUC'] - 0.009, f"M0 bureau-only, NTC: {m0['AUC']:.3f}", color=INK2, fontsize=8.5)
ax.errorbar(sw.gamma, sw.auc_ntc, yerr=[sw.auc_ntc - sw.auc_ntc_lo, sw.auc_ntc_hi - sw.auc_ntc], color=BLUE, marker='o', ms=6, lw=2, capsize=3, label='M1 bureau + AA/UPI, NTC borrowers')
ax.plot(sw.gamma, sw.auc_etc, color=AQUA, marker='o', ms=5, lw=2, label='M1, existing-to-credit borrowers')
ax.axhline(r['M0_bureau_only']['ETC']['AUC'], color=AQUA, lw=1, ls=':'); ax.text(0.36, r['M0_bureau_only']['ETC']['AUC'] - 0.008, f"M0 bureau-only, ETC: {r['M0_bureau_only']['ETC']['AUC']:.3f}", color=INK2, fontsize=8.5)
ax.set_xlabel('γ: outcome signal in alternative data (0 = none beyond bureau)'); ax.set_ylabel('Test AUC')
ax.set_title('(a) Discrimination for thin-file borrowers'); ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(0.27, 1.0))

ax = axs[1]
base = r['ntc_inclusion_M0']['ntc_approval_rate'] * 100
ax.axhline(base, color=GREY, lw=2); ax.text(0.2, base + 1.0, f'M0 bureau-only: {base:.1f}%', color=INK2, fontsize=8.5)
ax.axhline(0, color=RED, lw=1.5, ls='--'); ax.text(0.2, 1.0, 'Rule A (CIBIL≥750): 0% — NTC rejected', color=RED, fontsize=8.5)
ax.plot(sw.gamma, 100 * sw.ntc_inclusion_rate, color=BLUE, marker='o', ms=6, lw=2)
for g, v in zip(sw.gamma, 100 * sw.ntc_inclusion_rate):
    ax.text(g, v + 1.5, f'{v:.0f}%', ha='center', fontsize=8.5, color=INK2)
ax.set_ylim(-3, max(100 * sw.ntc_inclusion_rate) + 10)
ax.set_xlabel('γ: outcome signal in alternative data'); ax.set_ylabel('NTC applicants approved (%)')
ax.set_title(f"(b) NTC approvals at Rule A's bad rate ({100*r['assumptions']['target_bad_rate']:.1f}%)")
for a_ in axs:
    a_.axvline(0.15, color=INK2, lw=1, ls='--')
axs[0].text(0.155, 0.762, 'anchored to Berg et al. [21]', fontsize=8, color=INK2)
fig.suptitle('Fig. 7  India CIBIL-world: what Account Aggregator / UPI data could add for new-to-credit borrowers', x=0.01, ha='left', fontweight='bold', fontsize=11)
fig.tight_layout(); fig.savefig(F + 'fig7_india.png', dpi=200); plt.close()
print('india fig done')
