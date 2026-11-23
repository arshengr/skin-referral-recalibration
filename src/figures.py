"""
Step 4 - figures, lesion-bootstrap confidence intervals, skin-tone and artifact audits.

Project : Label-free recalibration of skin cancer referral thresholds under site shift
          (multi-center dermoscopy study)
Author  : Arshad Saeed, School of Information Sciences and Engineering,
          Bay Atlantic University, Washington, DC, USA (asaeed@stu.bau.edu)
Co-authors: Saira Afzal, Tahira Afzal (The Islamia University of Bahawalpur),
          Mubashir Hussain (Foundation University Islamabad)
License : MIT (see LICENSE)
"""
# ===== FIG 0. Setup: read outputs of the main pipeline run =====
# Project: Label-free recalibration of skin cancer referral thresholds under site shift
# Author: Arshad Saeed, Bay Atlantic University
import os, sys, glob, json, importlib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INP = os.environ.get('PROJ', './outputs/SkinCancer_Referral_Q1_2026')
OUT = os.environ.get('FIG_OUT', './figures'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import referral_lib; importlib.reload(referral_lib); from referral_lib import *
man = {n: pd.read_csv(f'{INP}/manifests/{n}.csv') for n in ['ham10000', 'bcn20000', 'milk10k', 'padufes20']}
C2I = {c: i for i, c in enumerate(CLASSES)}
R = pd.read_csv(f'{INP}/results/table4_referral_raw.csv')
T2 = pd.read_csv(f'{INP}/results/table2_discrimination_raw.csv')
print('input:', INP, '| referral rows:', len(R), '| sites:', sorted(R.site.unique()))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
METHOD_ORDER = ['SRC', 'QTC', 'QTC_LS', 'WCP', 'ORACLE_20', 'ORACLE_50']
METHOD_LABEL = {'SRC': 'Source-only', 'QTC': 'QTC', 'QTC_LS': 'QTC + label shift', 'WCP': 'Weighted conformal',
                'ORACLE_20': 'Oracle (20 labels)', 'ORACLE_50': 'Oracle (50 labels)'}
SITE_LABEL = lambda s: s.replace('ham10000_test', 'HAM10000 (internal)').replace('bcn20000', 'BCN20000').replace('padufes20', 'PAD-UFES-20 (phone)').replace('milk10k_', 'MILK10k: ').replace('milk10k', 'MILK10k (all)')

# ===== FIG 1. Leakage audit: image-level vs lesion-level split (seed 42) =====
la = T2[T2.seed == 42].pivot_table(index='model', columns='split', values='ham_test_balacc').dropna()
fig, ax = plt.subplots(figsize=(5, 2.6))
x = np.arange(len(la))
ax.bar(x - 0.18, la['leaky'], 0.36, label='Image-level split (leaky)', color='#bbbbbb')
ax.bar(x + 0.18, la['split'], 0.36, label='Lesion-level split (clean)', color='#2b6cb0')
for i, (a, b) in enumerate(zip(la['leaky'], la['split'])):
    ax.text(i, max(a, b) + 0.01, f'+{(a - b) * 100:.1f} pts', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(la.index, rotation=15); ax.set_ylabel('Balanced accuracy (HAM10000 test)')
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.set_title('Image-level splitting inflates HAM10000 results')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_leakage_audit.png', dpi=300); fig.savefig(f'{OUT}/fig_leakage_audit.pdf')
la.assign(inflation_pts=(la['leaky'] - la['split']) * 100).round(3).to_csv(f'{OUT}/table_leakage_audit.csv'); print(la.round(3))

# ===== FIG 2. Main result: malignant miss rate per site and method (alpha = 0.05) =====
def site_order(df):
    s = sorted(df.site.unique(), key=lambda v: (v != 'ham10000_test', v.startswith('milk10k_'), v))
    return s
for a in sorted(R.alpha.unique()):
    d = R[R.alpha == a]
    sites = site_order(d); meths = [m for m in METHOD_ORDER if m in d.method.unique()]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(sites) + 1.2))
    for j, m in enumerate(meths):
        g = d[d.method == m].groupby('site')['mal_miss']
        y = np.arange(len(sites)) + (j - (len(meths) - 1) / 2) * 0.12
        med = [g.median().get(s, np.nan) for s in sites]; lo = [g.min().get(s, np.nan) for s in sites]; hi = [g.max().get(s, np.nan) for s in sites]
        ax.errorbar(med, y, xerr=[np.array(med) - lo, np.array(hi) - med], fmt='o', ms=3.5, lw=1, capsize=0,
                    label=METHOD_LABEL[m], color=plt.cm.tab10(j))
    ax.axvline(a, ls='--', color='k', lw=0.8)
    ax.text(a, 1.0, f' promised ≤ {a:.0%}', fontsize=8, va='bottom', transform=ax.get_xaxis_transform())
    ax.set_yticks(np.arange(len(sites))); ax.set_yticklabels([SITE_LABEL(s) for s in sites]); ax.invert_yaxis()
    ax.set_xlabel('Share of malignant cases auto-cleared (median and range over models × seeds)')
    ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f'Does the referral rule keep its promise on new sites? (α = {a:.2f})', pad=14)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_missrate_alpha{int(a*100):02d}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{OUT}/fig_missrate_alpha{int(a*100):02d}.pdf', bbox_inches='tight')

# ===== FIG 3. Safety vs workload: miss rate vs referral rate (alpha = 0.05) =====
d = R[(R.alpha == 0.05) & (R.site != 'ham10000_test')]
fig, ax = plt.subplots(figsize=(5, 3.4))
for j, m in enumerate([m for m in METHOD_ORDER if m in d.method.unique()]):
    g = d[d.method == m].groupby('site')[['mal_miss', 'referral_rate']].mean()
    ax.scatter(g['referral_rate'], g['mal_miss'], s=18, label=METHOD_LABEL[m], color=plt.cm.tab10(j))
ax.axhline(0.05, ls='--', color='k', lw=0.8)
ax.set_xlabel('Referral rate (share of cases sent to dermatologist)'); ax.set_ylabel('Malignant miss rate')
ax.legend(frameon=False, fontsize=7); ax.set_title('Safety vs workload per site (mean over models)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_safety_vs_workload.png', dpi=300); fig.savefig(f'{OUT}/fig_safety_vs_workload.pdf')

# ===== FIG 4. Shift size vs safety breach (source-only vs weighted conformal) =====
d = R[(R.alpha == 0.05) & R.method.isin(['SRC', 'WCP'])]
fig, ax = plt.subplots(figsize=(5, 3.2))
for j, m in enumerate(['SRC', 'WCP']):
    g = d[d.method == m]
    ax.scatter(g['domain_auc'], g['mal_miss'], s=8, alpha=0.5, label=METHOD_LABEL[m], color=plt.cm.tab10([0, 3][j]))
ax.axhline(0.05, ls='--', color='k', lw=0.8)
ax.set_xlabel('Domain-classifier AUROC (how different the site is)'); ax.set_ylabel('Malignant miss rate')
ax.legend(frameon=False, fontsize=8); ax.set_title('Larger shift, larger breach for the source-only rule')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_shift_vs_breach.png', dpi=300); fig.savefig(f'{OUT}/fig_shift_vs_breach.pdf')
print(d.groupby('method')[['domain_auc', 'mal_miss']].corr().round(3))

# ===== TABLE. Per-image decisions for one model -> lesion-bootstrap CIs, skin tone and artifact audit =====
def decisions(f, alpha=0.05, seed=42):
    """Recompute per-image auto-clear decisions for every site and method for one run."""
    tag = f.split('/')[-2]; arch, split = tag.split('__')
    P = np.load(f); ham = man['ham10000']
    cal = (ham[split] == 'val').values
    T = fit_temperature(P['ham10000_logits'][cal], ham['label7'][cal].map(C2I).values)
    pc = softmax(P['ham10000_logits'][cal] / T); s_cal = mal_score(pc); y_cal = ham['malignant'][cal].values
    s_mal = s_cal[y_cal == 1]
    out = {}
    sites = {'bcn20000': man['bcn20000'], 'milk10k': man['milk10k'], 'padufes20': man['padufes20']}
    for n, d_ in sites.items():
        pt = softmax(P[f'{n}_logits'] / T); s = mal_score(pt)
        w_src, w_tgt, _ = density_ratio(P['ham10000_emb'][cal].astype(np.float32), P[f'{n}_emb'].astype(np.float32), seed=seed)
        tau = {'SRC': np.full(len(s), conformal_lower(s_mal, alpha)),
               'WCP': weighted_lower(s_mal, w_src[y_cal == 1], w_tgt, alpha)}
        out[n] = pd.DataFrame({'image_id': d_['image_id'], 'lesion_id': d_['lesion_id'], 'malignant': d_['malignant'],
                               'label7': d_['label7'], 's': s, **{f'clear_{k}': s < v for k, v in tau.items()}})
    return arch, out

runs = sorted(glob.glob(f'{INP}/models/*__split_s42/preds.npz'))
best = T2[(T2.split == 'split') & (T2.seed == 42)].set_index('model')[[c for c in T2.columns if c.endswith('_mal_auc') and not c.startswith('ham')]].mean(axis=1).idxmax()
f_best = [f for f in runs if f.split('/')[-2].startswith(best + '__')][0]
arch, D = decisions(f_best)
print('model used for per-image analysis:', arch)

rows = []
for n, df in D.items():
    for m in ['SRC', 'WCP']:
        tgt = dict(y_mal=df['malignant'].values, lesion=df['lesion_id'].values)
        c = df[f'clear_{m}'].values
        (mlo, mhi), (rlo, rhi) = lesion_bootstrap(c, tgt, B=1000)
        rows.append(dict(site=n, method=m, mal_miss=c[df.malignant == 1].mean(), miss_ci=f'{mlo:.3f}–{mhi:.3f}',
                         referral=(~c).mean(), referral_ci=f'{rlo:.3f}–{rhi:.3f}'))
CI = pd.DataFrame(rows); CI.round(3).to_csv(f'{OUT}/table_bootstrap_ci_{arch}.csv', index=False); print(CI.round(3).to_string(index=False))

# ===== AUDIT 1. MILK10k skin tone (descriptive only: very few dark-skin images) =====
mk = man['milk10k'][['image_id', 'skin_tone_class']].merge(D['milk10k'], on='image_id')
mk['tone_group'] = pd.cut(mk['skin_tone_class'], [-1, 2, 3, 5], labels=['0–2 (darker)', '3 (medium)', '4–5 (lighter)'])
st = (mk[mk.malignant == 1].groupby('tone_group', observed=True)
      .agg(n_malignant=('malignant', 'size'), miss_SRC=('clear_SRC', 'mean'), miss_WCP=('clear_WCP', 'mean')))
st.round(3).to_csv(f'{OUT}/table_skin_tone_audit_{arch}.csv'); print(st.round(3))

# ===== AUDIT 2. Artifacts (MONET concept scores) among missed malignant cases =====
monet = [c for c in man['milk10k'].columns if c.startswith('MONET_')]
mm = man['milk10k'][['image_id'] + monet].merge(D['milk10k'], on='image_id')
mal = mm[mm.malignant == 1]
art = pd.DataFrame({c: [mal.loc[mal.clear_SRC, c].mean(), mal.loc[~mal.clear_SRC, c].mean()] for c in monet},
                   index=['missed (auto-cleared)', 'caught (referred)']).T
art['difference'] = art.iloc[:, 0] - art.iloc[:, 1]
art.sort_values('difference', ascending=False).round(3).to_csv(f'{OUT}/table_artifact_audit_{arch}.csv'); print(art.sort_values('difference', ascending=False).round(3))
print('\nSaved figures and tables to', OUT, ':', sorted(os.listdir(OUT)))
