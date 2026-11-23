"""Turn the Kaggle result CSVs into manuscript-ready markdown tables + key numbers.
Usage: python make_results.py <results_dir> [out.md]
Author: Arshad Saeed (Bay Atlantic University)."""
import sys, json, numpy as np, pandas as pd

RES = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else 'results_for_manuscript.md'
NAME = {'resnet50': 'ResNet-50', 'mobilenetv2': 'MobileNetV2', 'xception': 'Xception',
        'efficientnet_b3': 'EfficientNet-B3', 'convnext_tiny': 'ConvNeXt-T', 'dinov2_vitb14': 'DINOv2 (probe)'}
SITE = {'ham10000_test': 'HAM10000 (internal test)', 'bcn20000': 'BCN20000', 'milk10k': 'MILK10k (all)',
        'padufes20': 'PAD-UFES-20'}
METH = ['SRC', 'WCP', 'QTC', 'QTC_LS', 'ORACLE_20', 'ORACLE_50']
MLAB = {'SRC': 'Source-only (SRC)', 'WCP': 'Weighted conformal (WCP)', 'QTC': 'Quantile transfer (QTC)',
        'QTC_LS': 'QTC + label shift', 'ORACLE_20': 'Oracle, 20 labels', 'ORACLE_50': 'Oracle, 50 labels'}

def nm(m):
    for k, v in NAME.items():
        if m.startswith(k): return v
    return m
def site(s): return SITE.get(s, s.replace('milk10k_', 'MILK10k – '))
def ms(x): return f'{np.nanmean(x):.3f} ± {np.nanstd(x, ddof=1) if len(x) > 1 else 0:.3f}'
def pct(x): return f'{100 * x:.1f}%'
def md(df):
    cols = list(df.columns)
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.values])

T2 = pd.read_csv(f'{RES}/table2_discrimination_raw.csv')
R = pd.read_csv(f'{RES}/table4_referral_raw.csv')
key, out = {}, []

# ---- Table 2: discrimination (lesion-level split, mean ± sd over seeds) ----
cl = T2[T2.split == 'split']
rows = []
for m, g in cl.groupby('model'):
    r = {'Model': nm(m), 'HAM bal. acc.': ms(g.ham_test_balacc), 'HAM malignant AUROC': ms(g.ham_test_mal_auc)}
    for s in ['bcn20000', 'milk10k', 'padufes20']:
        if f'{s}_mal_auc' in g: r[f'{SITE[s]} AUROC'] = ms(g[f'{s}_mal_auc'])
    rows.append(r)
t2 = pd.DataFrame(rows); out += ['## Table 2. Discrimination (lesion-level split, mean ± SD over seeds)', md(t2), '']

# ---- Table 3: leakage audit ----
lk = T2[T2.seed == 42].pivot_table(index='model', columns='split', values=['ham_test_balacc', 'ham_test_mal_auc'])
if ('ham_test_balacc', 'leaky') in lk.columns:
    lk = lk.dropna()
    t3 = pd.DataFrame({'Model': [nm(m) for m in lk.index],
                       'Bal. acc. image split': lk[('ham_test_balacc', 'leaky')].round(3).values,
                       'Bal. acc. lesion split': lk[('ham_test_balacc', 'split')].round(3).values,
                       'Inflation (pts)': ((lk[('ham_test_balacc', 'leaky')] - lk[('ham_test_balacc', 'split')]) * 100).round(1).values,
                       'AUROC image split': lk[('ham_test_mal_auc', 'leaky')].round(3).values,
                       'AUROC lesion split': lk[('ham_test_mal_auc', 'split')].round(3).values})
    out += ['## Table 3. Leakage audit (seed 42)', md(t3), '']
    infl = t3['Inflation (pts)']
    key['leak_inflation_range_pts'] = f'{infl.min():.1f}–{infl.max():.1f}'
    key['leak_inflation_mean_pts'] = round(float(infl.mean()), 1)

# ---- Table 4: referral safety ----
R['violation'] = R.mal_miss > R.alpha
for a in sorted(R.alpha.unique()):
    rows = []
    for s in [x for x in ['ham10000_test', 'bcn20000', 'milk10k', 'padufes20'] if x in R.site.unique()] + \
             sorted(x for x in R.site.unique() if x.startswith('milk10k_')):
        for m in METH:
            g = R[(R.alpha == a) & (R.site == s) & (R.method == m)]
            if g.empty: continue
            rows.append({'Site': site(s), 'Method': MLAB[m], 'Malignant miss': ms(g.mal_miss),
                         'Melanoma miss': ms(g.mel_miss), 'Referral rate': ms(g.referral_rate),
                         'NPV auto-cleared': ms(g.auto_clear_npv), 'Runs breaking promise': pct(g.violation.mean()),
                         'Domain AUC': f'{g.domain_auc.mean():.2f}'})
    out += [f'## Table 4. Referral safety at α = {a:.2f} (mean ± SD over 6 models × 3 seeds)', md(pd.DataFrame(rows)), '']


# ---- Table 5: paired statistical tests (same model, seed, site, alpha) ----
from itertools import combinations
from scipy.stats import wilcoxon, binomtest
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj
EXT = ['bcn20000', 'milk10k', 'padufes20']
LF = ['SRC', 'WCP', 'QTC', 'QTC_LS']
rows = []
for a in sorted(R.alpha.unique()):
    sub = R[(R.alpha == a) & R.site.isin(EXT) & R.method.isin(LF)]
    P = sub.pivot_table(index=['model', 'seed', 'site'], columns='method', values=['mal_miss', 'referral_rate', 'violation'])
    for m1, m2 in combinations(LF, 2):
        r = {'alpha': a, 'Comparison': f'{MLAB[m1]} vs {MLAB[m2]}', 'n pairs': len(P)}
        for v in ['mal_miss', 'referral_rate']:
            d = (P[(v, m1)] - P[(v, m2)]).dropna()
            r[f'Δ {v} (median)'] = round(float(d.median()), 3)
            r[f'p {v}'] = float(wilcoxon(d, zero_method='zsplit').pvalue) if (d != 0).any() else 1.0
        b = (P[('violation', m1)] > 0.5) & (P[('violation', m2)] < 0.5)
        c = (P[('violation', m1)] < 0.5) & (P[('violation', m2)] > 0.5)
        r['breaks only first / only second'] = f'{int(b.sum())} / {int(c.sum())}'
        r['p violation (McNemar)'] = float(binomtest(int(b.sum()), int(b.sum() + c.sum()), 0.5).pvalue) if (b.sum() + c.sum()) else 1.0
        rows.append(r)
T5 = pd.DataFrame(rows)
for col in [c for c in T5.columns if c.startswith('p ')]:
    T5[col + ' (Holm)'] = T5.groupby('alpha')[col].transform(lambda x: holm(x.values)).map(lambda v: f'{v:.3g}')
    T5 = T5.drop(columns=col)
out += ['## Table 5. Paired comparisons across external sites (Wilcoxon / exact McNemar, Holm-adjusted)', md(T5), '']

# ---- key numbers for abstract / highlights / cover letter ----
ext = R[(R.alpha == 0.05) & (R.site.isin(['bcn20000', 'milk10k', 'padufes20']))]
for m in ['SRC', 'WCP', 'QTC', 'QTC_LS']:
    g = ext[ext.method == m]
    key[f'{m}_ext_mal_miss_mean'] = round(float(g.mal_miss.mean()), 3)
    key[f'{m}_ext_mal_miss_max_site'] = round(float(g.groupby('site').mal_miss.mean().max()), 3)
    key[f'{m}_ext_violation_rate'] = round(float(g.violation.mean()), 3)
    key[f'{m}_ext_referral_mean'] = round(float(g.referral_rate.mean()), 3)
    key[f'{m}_ext_mel_miss_mean'] = round(float(g.mel_miss.mean()), 3)
per_site = ext.groupby(['site', 'method']).mal_miss.mean().unstack().round(3)
out += ['## Key: mean malignant miss at α = 0.05 by site', md(per_site.reset_index()), '',
        '## Key numbers', '```json', json.dumps(key, indent=1), '```']
key['n_runs'] = int(R.groupby(['model', 'seed']).ngroups)
open(OUT, 'w').write('\n'.join(out))
json.dump(key, open(OUT.replace('.md', '_key.json'), 'w'), indent=1)
print('\n'.join(out))
