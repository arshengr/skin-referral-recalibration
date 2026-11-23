"""
Step 3 - discrimination, leakage audit and the main referral / label-free recalibration experiment.

Project : Label-free recalibration of skin cancer referral thresholds under site shift
          (multi-center dermoscopy study)
Author  : Arshad Saeed, School of Information Sciences and Engineering,
          Bay Atlantic University, Washington, DC, USA (asaeed@stu.bau.edu)
Co-authors: Saira Afzal, Tahira Afzal (The Islamia University of Bahawalpur),
          Mubashir Hussain (Foundation University Islamabad)
License : MIT (see LICENSE)
"""
import os, sys, time, json, re
PROJ = os.environ.get('PROJ', './outputs/SkinCancer_Referral_Q1_2026')   # outputs: manifests, models, results
LOCAL = os.environ.get('LOCAL', './data_512')                            # resized images
for sub in ['manifests', 'manifests/milk10k_supp', 'models', 'results', 'notebooks', 'logs']:
    os.makedirs(f'{PROJ}/{sub}', exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)
# ===== ANALYSIS Cell 1: Setup (CPU is enough) =====
# Project: Label-free recalibration of skin cancer referral thresholds under site shift
# Author: Arshad Saeed, Bay Atlantic University
import os, sys, json, glob, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # referral_lib.py lives here
import referral_lib; importlib.reload(referral_lib); from referral_lib import *
RES = f'{PROJ}/results'; os.makedirs(RES, exist_ok=True)
man = {n: pd.read_csv(f'{PROJ}/manifests/{n}.csv') for n in ['ham10000', 'bcn20000', 'milk10k', 'padufes20']}
C2I = {c: i for i, c in enumerate(CLASSES)}
runs = sorted(glob.glob(f'{PROJ}/models/*__split_s*/preds.npz'))
leaky = sorted(glob.glob(f'{PROJ}/models/*__leaky_s*/preds.npz'))
print(len(runs), 'clean runs;', len(leaky), 'leaky runs')

# ===== ANALYSIS Cell 2: Table 2 — discrimination (in-domain vs external) + leakage audit =====
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
def bal_acc7(logits, df):
    m = df['label7'].isin(CLASSES).values
    return balanced_accuracy_score(df['label7'][m].map(C2I), logits[m].argmax(1))

rows = []
for f in runs + leaky:
    tag = f.split('/')[-2]; arch, split = tag.split('__'); seed = int(split[-2:])
    P = np.load(f); ham = man['ham10000']
    te = (ham[split] == 'test').values
    pr = softmax(P['ham10000_logits'])
    r = dict(model=arch, split=split.split('_')[0], seed=seed,
             ham_test_balacc=bal_acc7(P['ham10000_logits'][te], ham[te]),
             ham_test_mal_auc=roc_auc_score(ham['malignant'][te], mal_score(pr[te])))
    if split.startswith('split'):
        for n in ['bcn20000', 'milk10k', 'padufes20']:
            d = man[n]; pn = softmax(P[f'{n}_logits'])
            r[f'{n}_mal_auc'] = roc_auc_score(d['malignant'], mal_score(pn))
            r[f'{n}_balacc'] = bal_acc7(P[f'{n}_logits'], d)
    rows.append(r)
T2 = pd.DataFrame(rows)
T2.to_csv(f'{RES}/table2_discrimination_raw.csv', index=False)
summ = T2.groupby(['model', 'split']).agg(['mean', 'std']).round(3)
summ.to_csv(f'{RES}/table2_discrimination_summary.csv'); print(summ.to_string())

# leakage audit: same model + seed 42, leaky vs clean
la = T2[T2.seed == 42].pivot_table(index='model', columns='split', values=['ham_test_balacc', 'ham_test_mal_auc'])
la.to_csv(f'{RES}/table3_leakage_audit.csv'); print('\nLeakage audit (seed 42):\n', la.round(3))

# ===== ANALYSIS Cell 3: Main experiment — referral safety on new sites (label-free recalibration) =====
ALPHAS = (0.05, 0.10)
all_rows = []
for f in runs:
    tag = f.split('/')[-2]; arch, split = tag.split('__'); seed = int(split[-2:])
    P = np.load(f); ham = man['ham10000']
    cal = (ham[split] == 'val').values
    T = fit_temperature(P['ham10000_logits'][cal], ham['label7'][cal].map(C2I).values)   # source-only temperature
    pc = softmax(P['ham10000_logits'][cal] / T)
    src = dict(s_cal=mal_score(pc), y_cal_mal=ham['malignant'][cal].values,
               emb_cal=P['ham10000_emb'][cal].astype(np.float32), probs_cal=pc,
               prior_s=np.bincount(ham['label7'][(ham[split] == 'train').values].map(C2I), minlength=7) / (ham[split] == 'train').sum())
    # in-domain check (HAM10000 test split) + external sites
    sites = {'ham10000_test': (ham[(ham[split] == 'test').values], P['ham10000_logits'][(ham[split] == 'test').values], P['ham10000_emb'][(ham[split] == 'test').values])}
    for n in ['bcn20000', 'milk10k', 'padufes20']:
        sites[n] = (man[n], P[f'{n}_logits'], P[f'{n}_emb'])
    mk_ = man['milk10k']
    for st_ in sorted(mk_['site'].dropna().unique()):
        m_ = (mk_['site'] == st_).values
        if m_.sum() >= 200 and mk_['malignant'][m_].sum() >= 50:
            sites[f'milk10k_{st_}'] = (mk_[m_].reset_index(drop=True), P['milk10k_logits'][m_], P['milk10k_emb'][m_])
    for site, (d, lg, em) in sites.items():
        pt = softmax(lg / T)
        tgt = dict(s=mal_score(pt), y_mal=d['malignant'].values, is_mel=(d['label7'] == 'MEL').astype(int).values,
                   emb=em.astype(np.float32), probs=pt, lesion=d['lesion_id'].values)
        for r in evaluate_site(src, tgt, alphas=ALPHAS, seed=seed):
            r.update(model=arch, seed=seed, site=site, temperature=T)
            all_rows.append(r)
    print('done', tag)
R = pd.DataFrame(all_rows)
R.to_csv(f'{RES}/table4_referral_raw.csv', index=False)
main = R.groupby(['site', 'alpha', 'method'])[['mal_miss', 'mel_miss', 'referral_rate', 'auto_clear_npv', 'domain_auc']].agg(['mean', 'std'])
main.round(3).to_csv(f'{RES}/table4_referral_summary.csv')
print(main.round(3).to_string())

# ===== ANALYSIS Cell 4: Safety violations — how often each method breaks the promised miss rate =====
R['violation'] = R['mal_miss'] > R['alpha']
viol = R.groupby(['alpha', 'method', 'site'])['violation'].mean().unstack('site').round(2)
viol.to_csv(f'{RES}/table5_violation_rate.csv'); print(viol)
by_model = R[R.alpha == 0.05].pivot_table(index=['model'], columns=['site', 'method'], values='mal_miss', aggfunc='mean').round(3)
by_model.to_csv(f'{RES}/table6_missrate_by_model.csv')
