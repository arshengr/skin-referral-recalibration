"""
Step 1 - download HAM10000, BCN20000, MILK10k (ISIC Archive) and PAD-UFES-20 (Mendeley),
resize to 512 px, harmonize labels, build lesion-level splits, remove overlap and near-duplicates.

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
# ===== 1. Download ISIC datasets (HAM10000, BCN20000, MILK10k), resize longer side to 512 px =====
import requests, pandas as pd
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

API = 'https://api.isic-archive.com/api/v2/images/search/'
COLLECTIONS = {'ham10000': 212, 'bcn20000': 249, 'milk10k': 425}

def list_collection(cid):
    url, rows = f'{API}?collections={cid}&limit=100', []
    while url:
        for k in range(6):
            try:
                r = requests.get(url, timeout=60); r.raise_for_status(); j = r.json(); break
            except Exception:
                time.sleep(3 * (k + 1))
        else:
            raise RuntimeError(url)
        for x in j['results']:
            m = x.get('metadata', {})
            row = {'isic_id': x['isic_id'], 'url': x['files']['full']['url'], 'license': x.get('copyright_license')}
            row.update({f'acq_{k}': v for k, v in m.get('acquisition', {}).items()})
            row.update(m.get('clinical', {}))
            rows.append(row)
        url = j.get('next')
    return pd.DataFrame(rows)

def fetch_resize(job):
    url, out = job
    if os.path.exists(out):
        return True
    for k in range(5):
        try:
            r = requests.get(url, timeout=120); r.raise_for_status()
            im = Image.open(BytesIO(r.content)).convert('RGB'); im.thumbnail((512, 512), Image.LANCZOS)
            im.save(out, 'JPEG', quality=92); return True
        except Exception:
            time.sleep(2 * (k + 1))
    return False

for name, cid in COLLECTIONS.items():
    meta_path = f'{PROJ}/manifests/raw_meta_{name}.csv'
    df = pd.read_csv(meta_path) if os.path.exists(meta_path) else list_collection(cid)
    df.to_csv(meta_path, index=False)
    d = f'{LOCAL}/{name}'; os.makedirs(d, exist_ok=True)
    jobs = [(u, f'{d}/{i}.jpg') for u, i in zip(df['url'], df['isic_id'])]
    with ThreadPoolExecutor(32) as ex:
        ok = list(tqdm(ex.map(fetch_resize, jobs), total=len(jobs), desc=name, mininterval=30))
    print(f'{name}: {len(df)} listed, {len(ok) - sum(ok)} failed')

# ===== 2. MILK10k supplements (skin tone, site, MONET) + PAD-UFES-20 (smartphone) =====
import glob, subprocess, shutil
sup = f'{PROJ}/manifests/milk10k_supp'
base = 'https://isic-archive.s3.amazonaws.com/dois/10.34970-648456'
for f in ['milk10k.csv', 'supplements/training_supp.csv', 'supplements/training_gt.csv', 'supplements/training_input.csv']:
    out = f'{sup}/' + f.split('/')[-1]
    if not os.path.exists(out):
        subprocess.run(['wget', '-q', '-O', out, f'{base}/{f}'], check=True)
d = f'{LOCAL}/padufes20'
if not os.path.isdir(d) or len(os.listdir(d)) < 2000:
    raw = '/tmp/raw_pad'; os.makedirs(raw, exist_ok=True); os.makedirs(d, exist_ok=True)
    mp = 'https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files'
    subprocess.run(['wget', '-q', '-O', f'{PROJ}/manifests/raw_meta_padufes20.csv', f'{mp}/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded'], check=True)
    for i, fid in enumerate(['1cc2f71f-20a2-412d-b746-a9b9bc20c966', '559a60ed-5504-475d-996c-6a8bc253b5e7', '34dcdf8e-e5f1-4b35-aa0b-5135051ff852'], 1):
        z = f'{raw}/p{i}.zip'; print('PAD part', i)
        subprocess.run(['wget', '-q', '-O', z, f'{mp}/{fid}/file_downloaded'], check=True)
        subprocess.run(['unzip', '-q', '-o', z, '-d', raw], check=True); os.remove(z)
    def conv(p):
        out = f'{d}/' + os.path.splitext(os.path.basename(p))[0] + '.jpg'
        im = Image.open(p).convert('RGB'); im.thumbnail((512, 512), Image.LANCZOS); im.save(out, 'JPEG', quality=92)
    srcs = glob.glob(f'{raw}/**/*.png', recursive=True)
    with ThreadPoolExecutor(4) as ex:
        list(tqdm(ex.map(conv, srcs), total=len(srcs), desc='PAD', mininterval=30))
    shutil.rmtree(raw)
print('PAD images:', len(os.listdir(d)))

# ===== 3. Build clean manifests (labels, lesion-level splits, overlap removal) =====
import numpy as np, hashlib
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
# ---------- label scheme ----------
MALIGNANT = {'MEL', 'BCC', 'AKIEC', 'SCC', 'MAL_OTH'}
RULES = [  # first match wins
    (r'^mel$|melanoma', 'MEL'),
    (r'^nv$|^nev$|nevus|naevus', 'NV'),
    (r'^bcc$|basal cell', 'BCC'),
    (r'^akiec$|^ack$|actinic|bowen|in situ|intraepithelial', 'AKIEC'),
    (r'^scc$|squamous cell carcinoma', 'SCC'),
    (r'^bkl$|^sek$|seborrh|solar lentigo|lichen planus|lplk|benign keratosis|benign keratinocytic|pigmented benign keratosis', 'BKL'),
    (r'^df$|dermatofibroma', 'DF'),
    (r'^vasc$|vascular|angioma|angiokeratoma|pyogenic|hemorrhage|haemorrhage', 'VASC'),
    (r'mal_oth|other malignant|malignant', 'MAL_OTH'),
]
def map_dx(text):
    t = str(text).strip().lower()
    if t in ('', 'nan', 'none'):
        return 'OTHER'
    for pat, lab in RULES:
        if re.search(pat, t):
            return lab
    return 'OTHER'

def label_isic(df):
    # most specific diagnosis level available
    lab = pd.Series('OTHER', index=df.index)
    for col in ['diagnosis_3', 'diagnosis_4', 'diagnosis_2']:
        if col in df.columns:
            m = lab.eq('OTHER')
            lab[m] = df.loc[m, col].map(map_dx)
    return lab

def finish(df, name):
    df = df.copy()
    df['dataset'] = name
    df['malignant'] = df['label7'].isin(MALIGNANT).astype(int)
    df['path'] = name + '/' + df['image_id'].astype(str) + '.jpg'
    exists = df['path'].map(lambda p: os.path.exists(f'{LOCAL}/{p}'))
    if (~exists).any():
        print(f'{name}: {int((~exists).sum())} images missing on disk -> dropped')
    df = df[exists].reset_index(drop=True)
    print(f'\n== {name}: {len(df)} images, {df["lesion_id"].nunique()} lesions, malignant={df["malignant"].sum()}')
    print(df['label7'].value_counts().to_string())
    return df

# ---------- HAM10000 ----------
h = pd.read_csv(f'{PROJ}/manifests/raw_meta_ham10000.csv')
if 'acq_image_type' in h.columns:
    h = h[h['acq_image_type'] == 'dermoscopic']
h['label7'] = label_isic(h)
print('HAM10000 unmapped diagnoses:', h.loc[h.label7 == 'OTHER', ['diagnosis_1', 'diagnosis_2', 'diagnosis_3']].value_counts().head(20).to_dict())
h = h[h['label7'] != 'OTHER']
h['lesion_id'] = h['lesion_id'].fillna(h['isic_id'])
ham = finish(h.rename(columns={'isic_id': 'image_id'})[['image_id', 'lesion_id', 'label7', 'diagnosis_1', 'diagnosis_3', 'age_approx', 'sex', 'anatom_site_1']], 'ham10000')

# lesion-level split 70/15/15 x 5 seeds
SEEDS = [42, 43, 44, 45, 46]
for s in SEEDS:
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=s)
    fold = np.empty(len(ham), dtype=int)
    for f, (_, idx) in enumerate(sgkf.split(ham, ham['label7'], groups=ham['lesion_id'])):
        fold[idx] = f
    ham[f'split_s{s}'] = np.where(fold < 14, 'train', np.where(fold < 17, 'val', 'test'))
    assert (ham.groupby('lesion_id')[f'split_s{s}'].nunique() == 1).all()
    # leaky image-level split (for the leakage audit only)
    tr, rest = train_test_split(ham.index, test_size=0.30, stratify=ham['label7'], random_state=s)
    va, te = train_test_split(rest, test_size=0.50, stratify=ham.loc[rest, 'label7'], random_state=s)
    ham[f'leaky_s{s}'] = 'train'; ham.loc[va, f'leaky_s{s}'] = 'val'; ham.loc[te, f'leaky_s{s}'] = 'test'
print('\nlesions spread across subsets in leaky split (seed 42):',
      int((ham.groupby('lesion_id')['leaky_s42'].nunique() > 1).sum()))
print(pd.crosstab(ham['label7'], ham['split_s42']))

# ---------- BCN20000 ----------
b = pd.read_csv(f'{PROJ}/manifests/raw_meta_bcn20000.csv')
b['label7'] = label_isic(b)
print('\nBCN20000 unmapped:', b.loc[b.label7 == 'OTHER', 'diagnosis_3'].value_counts().head(20).to_dict())
b = b[b['label7'] != 'OTHER']; b['lesion_id'] = b['lesion_id'].fillna(b['isic_id'])
ext = {'bcn20000': finish(b.rename(columns={'isic_id': 'image_id'})[['image_id', 'lesion_id', 'label7', 'diagnosis_1', 'diagnosis_3', 'age_approx', 'sex', 'anatom_site_1']], 'bcn20000')}

# ---------- MILK10k (dermoscopic only) + skin tone / MONET ----------
m = pd.read_csv(f'{PROJ}/manifests/raw_meta_milk10k.csv')
m = m[m['acq_image_type'].astype(str).str.contains('dermoscop', case=False)]
m['label7'] = label_isic(m)
print('\nMILK10k unmapped:', m.loc[m.label7 == 'OTHER', 'diagnosis_3'].value_counts().head(20).to_dict())
m = m[m['label7'] != 'OTHER']
supp_dir = f'{PROJ}/manifests/milk10k_supp'
for f in ['training_supp.csv', 'milk10k.csv']:
    p = f'{supp_dir}/{f}'
    if os.path.exists(p):
        s_ = pd.read_csv(p); print(f'{f} columns:', list(s_.columns)[:30])
        key = next((k for k in ['isic_id', 'lesion_id'] if k in s_.columns and k in m.columns), None)
        if key:
            new = [c for c in s_.columns if c not in m.columns or c == key]
            m = m.merge(s_[new].drop_duplicates(key), on=key, how='left')
keep = ['image_id', 'lesion_id', 'label7', 'diagnosis_1', 'diagnosis_3', 'age_approx', 'sex', 'anatom_site_1'] + \
       [c for c in m.columns if re.search('skin_tone|monet|fitz', c, re.I)]
ext['milk10k'] = finish(m.rename(columns={'isic_id': 'image_id'})[[c for c in keep if c in m.rename(columns={'isic_id': 'image_id'}).columns]], 'milk10k')

# ---------- PAD-UFES-20 (smartphone) ----------
p = pd.read_csv(f'{PROJ}/manifests/raw_meta_padufes20.csv')
p['label7'] = p['diagnostic'].map(map_dx)
p['image_id'] = p['img_id'].str.replace(r'\.\w+$', '', regex=True)
p['lesion_id'] = 'PAD_' + p['lesion_id'].astype(str)
p = p.rename(columns={'age': 'age_approx', 'gender': 'sex', 'region': 'anatom_site_1', 'fitspatrick': 'fitzpatrick'})
ext['padufes20'] = finish(p[['image_id', 'lesion_id', 'label7', 'diagnostic', 'biopsed', 'age_approx', 'sex', 'anatom_site_1', 'fitzpatrick']], 'padufes20')

# ---------- remove overlap with HAM10000 ----------
ham_ids, ham_les = set(ham['image_id']), set(ham['lesion_id'])
for n, d in ext.items():
    hit = d['image_id'].isin(ham_ids) | d['lesion_id'].isin(ham_les)
    print(f'{n}: overlap with HAM10000 by image/lesion id = {int(hit.sum())} -> removed')
    ext[n] = d[~hit].reset_index(drop=True)

# ===== 4. Near-duplicate removal (perceptual hash) + save manifests =====
import imagehash, json, hashlib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def ph(path):
    try:
        with Image.open(f'{LOCAL}/{path}') as im:
            return imagehash.phash(im.convert('RGB').resize((256, 256))).hash.flatten()
    except Exception:
        return None

with ThreadPoolExecutor(8) as ex:
    H = np.array([h for h in tqdm(ex.map(ph, ham['path']), total=len(ham), desc='hash HAM10000') if h is not None])
for n, d in ext.items():
    with ThreadPoolExecutor(8) as ex:
        hs = list(tqdm(ex.map(ph, d['path']), total=len(d), desc=f'hash {n}'))
    dup = np.array([h is not None and bool((np.count_nonzero(H != h, axis=1) <= 4).any()) for h in hs])
    print(f'{n}: near-duplicates of HAM10000 = {int(dup.sum())} -> removed')
    ext[n] = d[~dup].reset_index(drop=True)

def save(df, name):
    pth = f'{PROJ}/manifests/{name}.csv'
    df.to_csv(pth, index=False)
    return {'dataset': name, 'images': len(df), 'lesions': df['lesion_id'].nunique(),
            'malignant': int(df['malignant'].sum()), 'melanoma': int((df['label7'] == 'MEL').sum()),
            'sha256': hashlib.sha256(open(pth, 'rb').read()).hexdigest()[:16]}

summary = pd.DataFrame([save(ham, 'ham10000')] + [save(d, n) for n, d in ext.items()])
summary.to_csv(f'{PROJ}/manifests/MANIFEST_SUMMARY.csv', index=False)
print(summary.to_string(index=False))
allx = pd.concat([ham[['dataset', 'label7']]] + [d[['dataset', 'label7']] for d in ext.values()])
tab = pd.crosstab(allx['label7'], allx['dataset'], margins=True)
tab.to_csv(f'{PROJ}/results/table1_label_counts.csv'); print(tab)

# ===== 5. Label fix (SCC -> AKIEC for 7-class head) + MILK10k skin tone / site / MONET merge =====
for name in ['ham10000', 'bcn20000', 'milk10k', 'padufes20']:
    p = f'{PROJ}/manifests/{name}.csv'; d_ = pd.read_csv(p)
    if 'label_fine' not in d_.columns:
        d_['label_fine'] = d_['label7']
    d_['label7'] = d_['label_fine'].replace({'SCC': 'AKIEC'})
    d_.to_csv(p, index=False)
ti = pd.read_csv(f'{PROJ}/manifests/milk10k_supp/training_input.csv')
extra = [c for c in ti.columns if c not in ('lesion_id', 'image_type', 'attribution', 'copyright_license', 'image_manipulation', 'age_approx', 'sex', 'anatom_site_general')]
mk = pd.read_csv(f'{PROJ}/manifests/milk10k.csv')
mk = mk.drop(columns=[c for c in extra if c in mk.columns and c != 'isic_id'])
mk = mk.merge(ti[extra].rename(columns={'isic_id': 'image_id'}), on='image_id', how='left')
mk.to_csv(f'{PROJ}/manifests/milk10k.csv', index=False)
print('skin tone:', mk['skin_tone_class'].value_counts().sort_index().to_dict())
print('sites:', mk['site'].value_counts().to_dict())
print(pd.read_csv(f'{PROJ}/manifests/MANIFEST_SUMMARY.csv').to_string(index=False))
