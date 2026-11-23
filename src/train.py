"""
Step 2 - train 5 CNNs x 3 seeds (lesion-level split), 4 leakage-audit runs (image-level split),
and a frozen DINOv2 linear probe; save logits and embeddings for every dataset.

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
import pandas as pd
# ===== TRAIN Cell 2: Pre-decode all images to 224x224 uint8 arrays (fast training on Colab) =====
import numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

SIZE = 224
DATASETS = ['ham10000', 'bcn20000', 'milk10k', 'padufes20']
man = {n: pd.read_csv(f'{PROJ}/manifests/{n}.csv') for n in DATASETS if os.path.exists(f'{PROJ}/manifests/{n}.csv')}
for n, d in man.items():
    print(n, len(d))

def load(p):
    im = Image.open(f'{LOCAL}/{p}').convert('RGB')
    w, h = im.size; s = min(w, h)                          # center square crop, then resize
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)

ARR = {}
for n, d in man.items():
    f = f'{LOCAL}/../{n}_{SIZE}.npy'
    if not os.path.exists(f):
        X = np.lib.format.open_memmap(f, mode='w+', dtype=np.uint8, shape=(len(d), SIZE, SIZE, 3))
        with ThreadPoolExecutor(8) as ex:
            for i, a in enumerate(tqdm(ex.map(load, d['path']), total=len(d), desc=n)):
                X[i] = a
        X.flush(); del X
    ARR[n] = np.load(f, mmap_mode='r')
print({n: a.shape for n, a in ARR.items()})

# ===== TRAIN Cell 3: Training code (7-class, GPU augmentation, AMP) =====
import torch, torch.nn as nn, torch.nn.functional as F, timm, math, random

DEV = 'cuda'
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
C2I = {c: i for i, c in enumerate(CLASSES)}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

MODELS = {   # thesis CNNs + modern baselines
    'resnet50': 'resnet50.a1_in1k',
    'mobilenetv2': 'mobilenetv2_100.ra_in1k',
    'xception': 'legacy_xception.tf_in1k',
    'efficientnet_b3': 'efficientnet_b3.ra2_in1k',
    'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k',
}

def to_gpu(x_uint8):
    x = torch.from_numpy(np.ascontiguousarray(x_uint8)).to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.
    return (x - MEAN) / STD

def gpu_augment(x):
    B = x.shape[0]
    if random.random() < 0.5: x = x.flip(3)
    if random.random() < 0.5: x = x.flip(2)
    k = random.randint(0, 3); x = torch.rot90(x, k, (2, 3))
    # random resized crop + small color jitter (per batch, cheap)
    s = random.uniform(0.75, 1.0); hh = int(SIZE * s)
    i0, j0 = random.randint(0, SIZE - hh), random.randint(0, SIZE - hh)
    x = F.interpolate(x[:, :, i0:i0 + hh, j0:j0 + hh], size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    x = x * (1 + 0.2 * (torch.rand(B, 1, 1, 1, device=DEV) - 0.5)) + 0.2 * (torch.rand(B, 3, 1, 1, device=DEV) - 0.5)
    return x

@torch.no_grad()
def predict(model, X, bs=256, return_feats=False):
    model.eval(); P, E = [], []
    for i in range(0, len(X), bs):
        x = to_gpu(X[i:i + bs])
        with torch.autocast('cuda', dtype=torch.float16):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
            logits = model.get_classifier()(f)
            logits = (logits + model.get_classifier()(model.forward_head(model.forward_features(x.flip(3)), pre_logits=True))) / 2  # flip TTA
        P.append(logits.float().cpu()); E.append(f.float().cpu())
    P = torch.cat(P).numpy()
    return (P, torch.cat(E).numpy()) if return_feats else P

def train_one(arch, split_col, seed, epochs=15, bs=64, lr=3e-4):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ham = man['ham10000']; X = ARR['ham10000']
    y = torch.tensor(ham['label7'].map(C2I).values)
    tr = np.where(ham[split_col] == 'train')[0]; va = np.where(ham[split_col] == 'val')[0]
    counts = np.bincount(y[tr].numpy(), minlength=7)
    w = torch.tensor((counts.sum() / (7 * counts)) ** 0.5, dtype=torch.float, device=DEV)   # sqrt class weights
    model = timm.create_model(MODELS[arch], pretrained=True, num_classes=7).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr); t0 = time.time()
        for i in range(0, len(perm) - bs + 1, bs):
            idx = np.sort(perm[i:i + bs])
            x = gpu_augment(to_gpu(X[idx])); yt = y[idx].to(DEV)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), yt, weight=w, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        pv = predict(model, X[np.sort(va)]).argmax(1)
        yv = y[np.sort(va)].numpy()
        bal = np.mean([(pv[yv == c] == c).mean() for c in range(7) if (yv == c).any()])
        if bal > best:
            best, best_state = bal, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'{arch} {split_col} s{seed} ep{ep+1}/{epochs} loss={loss.item():.3f} val_bal_acc={bal:.3f} ({time.time()-t0:.0f}s)')
    model.load_state_dict(best_state)
    return model, best

# ===== TRAIN Cell 4: Run training + save predictions/embeddings for every dataset =====
# Saves logits + penultimate embeddings for: HAM10000 (all images, with split labels) and each external set.
OUT = f'{PROJ}/models'
RUNS = [(a, s) for s in [42, 43, 44] for a in MODELS]          # main runs: 5 models x 3 seeds (clean split)
RUNS += [(a, 42) for a in ['resnet50', 'mobilenetv2', 'xception', 'efficientnet_b3']]  # leakage audit (leaky split)
LEAKY = len(RUNS) - 4

for k, (arch, seed) in enumerate(RUNS):
    split = f'leaky_s{seed}' if k >= LEAKY else f'split_s{seed}'
    tag = f'{arch}__{split}'
    if os.path.exists(f'{OUT}/{tag}/preds.npz'):
        print('done already:', tag); continue
    os.makedirs(f'{OUT}/{tag}', exist_ok=True)
    model, best = train_one(arch, split, seed)
    res = {}
    for n in ARR:
        logits, emb = predict(model, ARR[n], return_feats=True)
        res[f'{n}_logits'] = logits.astype(np.float32); res[f'{n}_emb'] = emb.astype(np.float16)
    np.savez_compressed(f'{OUT}/{tag}/preds.npz', **res)
    json.dump({'arch': arch, 'timm': MODELS[arch], 'split': split, 'seed': seed, 'best_val_bal_acc': float(best)},
              open(f'{OUT}/{tag}/info.json', 'w'))
    print('saved', tag, 'best val bal acc', round(best, 4))
    del model; torch.cuda.empty_cache()

# ===== TRAIN Cell 5: Foundation-model features (frozen) + linear probe =====
# DINOv2 ViT-B/14 (general foundation model). Linear probe = logistic regression on frozen features.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FM = {'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m'}
for name, tid in FM.items():
    tag = f'{name}__frozen'
    os.makedirs(f'{OUT}/{tag}', exist_ok=True)
    if not os.path.exists(f'{OUT}/{tag}/feats.npz'):
        fm = timm.create_model(tid, pretrained=True, num_classes=0, img_size=SIZE).to(DEV).eval()
        feats = {}
        with torch.no_grad():
            for n, X in ARR.items():
                fs = []
                for i in tqdm(range(0, len(X), 128), desc=f'{name} {n}'):
                    x = to_gpu(X[i:i + 128])
                    with torch.autocast('cuda', dtype=torch.float16):
                        fs.append(((fm(x) + fm(x.flip(3))) / 2).float().cpu())
                feats[n] = torch.cat(fs).numpy().astype(np.float32)
        np.savez_compressed(f'{OUT}/{tag}/feats.npz', **feats)
    feats = dict(np.load(f'{OUT}/{tag}/feats.npz'))
    ham = man['ham10000']; y = ham['label7'].map(C2I).values
    for seed in [42, 43, 44]:
        st = f'{OUT}/{name}__split_s{seed}'; os.makedirs(st, exist_ok=True)
        if os.path.exists(f'{st}/preds.npz'): continue
        tr = (ham[f'split_s{seed}'] == 'train').values
        sc = StandardScaler().fit(feats['ham10000'][tr])
        clf = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(sc.transform(feats['ham10000'][tr]), y[tr])
        res = {}
        for n in feats:
            res[f'{n}_logits'] = clf.decision_function(sc.transform(feats[n])).astype(np.float32)
            res[f'{n}_emb'] = feats[n].astype(np.float16)
        np.savez_compressed(f'{st}/preds.npz', **res)
        va = (ham[f'split_s{seed}'] == 'val').values
        pv = res['ham10000_logits'][va].argmax(1); yv = y[va]
        bal = np.mean([(pv[yv == c] == c).mean() for c in range(7)])
        json.dump({'arch': name, 'timm': tid, 'split': f'split_s{seed}', 'seed': seed, 'best_val_bal_acc': float(bal), 'probe': 'logreg'},
                  open(f'{st}/info.json', 'w'))
        print(name, seed, 'val bal acc', round(bal, 4))
