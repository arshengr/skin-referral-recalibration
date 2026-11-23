"""Extra manuscript figures: Fig 2 data flow, Fig 3 method schematic, graphical abstract.
Author: Arshad Saeed (Bay Atlantic University)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
BLUE, GREY, RED, GREEN, INK = '#2b6cb0', '#4a5568', '#c53030', '#2f855a', '#1a202c'
FILL = {BLUE: '#ebf8ff', RED: '#fff5f5', GREEN: '#f0fff4', GREY: '#f7fafc'}

def box(ax, x, y, w, h, lines, c=GREY, bold_first=True, fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=0.8',
                                fc=FILL[c], ec=c, lw=1.0))
    n = len(lines); step = 2.9
    y0 = y + h / 2 + (n - 1) * step / 2
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y0 - i * step, l, ha='center', va='center', fontsize=fs,
                color=c if (i == 0 and bold_first) else INK,
                fontweight='bold' if (i == 0 and bold_first) else 'normal')

def arrow(ax, p, q, c=GREY, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=0.9, color=c, ls=ls))

def save(fig, name):
    fig.savefig(f'figs/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'figs/{name}.pdf', bbox_inches='tight')
    plt.close(fig)

# ---------------- Fig 2: data flow (PRISMA-style) ----------------
cols = [  # name, colour, downloaded, usable, unmapped, id-overlap, near-dup, final, role
    ('HAM10000', BLUE, '11,720', '11,719 dermoscopic', '0', 'n/a', 'n/a', '11,719', 'development site'),
    ('BCN20000', RED, '18,946', '18,946 dermoscopic', '1,470', '0', '65', '17,411', 'external site 1'),
    ('MILK10k', RED, '10,480', '5,240 dermoscopic', '89', '985', '17', '4,149', 'external site 2'),
    ('PAD-UFES-20', RED, '2,298', '2,298 smartphone', '0', '0', '15', '2,283', 'external site 3'),
]
fig, ax = plt.subplots(figsize=(8.6, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(-1, 72); ax.axis('off')
W, X0, GAP = 20.5, 2.5, 3.6
rows_y = [61, 49, 23, 8]          # downloaded, usable, (exclusions in middle), final
for i, (nm, c, d, u, um, ov, nd, fin, role) in enumerate(cols):
    x = X0 + i * (W + GAP)
    box(ax, x, rows_y[0], W, 8, [nm, f'{d} images downloaded'], c)
    box(ax, x, rows_y[1], W, 7, ['Usable modality', u], GREY)
    excl = [f'Unmapped label: {um}', f'Same ID as HAM: {ov}', f'Near-duplicate: {nd}']
    bx = x + 3.6
    ax.add_patch(FancyBboxPatch((bx, 33), x + W - bx, 11.5, boxstyle='round,pad=0.3,rounding_size=0.8',
                                fc='white', ec=GREY, lw=0.8, ls='--'))
    ax.text(bx + (x + W - bx) / 2, 42.3, 'Excluded', ha='center', fontsize=6.6, fontweight='bold', color=GREY)
    for k, e in enumerate(excl):
        ax.text(bx + (x + W - bx) / 2, 39.2 - k * 2.8, e, ha='center', fontsize=5.8, color=INK)
    box(ax, x, rows_y[2], W, 7.5, ['Analysis set', f'{fin} images'], GREEN)
    ax.text(x + W / 2, rows_y[2] - 2.2, role, ha='center', va='top', fontsize=6.4, style='italic', color=c)
    xl = x + 1.9
    ax.plot([xl, xl], [rows_y[1] - 0.4, rows_y[2] + 8.2], color=GREY, lw=0.9)
    arrow(ax, (xl, rows_y[2] + 8.6), (xl, rows_y[2] + 7.9))
    arrow(ax, (xl, 38.75), (bx - 0.4, 38.75))
    arrow(ax, (x + W / 2, rows_y[0] - 0.4), (x + W / 2, rows_y[1] + 7.4))
# HAM split box
box(ax, X0, 0, W, 14, ['Lesion-level split', '70 / 15 / 15 by lesion ID', '5 seeds; 0 shared lesions',
                       'leaky image split: 1,172', 'lesions cross subsets'], BLUE, fs=6.0)
arrow(ax, (X0 + W / 2, rows_y[2] - 4.6), (X0 + W / 2, 14.6), BLUE)
ax.text(X0 + 1 * (W + GAP), 10, 'External sites are never used for training.\n'
        'Their labels are hidden from every recalibration method\nand used only to score the final decisions.',
        fontsize=6.6, color=RED, va='center')
save(fig, 'fig2_data_flow')

# ---------------- Fig 3: method schematic ----------------
fig = plt.figure(figsize=(8.6, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.08)
a = fig.add_subplot(gs[0])
x = np.linspace(0, 1, 400)
src = norm.pdf(x, 0.72, 0.13); tgt = norm.pdf(x, 0.50, 0.16)
alpha = 0.05
tau_src = norm.ppf(alpha, 0.72, 0.13); tau_w = norm.ppf(alpha, 0.50, 0.16)
a.plot(x, src, color=BLUE, lw=1.4, label='development site (labelled calibration set)')
a.plot(x, tgt, color=RED, lw=1.4, label='new site (unknown, shifted)')
a.fill_between(x, 0, tgt, where=x < tau_src, color=RED, alpha=0.18, label='malignant lesions missed by source-only τ')
a.axvline(tau_src, color=BLUE, ls='--', lw=1); a.axvline(tau_w, color=GREEN, ls='--', lw=1)
miss_src = norm.cdf(tau_src, 0.50, 0.16)
a.text(tau_src + 0.01, 4.05, f'τ (source only)\nnew-site miss ≈ {miss_src:.0%}', color=BLUE, fontsize=6.6, va='top')
a.text(tau_w - 0.01, 4.05, 'τ (label-free\nrecalibrated)\nmiss ≈ 5%', color=GREEN, fontsize=6.6, va='top', ha='right')
a.set_xlabel('malignancy score s(x) of malignant lesions'); a.set_ylabel('density')
a.set_yticks([]); a.set_xlim(0, 1); a.set_ylim(0, 4.2)
a.spines[['top', 'right']].set_visible(False)
a.legend(loc='upper left', bbox_to_anchor=(0, -0.2), fontsize=6.4, frameon=False, ncol=2, columnspacing=1.2)
a.set_title('A  Why a fixed threshold breaks under site shift', loc='left', fontsize=8, fontweight='bold')

b = fig.add_subplot(gs[1]); b.set_xlim(0, 50); b.set_ylim(0, 50); b.axis('off')
b.set_title('B  Label-free recalibration and the referral rule', loc='left', fontsize=8, fontweight='bold')
box(b, 1, 37, 21, 10, ['Calibration set', 'scores + embeddings', '+ malignant labels'], BLUE, fs=6.2)
box(b, 27, 37, 22, 10, ['New-site images', 'scores + embeddings', 'no labels'], RED, fs=6.2)
box(b, 4, 18, 42, 14, ['Choose τ for the new site', 'WCP: reweight calibration set by p(new)/p(dev)',
                       'QTC: keep the development auto-clear rate', 'QTC-LS: + EM label-shift correction',
                       'refuse to certify if weights too extreme'], GREY, fs=6.0)
arrow(b, (11.5, 36.6), (18, 32.4), BLUE); arrow(b, (38, 36.6), (32, 32.4), RED)
box(b, 1, 1, 22, 11, ['s(x) < τ', 'auto-clear', '(no specialist review)'], GREEN, fs=6.2)
box(b, 27, 1, 22, 11, ['s(x) ≥ τ', 'refer to', 'dermatologist'], RED, fs=6.2)
arrow(b, (18, 17.6), (12, 12.4)); arrow(b, (32, 17.6), (38, 12.4))
save(fig, 'fig3_method_schematic')

# ---------------- Graphical abstract (wide, CBM: ≥1328×531 px) ----------------
fig, ax = plt.subplots(figsize=(13 / 2.54 * 1.6, 5 / 2.54 * 1.6)); ax.set_xlim(0, 130); ax.set_ylim(0, 54); ax.axis('off')
box(ax, 1, 14, 26, 24, ['Train once', 'HAM10000', '6 CNN / ViT models', 'lesion-level split'], BLUE, fs=7.4)
box(ax, 33, 14, 28, 24, ['Deploy to new sites', 'BCN20000 · MILK10k', 'PAD-UFES-20', 'labels hidden'], RED, fs=7.4)
box(ax, 67, 14, 30, 24, ['Label-free recalibration', 'weighted conformal (WCP)', 'quantile transfer (QTC)',
                         '+ label-shift correction'], GREY, fs=7.4)
box(ax, 103, 26, 26, 14, ['Fixed threshold', '17.3% of cancers missed', '(promise: ≤ 5%)'], RED, fs=7.0)
box(ax, 103, 8, 26, 14, ['Label-free WCP', '4.6% missed', '(93% referred)'], GREEN, fs=7.0)
arrow(ax, (27.6, 26), (32.4, 26), BLUE); arrow(ax, (61.6, 26), (66.4, 26), RED)
arrow(ax, (97.6, 22), (102.4, 15), GREEN); ax.plot([47, 47, 116], [38.4, 44, 44], color=RED, ls='--', lw=0.9); arrow(ax, (116, 44), (116, 40.4), RED, '--')
ax.text(80, 45, 'without recalibration', ha='center', fontsize=6.6, color=RED, style='italic')
ax.text(65, 51, 'Does a skin-cancer triage rule keep its safety promise at a new clinic?',
        ha='center', fontsize=9, fontweight='bold', color=INK)
ax.text(65, 3, 'auto-clear if malignancy score < τ, otherwise refer;  promise: malignant miss rate ≤ α',
        ha='center', fontsize=6.8, color=GREY, style='italic')
save(fig, 'graphical_abstract')
print('ok')
