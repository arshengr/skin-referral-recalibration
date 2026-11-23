import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
fig, ax = plt.subplots(figsize=(8.6, 4.2)); ax.set_xlim(0,100); ax.set_ylim(0,56); ax.axis('off')
BLUE, GREY, RED, GREEN = '#2b6cb0', '#4a5568', '#c53030', '#2f855a'
def box(x,y,w,h,title,lines,c=GREY,fill='#f7fafc'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fill,ec=c,lw=1.1))
    ax.text(x+w/2,y+h-2.3,title,ha='center',va='top',fontweight='bold',color=c,fontsize=7.9)
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-6.2-i*3.3,l,ha='center',va='top',fontsize=6.9,color='#1a202c')
def arrow(x1,y1,x2,y2,c=GREY,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=1,color=c,ls=ls))
# band labels
ax.text(1,54.5,'A  Development site',fontweight='bold',fontsize=8.5,color=BLUE)
ax.text(1,26.5,'B  New site (no labels)',fontweight='bold',fontsize=8.5,color=RED)
# Row A
box(1,32,18,19,'HAM10000',['11.7k images','Austria, Australia','split by lesion ID','70 / 15 / 15'],BLUE,'#ebf8ff')
box(23,32,22,19,'Train classifiers',['ResNet-50, MobileNetV2,','Xception, EfficientNet-B3,','ConvNeXt-T, DINOv2 probe','3 seeds each'],BLUE,'#ebf8ff')
box(49,32,24,19,'Calibrate referral rule',['temperature scaling','malignancy score s(x)','conformal threshold τ','promise: miss ≤ α'],BLUE,'#ebf8ff')
box(77,32,22,19,'Side analysis: leakage',['same models trained on','image-level split','→ how much accuracy','is inflated?'],GREY)
arrow(19.6,41.5,22.4,41.5,BLUE); arrow(45.6,41.5,48.4,41.5,BLUE); 
# Row B
box(1,3,18,20,'External sites',['BCN20000 (Spain)','MILK10k','PAD-UFES-20','(smartphone, Brazil)'],RED,'#fff5f5')
box(23,3,22,20,'Unlabeled images',['same frozen model','scores s(x) and','embeddings only','(labels hidden)'],RED,'#fff5f5')
box(49,3,24,20,'Label-free recalibration',['SRC: source-only (baseline)','WCP: weighted conformal','QTC / QTC + label shift','certify or refuse'],RED,'#fff5f5')
box(77,3,22,20,'Decide + evaluate',['auto-clear if s(x) < τ','otherwise refer','labels used only here:','miss rate, referral rate'],GREEN,'#f0fff4')
arrow(19.6,13,22.4,13,RED); arrow(45.6,13,48.4,13,RED); arrow(73.6,13,76.4,13,GREEN)
arrow(61,31.4,61,23.6,BLUE)   # threshold flows down
ax.text(62,27.5,'τ, calibration scores\n& embeddings',fontsize=6.8,color=BLUE,va='center')
arrow(34,31.4,34,23.6,BLUE)
fig.savefig('figs/fig1_study_design.png',dpi=300,bbox_inches='tight'); fig.savefig('figs/fig1_study_design.pdf',bbox_inches='tight')
