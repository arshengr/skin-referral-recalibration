"""Fig 4-5 from final summary numbers (results_final.md). Author: Arshad Saeed (Bay Atlantic University)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
sites=['HAM10000\n(internal)','BCN20000','MILK10k','PAD-UFES-20\n(smartphone)']
M={'SRC':([.048,.172,.139,.207],[.015,.096,.065,.076],[.653,.764,.829,.757]),
   'QTC':([.046,.247,.289,.288],[.015,.116,.141,.132],[.663,.665,.665,.665]),
   'QTC-LS':([.052,.251,.229,.297],[.021,.155,.148,.145],[.646,.666,.733,.655]),
   'WCP':([.041,.046,.076,.015],[.025,.040,.056,.016],[.669,.909,.891,.978]),
   'ORACLE-50':([.037,.026,.038,.042],[.031,.023,.025,.026],[.675,.968,.949,.950])}
C={'SRC':'#4a5568','QTC':'#dd6b20','QTC-LS':'#d69e2e','WCP':'#2b6cb0','ORACLE-50':'#2f855a'}
fig,ax=plt.subplots(figsize=(7.2,3.0)); w=0.16; x=np.arange(4)
for k,(m,(mu,sd,_)) in enumerate(M.items()):
    ax.bar(x+(k-2)*w,mu,w,yerr=sd,capsize=1.5,color=C[m],label=m,error_kw=dict(lw=0.6))
ax.axhline(0.05,color='#c53030',ls='--',lw=1); ax.text(3.55,0.056,'promised ≤ 5%',color='#c53030',fontsize=7,ha='right')
ax.set_xticks(x); ax.set_xticklabels(sites); ax.set_ylabel('Malignant miss rate'); ax.set_ylim(0,0.47)
ax.legend(ncol=5,fontsize=7,frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.0))
fig.savefig('figs/fig4_missrate_alpha05.png',dpi=300,bbox_inches='tight'); fig.savefig('figs/fig4_missrate_alpha05.pdf',bbox_inches='tight')
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),sharey=True)
for j,s in enumerate(range(1,4)):
    a=axs[j]
    for m,(mu,sd,ref) in M.items():
        a.scatter(ref[s],mu[s],s=40,color=C[m],label=m,zorder=3); 
    a.axhline(0.05,color='#c53030',ls='--',lw=0.8); a.set_title(sites[s].replace('\n',' '),fontsize=8)
    a.set_xlabel('Referral rate (workload)'); a.set_xlim(0.6,1.0)
axs[0].set_ylabel('Malignant miss rate'); axs[0].legend(fontsize=6.5,frameon=False,loc='upper right')
fig.tight_layout(); fig.savefig('figs/fig5_safety_vs_workload.png',dpi=300,bbox_inches='tight'); fig.savefig('figs/fig5_safety_vs_workload.pdf',bbox_inches='tight')
print('ok')
