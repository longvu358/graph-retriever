import pandas as pd, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/mnt/user-data/uploads/graph-retriever/results_remote/'
d6=pd.concat(pd.read_csv(f) for f in glob.glob(R+'d6_tau_vs_baselines_test/*.csv'))
d9=pd.concat(pd.read_csv(f) for f in glob.glob(R+'d9_val/*.csv')); d9=d9[d9.split=='test']
x=pd.concat([d6,d9]).drop_duplicates(['city','variant','params','seed'])
x['lam']=x.params.str.extract(r'lam=([\d.]+)').astype(float)
m=x.groupby(['city','variant','lam'])[['exp_gini','ndcg20','recall20']].mean().reset_index()
m.to_csv('/home/claude/figs/curve_means.csv',index=False)
plt.rcParams.update({'font.size':15,'font.family':'DejaVu Sans'})
NAVY='#1F3B8C'
def fig(series,cities,out,w):
    f,axs=plt.subplots(1,len(cities),figsize=(w,4.6))
    axs=axs if len(cities)>1 else [axs]
    for ax,c in zip(axs,cities):
        for v,(lab,col,lw,z) in series.items():
            d=m[(m.city==c)&(m.variant==v)].sort_values('lam').reset_index(drop=True); d=d.iloc[:d.exp_gini.idxmin()+1]
            ax.plot(d.exp_gini,d.ndcg20,'-o',color=col,lw=lw,ms=4,label=lab,zorder=z)
            d0=d[d.lam==0]; ax.plot(d0.exp_gini,d0.ndcg20,'o',color=col,ms=10,zorder=z+1)
        ax.set_title(c.capitalize(),loc='left',fontweight='bold')
        ax.set_xlabel('Exposure Gini  (← more dispersed)'); ax.grid(alpha=.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    axs[0].set_ylabel('nDCG@20  (↑ more accurate)')
    h,l=axs[0].get_legend_handles_labels()
    f.legend(h,l,loc='lower center',ncol=len(series),frameon=False,bbox_to_anchor=(.5,-.02))
    f.tight_layout(rect=(0,.09,1,1)); f.savefig(out,dpi=200); plt.close(f)
S1={'v2_split_lq0.25':('Ours: v2 + split + τ',NAVY,3.2,10),
    'v2':('v2 (report version)','#7FA7E0',2,5),
    'gru4rec':('GRU4Rec','#2A9D8F',2,4),
    'dhcn':('DHCN','#8E6CC4',2,3),'gcegnn':('GCE-GNN','#E07A9A',2,3)}
fig(S1,['pisa','florence','rome'],'/home/claude/figs/main_tradeoff.png',15)
S2={'v2_lq0.25':('v2 + τ',NAVY,3,10),'v2_lq0.25_db0.1':('v2 + τ + β 0.1','#E9A23B',2.5,9),'v2_lq0.25_db0.3':('v2 + τ + β 0.3','#D1495B',2.5,8)}
fig(S2,['rome'],'/home/claude/figs/rome_beta.png',7)
print(m[m.lam==0].pivot_table(index='variant',columns='city',values=['ndcg20','exp_gini']).round(3).to_string())
