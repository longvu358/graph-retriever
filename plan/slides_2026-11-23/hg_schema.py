import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
plt.rcParams.update({'font.size':15,'font.family':'DejaVu Sans'})
NAVY='#1F3B8C'; ORANGE='#E07B39'; GREEN='#2A9D8F'
f,ax=plt.subplots(figsize=(7.2,3.9)); ax.set_xlim(0,10); ax.set_ylim(0.3,5.25); ax.axis('off')
# areas (dashed boxes)
for (x,y,w,h,lab) in [(0.3,0.5,4.4,3.7,'Area 1 (≤ 500 m)'),(5.3,0.5,4.4,3.7,'Area 2 (≤ 500 m)')]:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.4',fc='#F2F2F2',ec='#8C8C8C',ls='--',lw=1.8))
    ax.text(x+w-0.2,y+0.15,lab,ha='right',color='#555555',fontsize=12,fontweight='bold')
# theme edges
ax.add_patch(Ellipse((2.0,2.0),3.0,1.3,fc=NAVY,alpha=.16,ec=NAVY,lw=2))
ax.text(2.0,1.05,'churches @ area 1',ha='center',color=NAVY,fontsize=13,fontweight='bold')
ax.add_patch(Ellipse((7.6,2.0),3.0,1.3,fc=NAVY,alpha=.16,ec=NAVY,lw=2))
ax.text(7.6,1.05,'churches @ area 2',ha='center',color=NAVY,fontsize=13,fontweight='bold')
ax.add_patch(Ellipse((5.0,3.45),7.4,1.05,fc=ORANGE,alpha=.18,ec=ORANGE,lw=2))
ax.text(5.0,4.45,'theme: Renaissance palaces (small → kept whole)',ha='center',color=ORANGE,fontsize=13,fontweight='bold')
pts={'Church A':(1.3,2.15),'Church B':(2.7,2.15),'Church C':(6.9,2.15),'Church D':(8.3,2.15),'Palace P':(2.3,3.6),'Palace Q':(7.7,3.6)}
for n,(x,y) in pts.items():
    ax.plot(x,y,'o',ms=15,color='white',mec='#262626',mew=2,zorder=5)
    ax.text(x,y-0.42,n,ha='center',va='top',fontsize=12,zorder=6)
f.tight_layout(); f.savefig('/home/claude/figs/hg_schema.png',dpi=200)
