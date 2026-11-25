"""Static research figures from completed offline analysis; no API calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cc4d_oracle_argon_20261125'


def main():
    a=json.loads((OUT/'analysis.json').read_text())
    d={(r['run'],r['metric']):r for r in a['metrics']}
    names=['oracle','k8','kall'];titles=['Oracle clip, single step','Full video, k=8','Full video, full report']
    colors=['#176B86','#65737E','#B0B8BE']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,(metric,title,n) in enumerate([('balanced_accuracy_standard','Original labels',951),('balanced_accuracy_exclude_order','Exclude all Order-tagged steps',753)]):
        rs=[d[name,metric] for name in names];v=np.array([r['value'] for r in rs])
        err=np.array([[r['value']-r['ci_low'] for r in rs],[r['ci_high']-r['value'] for r in rs]])
        axs[j].bar(range(3),v,color=colors,width=.6,yerr=err,capsize=4)
        for i,value in enumerate(v):axs[j].text(i,value+7,f'{value:.1f}%',ha='center')
        axs[j].axhline(50,color='#888888',ls='--',lw=.8,label='50% reference')
        axs[j].set(ylim=(0,100),xticks=range(3),xticklabels=['Oracle','k=8','Full report'],ylabel='Balanced accuracy (%)',xlabel='Video / question condition',title=f'{title} (n={n} steps)')
    fig.suptitle('CC4D Argon: oracle clips and full-video baselines',fontsize=13)
    caption='Source: saved Argon runs; paired by step. Error bars: 95% recording bootstrap, 64 recordings, 2,000 resamples.'
    if any('reconstructed' in s for s in a['baseline_sources'].values()):caption+=' Full-video boolean answers reconstructed from archived rows.'
    fig.text(.5,-.03,caption,ha='center',fontsize=8,wrap=True)
    fig.savefig(OUT/'balanced_accuracy.png',bbox_inches='tight');fig.savefig(OUT/'balanced_accuracy.pdf',bbox_inches='tight');plt.close(fig)
    groups=['done correctly','Order only','Order + other','no Order'];counts=[411,104,94,342]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    x=np.arange(4)
    for i,(name,label,color) in enumerate(zip(names,titles,colors)):
        rs=[d[name,'accuracy_group_'+g] for g in groups];v=np.array([r['value'] for r in rs])
        err=np.array([[r['value']-r['ci_low'] for r in rs],[r['ci_high']-r['value'] for r in rs]])
        ax.bar(x+(i-1)*.24,v,width=.24,color=color,label=label,yerr=err,capsize=2)
    ax.set(ylim=(0,100),xticks=x,xticklabels=[f'{g}\n(n={n})' for g,n in zip(groups,counts)],ylabel='Accuracy against original correctness labels (%)',xlabel='Ground-truth step group',title='Argon correctness accuracy by error group')
    ax.legend(frameon=False,loc='upper right',fontsize=8)
    fig.text(.5,-.035,'Source: same 951 paired steps. Error bars: 95% recording bootstrap (2,000 resamples). Order-only labels require sequence context.',ha='center',fontsize=8)
    fig.savefig(OUT/'accuracy_by_group.png',bbox_inches='tight');fig.savefig(OUT/'accuracy_by_group.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
