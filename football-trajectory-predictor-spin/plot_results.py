"""Plot the example's goal-plane densities and export PDF, SVG and PNG files."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent


def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    summaries=json.loads((ROOT/'results/posterior_summary.json').read_text())
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    labels=['Gaussian 1 cm / wide prior','Gaussian 1 cm / strong prior',
            'Rounding ±0.5 cm / wide prior','Rounding ±0.5 cm / strong prior']
    fig,ax=plt.subplots(figsize=(7.09,4.9));fig.subplots_adjust(left=.11,right=.96,bottom=.15,top=.76)
    legend=[]
    for s,color,label in zip(summaries,colors,labels):
        grid=np.load(ROOT/f"results/{s['noise']}_{s['prior']}_density.npz")
        ax.contour(grid['Y'],grid['Z'],grid['density'],levels=[float(grid['level'])],colors=[color],linewidths=1.5)
        ax.scatter(*s['mode_yz'],s=24,c=color,marker='o',zorder=4)
        legend.append(Line2D([0],[0],color=color,lw=1.5,marker='o',markersize=4,label=label))
    # Draw only the visible part of the rectangular goal to avoid off-axis PDF paths.
    ax.plot([.9,3.66,3.66],[2.44,2.44,.45],color='#777777',lw=1.3,linestyle='--')
    rows=list(csv.DictReader((ROOT/'results/point_estimates.csv').open(encoding='utf-8-sig')))
    for row in rows[:3]: ax.scatter(float(row['yg']),float(row['zg']),marker='x',c='#333333',s=30,zorder=5)
    legend.append(Line2D([0],[0],color='#333333',lw=0,marker='x',label='Unregularized baseline fits'))
    legend.append(Line2D([0],[0],color='#777777',linestyle='--',label='Geometric goal boundary'))
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.52,.985),ncol=2,frameon=False,
               columnspacing=1.5,handlelength=2.0,fontsize=8)
    ax.set(xlabel='Lateral goal-plane position y (m)',ylabel='Height z (m)',xlim=(.9,4.45),ylim=(.45,3.45))
    ax.set_xticks([1,2,3,4]);ax.set_yticks([.5,1,1.5,2,2.5,3]);ax.set_aspect('equal',adjustable='box')
    fig.text(.11,.035,'95% posterior HDR contours; dots: goal-plane KDE modes. Right side of goal shown.',fontsize=8)
    fig.savefig(ROOT/'results/goal_plane.pdf')
    fig.savefig(ROOT/'results/goal_plane.svg')
    fig.savefig(ROOT/'results/goal_plane.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':main()
