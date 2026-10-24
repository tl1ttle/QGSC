"""Figure 10: vector scatter from the 60 individually matched circuits."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment
from fonts import register_times
register_times()
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/fig10_excess_amplitude_clear'
rows=list(csv.DictReader((ROOT/'data/alltoall_60.csv').open(encoding='utf-8')))
summary=json.loads((ROOT/'data/alltoall_summary.json').read_text(encoding='utf-8'))
x=[float(r['excess_proportion_percent']) for r in rows]
y=[float(r['reduction_percent']) for r in rows]
mean=summary['ugs_vs_reqisc_percent']
# Match manuscript Times text; preserve signed reductions and all circuit observations.
plt.rcParams.update({'font.family':'Times New Roman','font.size':9.96,
 'axes.labelsize':9.96,'xtick.labelsize':9.96,'ytick.labelsize':9.96,
 'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
fig=plt.figure(figsize=(3.3858267717,2.3622047244),facecolor='white')
ax=fig.add_axes([.22,.22,.74,.73])
point_colors=['#9467BD' if value < 0 else '#009E73' for value in y]
ax.scatter(x,y,s=17,marker='o',facecolors=point_colors,edgecolors='white',linewidths=.3,alpha=1,zorder=3)
ax.set_xlim(-2,102);ax.set_ylim(-45,85)
ax.set_xticks([0,20,40,60,80,100]);ax.set_yticks([-40,-20,0,20,40,60,80])
ax.set_xlabel('Excess-amplitude gates (%)',labelpad=3)
ax.set_ylabel('Time reduction vs ReQISC (%)',labelpad=3)
ax.tick_params(width=.6,length=2.5,pad=2)
ax.spines[['right','top']].set_visible(False)
ax.axhline(0,color='#B1B1B1',linewidth=.5,zorder=1)
ax.axhline(mean,color='#FF0000',linestyle=(0,(4,3)),linewidth=.7,zorder=2)
ax.text(4,80,f'Mean: {mean:.2f}%',ha='left',va='top',fontsize=9.96,color='#000000')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=[ax],panel_ids=['a'],json_out=str(OUT)+'.alignment.json',
 overlay_svg=str(OUT)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(str(OUT)+'.pdf')
fig.savefig(str(OUT)+'.svg')
fig.savefig(str(OUT)+'.png',dpi=600)
fig.savefig(str(OUT)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
report={'source':'assets/fig10_circuit_source_60.csv','dimensions_mm':[86,60],
 'minimum_font_pt':9.96,'font_family':'Times New Roman','mean_reduction_percent':mean,
 'circuit_count':60,'point_count':60,'family_aggregation':False,'negative_reductions_preserved':True,
 'data_integrity':'No raster or fitted curve. Every point is one individually matched circuit; no smoothing or invented interpolation.',
 'mean_basis':summary['mean_basis'],'point_basis':summary['point_basis'],
 'conclusion':'Excess-amplitude prevalence and paired ReQISC-relative time reductions, with overall circuit-weighted benefit.',
 'archetype':'single-panel quantitative scatter','export':'Editable vector PDF/SVG; PNG/TIFF 600 dpi.'}
(ROOT/'metadata/fig10_clear_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Figure 10: 60 individual circuits; mean',f'{mean:.2f}%')
