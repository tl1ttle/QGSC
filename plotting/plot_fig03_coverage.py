"""Vector schematic redraw of the three CX coverage panels in the supplied Figure 3."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fonts import register_times
register_times()
from matplotlib.patches import Polygon
from figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/fig03_cx_gate_coverage_revised'
plt.rcParams.update({'font.family':'Times New Roman',
 'mathtext.fontset':'stix','font.size':5.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
fig=plt.figure(figsize=(3.3858267717,1.6535433071),facecolor='white')
# Projected schematic vertices follow the original panels. No experimental samples are inferred.
v={'O':(52,240),'A':(280,38),'B':(280,210),'C':(212,312),'D':(170.56,134.96),'E':(246,175),'F':(128,274.2)}
edges=[('O','A'),('A','B'),('B','C'),('C','O'),('D','F'),('D','E'),('A','E'),('E','C'),('E','B'),('F','B')]
axes=[]
for i in range(3):
 ax=fig.add_axes([.012+i*.333,.13,.31,.84]);ax.set_xlim(0,340);ax.set_ylim(350,0);ax.axis('off');axes.append(ax)
 if i==1:ax.add_patch(Polygon([v[k] for k in ['F','C','B']],closed=True,facecolor='#B4B2F3',edgecolor='none',zorder=1))
 if i==2:ax.add_patch(Polygon([v[k] for k in ['O','A','B','C']],closed=True,facecolor='#B4B2F3',edgecolor='none',zorder=1))
 for a,b in edges:
  ax.plot([v[a][0],v[b][0]],[v[a][1],v[b][1]],color='#444444',linewidth=.4,zorder=2)
 ax.annotate('',xy=(240,325),xytext=v['O'],arrowprops={'arrowstyle':'->','color':'#222222','lw':.65,'shrinkA':0,'shrinkB':0},zorder=3)
 ax.annotate('',xy=(52,18),xytext=v['O'],arrowprops={'arrowstyle':'->','color':'#222222','lw':.65,'shrinkA':0,'shrinkB':0},zorder=3)
 ax.annotate('',xy=(337,259),xytext=v['C'],arrowprops={'arrowstyle':'->','color':'#222222','lw':.65,'shrinkA':0,'shrinkB':0},zorder=3)
 for yy in [139,38]:ax.plot([49,55],[yy,yy],color='black',linewidth=.35)
 for xx,yy in [(90,257.1),(128,274.2),(166,291.3),(204,308.4)]:ax.plot([xx-1,xx+1],[yy+3,yy-3],color='black',linewidth=.35)
 for xx,yy in [(268.5,288),(325,264)]:ax.plot([xx-1,xx+1],[yy-3,yy+3],color='black',linewidth=.35)
 labels=[(24,38,r'$\pi/4$'),(73,14,r'$z$'),(46,263,r'$0$'),
         (126,298,r'$\pi/4$'),(253,331,r'$x$'),(324,236,r'$y$')]
 for xx,yy,label in labels:ax.text(xx,yy,label,ha='center',va='center',fontsize=5.6,color='black' if label==r'$\pi/4$' else 'black')
 # Tie the y-axis endpoint label to the final tick, rather than to the arrow tip.
 ax.annotate(r'$\pi/4$',xy=(325,264),xytext=(0,-6),textcoords='offset points',ha='center',va='top',fontsize=5.6,color='black')
 if i<2:ax.plot(*v['F'],marker='o',markersize=2.2,color='#3F36BD',zorder=4)
 fig.text(.167+i*.333,.045,f'({chr(97+i)}) {i+1} CX',ha='center',va='center',fontsize=9.96)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=['a','b','c'],row_groups=[['a','b','c']],
 json_out=str(OUT)+'.alignment.json',overlay_svg=str(OUT)+'.alignment.svg',
 tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(str(OUT)+'.pdf');fig.savefig(str(OUT)+'.svg')
fig.savefig(str(OUT)+'.png',dpi=600)
fig.savefig(str(OUT)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
report={'source_kind':'schematic geometry defined in this script',
 'dimensions_mm':[86,42],'minimum_font_pt':5.6,'subcaption_font':'Times New Roman','subcaption_native_font_pt':9.96,'subcaption_printed_font_pt':9.96,
 'method':'Schematic vector redraw retaining the original three-panel sequence, axis labels, CX point, two-CX triangular region, three-CX chamber shading and projected internal edges. No empirical data inferred.',
 'geometry':v,'edges':edges,'outputs':'Vector PDF/SVG; 600 dpi PNG/TIFF.'}
(ROOT/'metadata/fig03_coverage_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Figure3 exported as vector 86x42mm, 5.6pt labels.')
