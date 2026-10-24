"""Grouped dual-axis comparison: retain 30 circuits and all existing metrics."""
from pathlib import Path
import csv,json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fonts import register_times
register_times()
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment

def fidelity_forward(y):
 y=np.asarray(y)
 return np.where(y<=.7,(y-.4)*(.4/.3),
        np.where(y<=.8,.4+(y-.7)*4,
        np.where(y<=.9,.8+(y-.8)*8,1.6+(y-.9)*30)))

def fidelity_inverse(z):
 z=np.asarray(z)
 return np.where(z<=.4,.4+z*(.3/.4),
        np.where(z<=.8,.7+(z-.4)/4,
        np.where(z<=1.6,.8+(z-.8)/8,.9+(z-1.6)/30)))

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/restricted_topology_30.csv'
OUT=ROOT/'assets/fig13_restricted_topology_dual_axis'
rows=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
n=len(rows)
assert n==30 and len({r['circuit'] for r in rows})==30
assert not ({r['circuit'] for r in rows}&{'adder_n4','qft_16','qft_20','qft_n18'})
time_methods=['SQiSW','B','UGS']
fid_methods=['SQiSW','UGS']
colors={'SQiSW':'#AFD9CB','iSWAP':'#D3C4DE','B':'#CFD9E7','UGS':'#E6B6D1','SU4':'#AEBFD7'}
line_colors={'SQiSW':'#238B75','iSWAP':'#9273AB','B':'#6D82A0','UGS':'#B84787'}
display={'SQiSW':'SQiSW','iSWAP':'iSWAP','B':'B gate','UGS':'UGS','SU4':'SU(4)'}
hatches={'SQiSW':'','B':'','UGS':'///'}
markers={'SQiSW':'o','UGS':'s'}
offsets={m:(j-(len(time_methods)-1)/2)*.24 for j,m in enumerate(time_methods)}

def stripe_segments(x0,x1,low,high,dy,step):
 for phase in np.arange(low-abs(dy),high+abs(dy)+step,step):
  ta,tb=sorted([(low-phase)/dy,(high-phase)/dy])
  ta=max(0,ta);tb=min(1,tb)
  if tb>ta:yield ([x0+ta*(x1-x0),x0+tb*(x1-x0)],[phase+ta*dy,phase+tb*dy])

class BarLegendHandler(HandlerBase):
 def create_artists(self,legend,orig,xdescent,ydescent,width,height,fontsize,trans):
  rect=Rectangle((-xdescent,-ydescent),width,height,facecolor=orig.get_facecolor(),
                 edgecolor='#51545A',linewidth=.45,transform=trans)
  artists=[rect]
  if getattr(orig,'stripe',''):
   for xx,yy in stripe_segments(-xdescent,width-xdescent,-ydescent,height-ydescent,height,height*.6):
    artists.append(Line2D(xx,yy,color='#51545A',linewidth=.3,transform=trans))
  return artists

def short_label(name):
 if name.startswith('bmw_quark_cardinality_n'):return name.replace('bmw_quark_cardinality_n','bmw-card')
 if name.startswith('bmw_quark_copula_n'):return name.replace('bmw_quark_copula_n','bmw-cop')
 if name.startswith('basis_change_n'):return name.replace('basis_change_n','b-change')
 if name.startswith('basis_test_n'):return name.replace('basis_test_n','b-test')
 if name.startswith('basis_trotter_n'):return name.replace('basis_trotter_n','trotter')
 if name=='error_correctiond3_n5':return 'err-corr5'
 if name.startswith('qpeinexact_n'):return name.replace('qpeinexact_n','qpe-in')
 if name.startswith('qpeexact_n'):return name.replace('qpeexact_n','qpe-ex')
 return name.replace('_n','').replace('_','-')

# Merge only the trailing qubit-size suffix; keep distinct algorithm families separate.
source_rows=rows
groups={}
for row in source_rows:
 family=re.sub(r'_n?\d+$','',row['circuit'])
 groups.setdefault(family,[]).append(row)
rows=[];group_mapping={}
metric_columns=[k for k in source_rows[0] if '_time_' in k or '_HF_' in k]
for family,members in groups.items():
 sizes=sorted({int(row['qubits']) for row in members})
 label=(short_label(family+'_n0')[:-1]+f'({sizes[0]}-{sizes[-1]})'
        if len(members)>1 else short_label(members[0]['circuit']))
 row={'label':label,'count':len(members),'qubits':','.join(map(str,sizes)),
      'members':';'.join(m['circuit'] for m in members)}
 row.update({k:float(np.mean([float(m[k]) for m in members])) for k in metric_columns})
 rows.append(row)
 group_mapping[label]={'members':[m['circuit'] for m in members],'qubits':sizes,'count':len(members)}
n=len(rows);labels=[row['label'] for row in rows]
assert len(set(labels))==n
assert sum(row['count'] for row in rows)==30
with (ROOT/'data/restricted_topology_family_means.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
x=np.arange(n)
plt.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman'],
 'font.size':8.5,'axes.labelsize':9.0,'axes.labelweight':'normal',
 'xtick.labelsize':8.0,'ytick.labelsize':8.5,'axes.linewidth':.7,
 'hatch.linewidth':.4,'legend.frameon':True,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','axes.facecolor':'white','savefig.facecolor':'white'})
fig=plt.figure(figsize=(7.0866141732,4.9),facecolor='white')
axes=[];twins=[];series=[]
for index,(t,bottom,name) in enumerate([('chain',.68,'1D Chain'),('grid',.24,'2D Grid')]):
 ax=fig.add_axes([.14,bottom,.72,.23],label=f'{t}_time')
 right=ax.twinx();right.set_label(f'{t}_fidelity');right.patch.set_visible(False)
 axes.append(ax);twins.append(right)
 for m in time_methods:
  v=np.array([float(r[f'{t}_time_{m}']) for r in rows]);assert np.isfinite(v).all()
  mean=float(np.average(v,weights=[row['count'] for row in rows]))
  displayed=np.maximum(v,0)
  ax.bar(x+offsets[m],displayed,width=.215,color=colors[m],edgecolor='#51545A',
         linewidth=.42,zorder=2)
  if hatches[m]:
   for center,value in zip(x+offsets[m],displayed):
    for xx,yy in stripe_segments(center-.1075,center+.1075,min(0,value),max(0,value),2.8,4):
     ax.plot(xx,yy,color='#51545A',linewidth=.25,zorder=3)
  ax.axhline(max(mean,0),color=line_colors[m],linewidth=.65,linestyle=(0,(1.2,2.2)),zorder=4)
  series.append({'topology':t,'metric':'time_reduction_percent','method':m,'values':v.tolist(),
                 'display_values':displayed.tolist(),'mean':mean,'display_mean':max(mean,0),
                 'display_transform':'max(original,0); mean calculated before clipping'})
 fv={}
 for m in fid_methods:
  v=np.array([float(r[f'{t}_HF_{m}']) for r in rows]);assert ((v>=0)&(v<=1)).all()
  fv[m]=v
  mean=float(np.average(v,weights=[row['count'] for row in rows]))
  series.append({'topology':t,'metric':'hellinger_fidelity','method':m,'values':v.tolist(),'mean':mean})
  phase=0 if m=='SQiSW' else 3.5
  right.axhline(mean,color=line_colors[m],linewidth=.9,linestyle=(phase,(4,3)),zorder=5)
 for m in fid_methods:
  visible=fv[m]>=.4  # Display mask only; all original values remain in series and means.
  right.plot((x+offsets[m])[visible],fv[m][visible],linestyle='none',marker=markers[m],markersize=2.8,
             markerfacecolor=line_colors[m],markeredgecolor=line_colors[m],markeredgewidth=.45,zorder=6)
 ax.set_xlim(-.65,n-.35);right.set_xlim(ax.get_xlim())
 ax.set_ylim(0,75);ax.set_yticks([0,25,50,75])
 # Compress the lower fidelity range; expand 0.9--1.0 with the same transform for points and means.
 right.set_yscale('function',functions=(fidelity_forward,fidelity_inverse))
 right.set_ylim(.4,1.02);right.set_yticks([.4,.7,.8,.9,1.0])
 right.set_yticks([.92,.94,.96,.98],minor=True)
 right.tick_params(axis="y",which="minor",length=1.7,width=.45)
 ax.set_ylabel('Time reduction (%)',labelpad=3)
 right.set_ylabel('Fidelity',labelpad=3)
 ax.set_xticks(x,labels,rotation=60,ha='right',va='center',rotation_mode='anchor')
 for label in ax.get_xticklabels()+ax.get_yticklabels()+right.get_yticklabels():label.set_fontweight('normal')
 ax.tick_params(axis='x',width=.55,length=2.5,pad=3)
 ax.tick_params(axis='y',width=.55,length=2.5,pad=2)
 right.tick_params(axis='y',width=.55,length=2.5,pad=2)
 ax.set_axisbelow(True);ax.grid(axis='y',color='#E2E2E2',linewidth=.35);right.grid(False)
 ax.axhline(0,color='#55585E',linewidth=.65,zorder=3)
 # A thin full frame follows the visual reference; each spine is drawn once.
 for spine in ax.spines.values():spine.set_color('#45484D')
 ax.spines['right'].set_visible(False)
 right.spines['left'].set_visible(False);right.spines['top'].set_visible(False)
 right.spines['bottom'].set_visible(False);right.spines['right'].set_color('#45484D')
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer()
 label_bottom=min(label.get_window_extent(renderer).y0 for label in ax.get_xticklabels())/fig.bbox.height
 caption_y=label_bottom-(4.0+4.5)/(fig.get_figheight()*72)
 fig.text(.50,caption_y,f'({chr(97+index)}) {name}',ha='center',va='center',fontsize=9.0,fontfamily='Times New Roman')
bh=[]
for m in time_methods:
 h=Patch(facecolor=colors[m],edgecolor='#51545A',linewidth=.45,label=display[m])
 h.stripe=hatches[m];bh.append(h)
lh=[Line2D([],[],color=line_colors[m],linewidth=.9,linestyle=(0,(4,3)),
    marker=markers[m],markersize=3.4,markerfacecolor=line_colors[m],markeredgewidth=.45,label=display[m]) for m in fid_methods]
for h in lh:h.set_label(h.get_label()+' (F)')
mean_handles=[Line2D([],[],color=line_colors[m],linewidth=.65,linestyle=(0,(1.2,2.2)),label=display[m]+', Time Avg.') for m in time_methods]
legend_handles=[mean_handles[0],bh[0],mean_handles[1],bh[1],mean_handles[2],bh[2],lh[0],lh[1]]
leg=fig.legend(handles=legend_handles,loc='upper center',bbox_to_anchor=(.5,.995),
 ncol=4,prop={'family':'Times New Roman','size':10.06333432,'weight':'normal'},handlelength=2.0,
 handletextpad=.35,columnspacing=.8,labelspacing=.15,borderaxespad=0,borderpad=.25,
 fancybox=False,edgecolor='#555555',handler_map={Patch:BarLegendHandler()})
leg.get_frame().set_linewidth(.35);leg.get_frame().set_facecolor('white');leg.get_frame().set_alpha(1)
fig.canvas.draw()
# Axis-aligned PDF boxes overlap for diagonal parallel labels. Check actual
# perpendicular spacing against unrotated text heights before exporting.
renderer=fig.canvas.get_renderer();tick_labels=axes[1].get_xticklabels()
heights=[];anchors=[]
for label in tick_labels:
 angle=label.get_rotation();label.set_rotation(0)
 heights.append(label.get_window_extent(renderer).height)
 label.set_rotation(angle)
 anchors.append(label.get_transform().transform(label.get_position()))
normal=np.array([np.sin(np.deg2rad(60)),np.cos(np.deg2rad(60))])
projected=np.array(anchors)@normal
clearances=np.diff(projected)-(np.array(heights[:-1])+np.array(heights[1:]))/2
assert np.min(clearances)>0, 'Tilted tick labels overlap'
rotated_label_qa={'method':'Separate parallel text rectangles along their common normal, using measured unrotated renderer heights.',
 'angle_deg':60,'minimum_clearance_pt':float(np.min(clearances)*72/fig.dpi),
 'label_count':len(tick_labels),'verdict':'PASS',
 'note':'Axis-aligned text boxes overlap for rotated text; this oriented check establishes nonoverlap.'}
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=['a','b'],column_groups=[['a','b']],
 json_out=str(OUT)+'.alignment.json',overlay_svg=str(OUT)+'.alignment.svg',
 tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
for a,b in zip(axes,twins):assert np.allclose(a.get_position().bounds,b.get_position().bounds)
from matplotlib.transforms import Bbox
# Crop only the unused bottom margin below the (b) label; preserve physical width and fonts.
renderer=fig.canvas.get_renderer()
bottom_inches=max(0,fig.texts[-1].get_window_extent(renderer).y0/fig.dpi-2/72)
export_bbox=Bbox.from_extents(0,bottom_inches,fig.get_figwidth(),fig.get_figheight())
fig.savefig(str(OUT)+'.pdf',bbox_inches=export_bbox,pad_inches=0)
fig.savefig(str(OUT)+'.svg',bbox_inches=export_bbox,pad_inches=0)
fig.savefig(str(OUT)+'.png',dpi=300,bbox_inches=export_bbox,pad_inches=0)
fig.savefig(str(OUT)+'.tiff',dpi=600,bbox_inches=export_bbox,pad_inches=0,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
report={'rows':n,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'source_rows':30,'exclusion':{'rule':'Previously requested exclusions retained','circuits':['adder_n4','qft_16','qft_20','qft_n18'],'before':34,'after':30},
 'plotted_values':sum(len(s['values']) for s in series),'mean_lines':len(series),'series':series,
 'label_mapping':group_mapping,'rotated_label_qa':rotated_label_qa,
 'aggregation':'Arithmetic means within each circuit family, before display clipping; horizontal means weighted by family size to preserve original 30-circuit means.',
 'contract':{'dimensions_mm':[180,124.46],'minimum_font_pt':5.8,
  'axes':{'left':'Time reduction relative to CX (%), 0 to 75; negative bars displayed as zero','right':'Hellinger fidelity, 0.4 to 1.02; major ticks 0.4, 0.7, 0.8, 0.9, 1.0; minor ticks 0.92, 0.94, 0.96, 0.98 on a piecewise scale: axis positions for 0.4,0.7,0.8,0.9,1.0 are 0,0.4,0.8,1.6,4.6; lower range compressed and 0.9--1.0 expanded'},
  'display_transform':'Negative time reductions clipped to zero for display only; all means use original signed data. Fidelity values below 0.4 are outside the displayed range but remain in mean calculations.',
  'connections':'No circuit-to-circuit or within-group connections. Each bar/point is a family arithmetic mean. Dotted horizontal lines are mean time reductions; dashed horizontal lines are mean fidelities. No final Mean group.',
  'methods':'Time: SQiSW, B, UGS; fidelity: SQiSW, UGS. iSWAP and SU(4) removed from this comparison at author request; unused method columns are not included in the packaged dataset.',
  'symbols':'SQiSW and B time: solid; UGS time: diagonal lines. SQiSW fidelity: filled circle; UGS fidelity: filled square. Legend symbols match plotted marks.',
  'uncertainty':'No replicate uncertainties supplied; no error bars invented.'}}
(ROOT/'metadata/restricted_dual_axis_plot_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'Exported: 30 original circuits, {n} circuit families, {len(series)*n} plotted metrics, {len(series)} original circuit-weighted mean lines.');print(labels)
