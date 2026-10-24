# 2026-10-15：按作者要求，图1四个英文分区标签改为黄底黑字。
"""Regenerate Figure 1 in the updated PISGNC.drawio style (Qiskit 2.4).

The circuit operations, palette, separators and conditional frame follow the
user's embedded SVG. Text is editable and set to 8.0 pt (7 pt for auxiliary labels).
The four bold English headings with yellow backgrounds remain separate Drawio objects.
"""
from pathlib import Path
import base64, re, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'
NAME='fig01_instruction_primitives_drawio_matched'
plt.rcParams.update({'font.family':'Arial','font.size':8.0,'svg.fonttype':'none','pdf.fonttype':42})
qc=QuantumCircuit(4,2)
qc.x(0);qc.x(1);qc.h(2);qc.h(3)
qc.t(0);qc.tdg(1);qc.t(2);qc.tdg(3)
qc.cx(0,1);qc.cx(2,3);qc.barrier()
qc.reset(3);qc.barrier();qc.measure(0,0);qc.measure(1,1);qc.barrier()
with qc.if_test((qc.cregs[0],1)):
    qc.x(2)
assert qc.count_ops()['cx']==2 and qc.count_ops()['measure']==2
fig=qc.draw(output='mpl',initial_state=True,fold=-1,scale=1,
    style={'name':'iqp','fontsize':8.0,'subfontsize':7,'backgroundcolor':'#FFFFFF'})
fig.set_size_inches(6.5,2.8)
ax=fig.axes[0]
# Keep the source stroke/gate proportions when reducing the physical canvas;
# otherwise Qiskit's two classical wires merge into one heavy-looking line.
for line in ax.lines:
    line.set_linewidth(line.get_linewidth()*.65)
for patch in ax.patches:
    patch.set_linewidth(patch.get_linewidth()*.65)
# Qiskit enlarges wire labels by 1.25 by default. Match gate/caption scale.
for text in ax.texts:
    content=text.get_text()
    text.set_fontfamily('Arial')
    text.set_fontsize(7 if content in ['2','0','1','0x1'] else 8.0)
    if content.strip()=='If':
        text.set_fontsize(6.5)
        text.set_position((text.get_position()[0],-1.835))
ax.set_ylim(-4.6,1.04)
# Delete only wire segments already hidden behind Qiskit's opaque gate faces.
# Visible geometry is unchanged; vector geometry audits then see the same
# unobscured wires that readers see in the rendered figure.
opaque=[]
for patch in ax.patches:
    if type(patch).__name__=='Rectangle' and patch.get_facecolor()[3]==1:
        x,y,w,h=patch.get_bbox().bounds
        if 0<w<2 and 0<h<2:opaque.append((x,y,w,h))
for line in ax.lines:
    xs,ys=line.get_data()
    if len(xs)!=2 or len(ys)!=2 or ys[0]!=ys[1]:continue
    lo,hi=sorted(xs);y=ys[0]
    masks=sorted((max(lo,x),min(hi,x+w)) for x,yy,w,h in opaque if yy<y<yy+h and x<hi and x+w>lo)
    if not masks:continue
    cursor=lo;xx=[];yy=[]
    for a,b in masks:
        if a>cursor:xx.extend([cursor,a,np.nan]);yy.extend([y,y,np.nan])
        cursor=max(cursor,b)
    if cursor<hi:xx.extend([cursor,hi]);yy.extend([y,y])
    line.set_data(xx,yy)
# Keep an SVG containing only the regenerated circuit for the Drawio image node.
fig.savefig(OUT/(NAME+'_circuit.svg'),bbox_inches='tight',pad_inches=.025)
circuit_tight=fig.get_tightbbox(fig.canvas.get_renderer())

# Headings follow the same four sections as the source Drawio. Their centers are
# obtained from Qiskit gate and separator positions, so they remain aligned.
gate_x=next(t.get_position()[0] for t in ax.texts if t.get_text()=='T')
base=gate_x-1
headings=[('Quantum Gates',base+1.0),('Reset',base+4.0),
          ('Measurement',base+6.5),('Classical Control',base+9.3)]
artists=[ax.text(x,.77,label,ha='center',va='center',fontsize=8.0,
                fontfamily='Arial',fontweight='bold',color='#000000',
                bbox={'facecolor':'#FFD700','edgecolor':'none','pad':1.6}) for label,x in headings]
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
fig.savefig(OUT/(NAME+'.pdf'),bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/(NAME+'.svg'),bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/(NAME+'.png'),dpi=300,bbox_inches='tight',pad_inches=.025)
fig.savefig(OUT/(NAME+'.tiff'),dpi=600,bbox_inches='tight',pad_inches=.025)

# Editable Drawio source: the circuit is a regenerated SVG image and the
# headings are independent English text cells, as in PISGNC.drawio.
svg=(OUT/(NAME+'_circuit.svg')).read_text(encoding='utf8')
svgr=ET.fromstring(svg)
w=float(svgr.get('width').replace('pt',''));h=float(svgr.get('height').replace('pt',''))
mx=ET.Element('mxfile',host='Electron')
dia=ET.SubElement(mx,'diagram',name='Figure 1 - Qiskit and English labels')
model=ET.SubElement(dia,'mxGraphModel',page='0',background='#ffffff')
rr=ET.SubElement(model,'root');ET.SubElement(rr,'mxCell',id='0');ET.SubElement(rr,'mxCell',id='1',parent='0')
cell=ET.SubElement(rr,'mxCell',id='qiskit-circuit',parent='1',vertex='1',
    style='shape=image;aspect=fixed;imageAspect=0;image=data:image/svg+xml,'+base64.b64encode(svg.encode()).decode()+';')
ET.SubElement(cell,'mxGeometry',x='0',y='0',width=str(w),height=str(h),attrib={'as':'geometry'})
# Compute exact centers and preserve native font sizes in the standalone source.
# The unadorned SVG includes the same axes and tight-bounds as the combined image.
fig.canvas.draw()
for i,artist in enumerate(artists):
    box=artist.get_bbox_patch().get_window_extent(renderer)
    x=(box.x0/fig.dpi-circuit_tight.x0+.025)*72
    y=(circuit_tight.y1+.025-box.y1/fig.dpi)*72
    cell=ET.SubElement(rr,'mxCell',id=f'english-heading-{i+1}',parent='1',vertex='1',value=artist.get_text(),
        style='text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=#FFD700;align=center;verticalAlign=middle;spacing=0;fontFamily=Arial;fontStyle=1;fontSize=8.0;fontColor=#000000;')
    ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(box.width/fig.dpi*72),height=str(box.height/fig.dpi*72),attrib={'as':'geometry'})
ET.indent(mx)
ET.ElementTree(mx).write(ROOT/'figures/fig01_instruction_primitives_revised.drawio',encoding='utf8',xml_declaration=True)
plt.close(fig)
print('Generated Figure 1: source palette, four English headings, 8.0 pt text.')
