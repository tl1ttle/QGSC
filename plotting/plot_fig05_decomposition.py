"""Figure 5: editable vector decomposition template matching the QFT3 gate palette."""
from pathlib import Path
import json,re,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/fig05_decomposition_template_revised'
qft=ET.parse(ROOT/'figures/fig10_panel_c_v2.drawio')
colors={}
for cell in qft.iter('mxCell'):
 value=cell.get('value','');match=re.search(r'fillColor=(#[0-9A-Fa-f]{6})',cell.get('style',''))
 if match:
  if value=='U<sub>3</sub>':colors['U3']=match[1]
  if value=='G(θ)':colors['G']=match[1]
assert colors=={'U3':'#E5EFF7','G':'#FBEADB'},colors
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'mathtext.fontset':'stix','font.size':7.8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
fig=plt.figure(figsize=(3.3858267717,1.0629921260),facecolor='white')
ax=fig.add_axes([.02,.08,.96,.84]);ax.set_xlim(0,100);ax.set_ylim(0,28);ax.axis('off')
wire_segments=[(15,18.8),(27.2,30),(42,45.8),(54.2,72),(84,86.8),(95.2,99)]
for y in [8,20]:
 for x0,x1 in wire_segments:ax.plot([x0,x1],[y,y],color='#4A4A4A',linewidth=.65,zorder=1)
ax.text(5.8,14,r'$U=$',ha='center',va='center',fontsize=9)
gates=[]
for x in [23,50,91]:
 for y in [8,20]:
  ax.add_patch(Rectangle((x-4.2,y-3.5),8.4,7,facecolor=colors['U3'],edgecolor='#4A4A4A',linewidth=.55,zorder=2))
  ax.text(x,y,r'$U_3$',ha='center',va='center',fontsize=7.8,zorder=3)
  gates.append({'type':'U3','center':[x,y],'width':8.4,'height':7,'label':'U<sub>3</sub>'})
for x in [36,78]:
 ax.add_patch(Rectangle((x-6,4.4),12,19.2,facecolor=colors['G'],edgecolor='#4A4A4A',linewidth=.55,zorder=2))
 ax.text(x,14,r'$G(\theta)$',ha='center',va='center',fontsize=7.8,zorder=3)
 gates.append({'type':'G','center':[x,14],'width':12,'height':19.2,'label':'G(θ)'})
ax.text(63,14,r'$\cdots$',ha='center',va='center',fontsize=10)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=[ax],panel_ids=['a'],json_out=str(OUT)+'.alignment.json',
 overlay_svg=str(OUT)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(str(OUT)+'.pdf');fig.savefig(str(OUT)+'.svg')
fig.savefig(str(OUT)+'.png',dpi=600)
fig.savefig(str(OUT)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
# Editable Drawio counterpart, with the same gate geometry and semantic palette.
mx=ET.Element('mxfile',host='Electron');dia=ET.SubElement(mx,'diagram',name='Figure 5 decomposition template')
model=ET.SubElement(dia,'mxGraphModel',page='0',background='#ffffff');root=ET.SubElement(model,'root')
ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
for y in [8,20]:
 for j,(x0,x1) in enumerate(wire_segments):
  cell=ET.SubElement(root,'mxCell',id=f'wire{y}_{j}',edge='1',parent='1',style='endArrow=none;strokeColor=#4A4A4A;strokeWidth=1.2;')
  geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
  ET.SubElement(geo,'mxPoint',x=str(x0*6),y=str((28-y)*6),attrib={'as':'sourcePoint'})
  ET.SubElement(geo,'mxPoint',x=str(x1*6),y=str((28-y)*6),attrib={'as':'targetPoint'})
for i,g in enumerate(gates):
 x,y=g['center'];w,h=g['width'],g['height']
 cell=ET.SubElement(root,'mxCell',id=f'gate{i}',parent='1',vertex='1',value=g['label'],
 style=f'rounded=0;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;fillColor={colors[g["type"]]};strokeColor=#4A4A4A;fontColor=#000000;fontFamily=Times New Roman;fontSize=18;')
 ET.SubElement(cell,'mxGeometry',x=str((x-w/2)*6),y=str((28-y-h/2)*6),width=str(w*6),height=str(h*6),attrib={'as':'geometry'})
for label,x,w in [('U =',0,12),('⋯',57,12)]:
 cell=ET.SubElement(root,'mxCell',id='label'+str(x),parent='1',vertex='1',value=label,style='text;html=1;align=center;verticalAlign=middle;fontFamily=Times New Roman;fontSize=22;strokeColor=none;fillColor=none;')
 ET.SubElement(cell,'mxGeometry',x=str(x*6),y='54',width=str(w*6),height='60',attrib={'as':'geometry'})
ET.indent(mx)
ET.ElementTree(mx).write(ROOT/'figures/fig05_decomposition_template_revised.drawio',encoding='utf-8',xml_declaration=True)
report={'dimensions_mm':[86,27],'palette_source':'figures/fig10_panel_c_v2.drawio','palette':colors,
 'topology':'Two wires, three local U3 layers, two shown G(theta) gates and an ellipsis for the repeated sequence.',
 'label_font_pt':7.8,'minimum_subscript_font_pt':5.46,'shown_gate_counts':{'U3':6,'G':2},'text_color':'black','all_graphics':'vector PDF/SVG','gates':gates}
(ROOT/'metadata/fig05_decomposition_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Figure5:86x27mm; six U3 and two G(theta) gates; QFT3 palette',colors)
