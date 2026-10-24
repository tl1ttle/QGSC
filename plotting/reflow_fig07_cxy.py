"""Compact existing vector Figure 7; retain circuit connectivity and shared layer semantics."""
from pathlib import Path
import pymupdf,json
R=Path(__file__).resolve().parents[1]
src=pymupdf.open(R/'inputs/fig07_geometry.pdf')
out=pymupdf.open();page=out.new_page(width=245.5,height=158+168*180/293+21+4)
placements=[]
for clip,y,sub,centers in [(pymupdf.Rect(40,0,305,187),0,'(a) Logical circuit',[109.6,191.9,269.9]),(pymupdf.Rect(397,0,690,168),158,'(b) Weighted directed acyclic graph',[439.5,532.6,622.1])]:
 scale=180/clip.width;h=clip.height*scale;x=(245.5-180)/2
 target=pymupdf.Rect(x,y,x+180,y+h)
 page.show_pdf_page(target,src,0,clip=clip)
 for label,cx in zip(['L','F','E'],centers):
  tx=x+(cx-clip.x0)*scale
  page.insert_text((tx-2.75,y+h+9),label,fontname='tiro',fontsize=9)
 baseline=y+h+21
 width=pymupdf.get_text_length(sub,fontname='tiro',fontsize=9)
 page.insert_text(((245.5-width)/2,baseline),sub,fontname='tiro',fontsize=9)
 placements.append(dict(source_clip=list(clip),target=list(target),font_size=9,caption_baseline=baseline))
# Canvas is sized before drawing, so the PDF origin and top qubit labels remain intact.
out.save(R/'assets/fig07_circuit_dag_compact_cxy.pdf',garbage=4,deflate=True)
page.get_pixmap(matrix=pymupdf.Matrix(2,2)).save(R/'assets/fig07_circuit_dag_compact_cxy.png')
(R/'metadata/fig07_cxy_layout.json').write_text(json.dumps(placements,indent=2),encoding='utf-8')
