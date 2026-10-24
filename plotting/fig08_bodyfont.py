from pathlib import Path
import pymupdf,json
R=Path(__file__).resolve().parents[1]
source=pymupdf.open(R/'inputs/fig08_geometry.pdf');p=source[0]
out=pymupdf.open();dest=out.new_page(width=p.rect.width,height=258)
for clip,dy in [(pymupdf.Rect(0,0,p.rect.width,91),0),(pymupdf.Rect(0,110,p.rect.width,204),132)]:
 dest.show_pdf_page(pymupdf.Rect(0,dy,p.rect.width,dy+clip.height),source,0,clip=clip,keep_proportion=True)
from fonts import times_path
fontfile=times_path();font=pymupdf.Font(fontfile=fontfile);dest.insert_font(fontname='BodyTimes',fontfile=fontfile)
labels=[(56,105,['(a) SWAP between','q₃ and q₄']),(181,105,['(b) Routing cost after','SWAP(q₃, q₄)']),(56,240,['(c) SWAP between','q₁ and q₂']),(181,240,['(d) Routing cost after','SWAP(q₁, q₂)'])]
for cx,y,lines in labels:
 for j,text in enumerate(lines):
  width=font.text_length(text,fontsize=8.918595962467212)
  dest.insert_text((cx-width/2,y+j*11.5),text,fontname='BodyTimes',fontsize=8.918595962467212,color=(0,0,0))
asset=R/'assets/fig08_routing_progress_single_column.pdf';out.save(asset,garbage=4,deflate=True)
dest.get_pixmap(matrix=pymupdf.Matrix(300/72,300/72)).save(R/'assets/fig08_routing_progress_single_column.png')
(R/'metadata/fig08_bodyfont_manifest.json').write_text(json.dumps({'subcaption_font':'Times New Roman','subcaption_font_pt':8.918595962467212,'layout':'2x2 with two-line subcaptions','circuit_geometry_unchanged':True,'output_width_mm':86},indent=2),encoding='utf-8')
print('Figure 8 subcaptions: Times New Roman 8.97pt, matching Figure 8 caption.')
