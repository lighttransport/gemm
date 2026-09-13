"""Create a narrow RGB pencil and explicit mask for the full CPU smoke run.

This limits object complexity, not pipeline resolution or diffusion steps. It
exercises all 1024-cascade stages and the same 4096 PBR export as the GPU cases.
"""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
rgba=Image.new('RGBA',(384,512),(0,0,0,0));d=ImageDraw.Draw(rgba)
# Flat facets give DINO a recognizable isolated object with a narrow silhouette.
d.rounded_rectangle((177,40,207,74),radius=5,fill=(218,103,130,255))
d.rectangle((177,69,207,87),fill=(170,176,179,255))
d.rectangle((180,72,183,85),fill=(222,224,225,255))
d.rectangle((177,87,207,416),fill=(219,164,30,255))
d.rectangle((183,87,198,416),fill=(249,201,43,255))
d.rectangle((199,87,203,416),fill=(183,127,17,255))
d.polygon([(177,416),(207,416),(192,468)],fill=(187,140,92,255))
d.polygon([(183,416),(200,416),(192,454)],fill=(226,184,132,255))
d.polygon([(187,451),(197,451),(192,468)],fill=(41,43,44,255))
rgba.convert('RGB').save(a.output_dir/'pencil-rgb.png')
rgba.getchannel('A').save(a.output_dir/'pencil-mask.png')
rgba.save(a.output_dir/'pencil-rgba.png')
print(a.output_dir/'pencil-rgb.png')
