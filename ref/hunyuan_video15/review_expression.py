"""Export every eye/mouth crop and sampled full frames for expression review.

This creates review artifacts, not an automatic identity or expression score.
Eye and mouth regions are explicit pixel boxes for the selected portrait.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

def box(value):
    coordinates=tuple(int(v) for v in value.split(','))
    if len(coordinates)!=4 or coordinates[0]<0 or coordinates[1]<0 or coordinates[2]<=coordinates[0] or coordinates[3]<=coordinates[1]:
        raise argparse.ArgumentTypeError('use left,top,right,bottom pixel coordinates')
    return coordinates

def sheet(frames,path,region=None,columns=9):
    w,h=(region[2]-region[0],region[3]-region[1]) if region else (240,424)
    result=Image.new('RGB',(columns*w,((len(frames)+columns-1)//columns)*h),(32,32,32))
    draw=ImageDraw.Draw(result)
    for i,(number,frame) in enumerate(frames):
        with Image.open(frame) as image:
            value=image.crop(region) if region else image.resize((w,h),Image.Resampling.LANCZOS)
            x,y=(i%columns)*w,(i//columns)*h
            result.paste(value,(x,y))
            draw.rectangle((x,y,x+30,y+14),fill=(0,0,0))
            draw.text((x+2,y),str(number),fill=(255,255,255))
    result.save(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--eye-box',type=box,required=True)
    ap.add_argument('--mouth-box',type=box)
    ap.add_argument('--expected-frames',type=int,choices=(81,121),default=81)
    ap.add_argument('--ffmpeg',default='ffmpeg')
    args=ap.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    decoded=out/'frames'
    decoded.mkdir()
    subprocess.run([args.ffmpeg,'-v','error','-i',str(args.video.resolve()),
                    '-vsync','0',str(decoded/'frame_%05d.png')],check=True)
    frames=sorted(decoded.glob('frame_*.png'))
    if len(frames)!=args.expected_frames:
        raise ValueError('unexpected decoded frame count')
    for frame in frames:
        with Image.open(frame) as image:
            if image.size!=(480,848):
                raise ValueError('requires the portrait video bucket')
    for region in (args.eye_box,args.mouth_box):
        if region and (region[2]>480 or region[3]>848):
            raise ValueError('crop lies outside the video')
    numbered=list(enumerate(frames))
    sheet(numbered,out/'full_frames.png',columns=9)
    sheet(numbered,out/'eyes.png',args.eye_box)
    if args.mouth_box:
        sheet(numbered,out/'mouth.png',args.mouth_box)
    sample=[(round(i*(len(frames)-1)/8),frames[round(i*(len(frames)-1)/8)]) for i in range(9)]
    sheet(sample,out/'contact_sheet.png',columns=3)
    report={'scope':'visual_review_artifacts','automatic_quality_score':None,
            'video_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
            'decoded_frames':len(frames),'size':[480,848],
            'eye_box':args.eye_box,'mouth_box':args.mouth_box,
            'full_frame_sheet':'full_frames.png',
            'sampled_frames':[number for number,_ in sample]}
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
