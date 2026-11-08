"""Compare signed Cycles shading normals with Windows UE SCS_Normal captures.

Angles include residual geometry/correspondence differences. They diagnose
normal transport; they are not a substitute for beauty-image evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .images import read_exr
from .samples import MH_TO_VH, V1
from .compare_render import box_down


def load_normal_capture(path):
    import OpenEXR
    with OpenEXR.File(str(path)) as exr:
        fields={}
        for part in exr.parts:
            prefix=part.header.get('name','')
            for key,channel in part.channels.items():
                name=prefix if key in ('RGB','RGBA') and prefix else key
                fields[name]=np.asarray(channel.pixels,np.float32).copy()
    result={}
    for name in ('shader_normal','geometric_normal','skin_mask','uv','Position'):
        matches=[value for key,value in fields.items() if key==name or key.endswith('.'+name)]
        if len(matches)!=1:raise ValueError(f'expected one {name} field')
        result[name]=matches[0][...,:3]
    shape=result['skin_mask'].shape
    if any(x.shape!=shape or not np.isfinite(x).all() for x in result.values()):
        raise ValueError('invalid normal capture fields')
    return result


def normalized(vectors):
    vectors=np.asarray(vectors,np.float32)
    return vectors/np.maximum(np.linalg.norm(vectors,axis=-1,keepdims=True),1e-8)


def angular_error(a,b):
    if a.shape!=b.shape or a.shape[-1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matching finite normal arrays required')
    return np.degrees(np.arccos(np.clip((normalized(a)*normalized(b)).sum(-1),-1,1)))


def summarize(values,mask):
    x=np.asarray(values)[mask]
    if not len(x):return dict(pixels=0,mean=None,median=None,p95=None)
    return dict(pixels=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ue-root',type=Path,required=True)
    ap.add_argument('--cycles-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--ids',default='mh_000020,mh_000100')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.out/'mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ..face_parsing import FaceParser
    from ..reconstruction import ab_metrics as am
    parser=FaceParser(threads=8)
    ids=args.ids.split(',');modes=('baseline','measured_only','regional');records={}
    fig,axes=plt.subplots(len(ids)*2,4,figsize=(14,len(ids)*7),squeeze=False,layout='constrained')
    mappable=None
    for row,sid in enumerate(ids):
        ue=read_exr(args.ue_root/sid/'eye_left__normal.exr')[...,:3]
        ue=normalized(box_down(ue,512)@MH_TO_VH.T)
        ref=box_down(read_exr(V1/'samples'/sid/'renders/eye_left__vhstudio.exr')[...,:3],512)
        k=.6/max(float(np.percentile(ref,95)),1e-6)
        display=np.uint8(np.clip(np.where(ref*k<=.0031308,ref*k*12.92,1.055*np.maximum(ref*k,0)**(1/2.4)-.055),0,1)*255+.5)
        skin=am.label_mask(parser,display,['skin','nose'])
        eye=am.drop_specks(am.label_mask(parser,display,['left_eye','right_eye','left_brow','right_brow']))
        _,_,box=am.bbox_crop(display,eye,margin=.45)
        if box is None:raise ValueError('reference eye crop unavailable')
        x0,y0,x1,y1=box;roi=np.zeros(skin.shape,bool);roi[y0:y1,x0:x1]=True
        captures={mode:load_normal_capture(args.cycles_root/sid/(mode+'.exr')) for mode in modes}
        common=skin.copy()
        for fields in captures.values():
            common &= (fields['skin_mask'][...,0]>.99)&(np.linalg.norm(fields['shader_normal'],axis=-1)>.2)
        if common.sum()<500:raise ValueError('insufficient common skin coverage')
        rec=dict(coordinates='UE world xyz permuted to vhuman x/y/z by MH_TO_VH',box=list(map(int,box)),
                 caveat='includes geometry/correspondence differences; all modes use common skin mask',modes={})
        vectors=[ue]+[normalized(captures[mode]['shader_normal']) for mode in modes]
        for col,(name,n) in enumerate(zip(('UE',)+modes,vectors)):
            top,bottom=axes[row*2,col],axes[row*2+1,col]
            top.imshow(np.where(common[...,None],n*.5+.5,.03));top.set_title(sid+' '+name);top.axis('off')
            if name=='UE':
                bottom.text(.05,.5,f'Common skin: {common.sum()} pixels\nDashed box: eye crop\nAngles in degrees',transform=bottom.transAxes);bottom.axis('off');continue
            fields=captures[name];error=angular_error(ue,n)
            rec['modes'][name]=dict(frame_skin=summarize(error,common),periocular_skin=summarize(error,common&roi),
                                    geometric=summarize(angular_error(ue,fields['geometric_normal']),common),
                                    position_difference_m=summarize(np.linalg.norm(fields['Position']-captures['baseline']['Position'],axis=-1),common))
            mappable=bottom.imshow(np.where(common,error,np.nan),vmin=0,vmax=60,cmap='magma')
            bottom.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='cyan',linestyle='--',linewidth=.7));bottom.axis('off')
            bottom.set_title(f"mean {rec['modes'][name]['frame_skin']['mean']:.2f}° / eye {rec['modes'][name]['periocular_skin']['mean']:.2f}°")
        records[sid]=rec
        print(sid,json.dumps(rec['modes']),flush=True)
    if mappable is not None:fig.colorbar(mappable,ax=axes.ravel().tolist(),shrink=.5,label='Angular error (degrees)')
    fig.savefig(args.out/'world_normals.png',dpi=160);plt.close(fig)
    (args.out/'report.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
