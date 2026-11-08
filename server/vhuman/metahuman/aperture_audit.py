"""Compare opaque eye-aperture masks under the existing calibrated cameras.

These masks isolate mesh intersections; they intentionally exclude translucent
shells, tear layers and lashes, and do not measure their final visible shading.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_fill_holes,binary_erosion,distance_transform_edt
from .images import read_exr
from .compare_render import box_down


def aperture_mask(image):
    import cv2
    raw=np.asarray(image)[...,:3].mean(-1)>.5
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype(np.uint8),8)
    if count<2:raise ValueError('no white aperture component')
    selected=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    return binary_fill_holes(labels==selected)


def shape_stats(mask):
    y,x=np.nonzero(mask)
    if not len(x):raise ValueError('empty aperture mask')
    return dict(area_px=int(len(x)),centroid_px=[float(x.mean()+.5),float(y.mean()+.5)],bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],width_px=int(x.max()-x.min()+1),height_px=int(y.max()-y.min()+1))


def compare_masks(reference,candidate):
    if reference.shape!=candidate.shape:raise ValueError('mask dimensions differ')
    a,b=shape_stats(reference),shape_stats(candidate)
    ea=reference&~binary_erosion(reference);eb=candidate&~binary_erosion(candidate)
    distances=np.concatenate([distance_transform_edt(~ea)[eb],distance_transform_edt(~eb)[ea]])
    return dict(iou=float((reference&candidate).sum()/(reference|candidate).sum()),boundary_mean_px=float(distances.mean()),boundary_p95_px=float(np.percentile(distances,95)),centroid_offset_px=(np.asarray(b['centroid_px'])-a['centroid_px']).tolist(),reference=a,candidate=b)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ue-root',type=Path,required=True);ap.add_argument('--cycles-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ids',nargs='+',default=['mh_000020','mh_000100']);ap.add_argument('--cameras',nargs='+',default=['eye_left','eye_right']);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(a.out/'mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ids=a.ids;cams=a.cameras;fig,axes=plt.subplots(len(ids),len(cams),squeeze=False,figsize=(6*len(cams),4*len(ids)),layout='constrained');records={}
    for i,sid in enumerate(ids):
        for j,cam in enumerate(cams):
            ref=aperture_mask(box_down(read_exr(a.ue_root/sid/(cam+'__geometry_mask.exr')),512));cy=aperture_mask(read_exr(a.cycles_root/sid/(cam+'.exr')))
            row=compare_masks(ref,cy)
            paths=dict(reference_mask=a.ue_root/sid/(cam+'__geometry_mask.exr'),candidate_mask=a.cycles_root/sid/(cam+'.exr'),reference_beauty=a.ue_root/sid/(cam+'__reference.exr'))
            row['sources']={key:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key,path in paths.items()};records[sid+'/'+cam]=row
            rgb=box_down(read_exr(paths['reference_beauty'])[...,:3],512);k=.6/np.percentile(rgb,95);rgb=np.maximum(rgb*k,0);rgb=np.clip(np.where(rgb<=.0031308,rgb*12.92,1.055*rgb**(1/2.4)-.055),0,1)
            ax=axes[i,j];ax.imshow(rgb);ax.contour(ref,levels=[.5],colors=['cyan'],linewidths=1);ax.contour(cy,levels=[.5],colors=['magenta'],linewidths=1)
            ra,rb=row['reference']['bbox'],row['candidate']['bbox'];x0=max(0,min(ra[0],rb[0])-20);x1=min(512,max(ra[2],rb[2])+20);y0=max(0,min(ra[1],rb[1])-20);y1=min(512,max(ra[3],rb[3])+20)
            ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_title(f"{sid} {cam}: IoU {row['iou']:.3f}, boundary {row['boundary_mean_px']:.2f}px");ax.axis('off');print(sid,cam,row,flush=True)
    fig.legend(handles=[Line2D([0],[0],color='cyan',label='UE mesh aperture'),Line2D([0],[0],color='magenta',label='vhuman mesh aperture')],loc='outside lower center',ncol=2)
    fig.savefig(a.out/'aperture_contours.png',dpi=160);plt.close(fig)
    (a.out/'report.json').write_text(json.dumps(dict(resolution=512,method='largest white component, interior holes filled; fixed cameras; no alignment fit',caveats='Opaque intersections exclude translucent shells, tear edges and lashes. UE replacement removes original material vertex/depth offsets; Cycles retains skin displacement. This is not a final photometric aperture measurement.',views=records),indent=2))


if __name__=='__main__':main()
