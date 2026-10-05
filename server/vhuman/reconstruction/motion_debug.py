"""Dedicated landmark, anatomy and parsing movies for native GNM motion."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from ..rig.expression_catalog import Movie
from .observations import sha256
from .reference import Camera


def render(directory, original, out):
    import cv2
    directory,original,out=map(Path,(directory,original,out));out.mkdir(parents=True,exist_ok=True)
    track=json.loads((directory/'motion.json').read_text())
    if sha256(track['clip'])!=track['clip_sha256']:raise ValueError('motion source clip changed')
    motion=np.load(directory/'motion.npz',allow_pickle=False)
    labels=np.load(original/'face_parsing.npz',allow_pickle=False)['labels']
    palette_file=original/'parsing_palette.json'
    if palette_file.is_file():
        palette_doc=json.loads(palette_file.read_text())
        palette=np.array([row['rgb'][::-1] for row in sorted(palette_doc.values(),key=lambda row:row['index'])],np.uint8)
    else:
        rng=np.random.default_rng(19);palette=rng.integers(60,240,(19,3),dtype=np.uint8);palette[0]=0
    w,h=track['size'];camera=Camera.from_dict(track['camera']);fps=track['fps']
    names=('generated','landmarks','gnm_mesh','face_parsing','four_panel')
    writers={name:Movie(out/(name+'.mp4'),w*(2 if name=='four_panel' else 1),h*(2 if name=='four_panel' else 1),fps) for name in names}
    cap=cv2.VideoCapture(track['clip']);tri=motion['triangles'];count=0
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            i=count
            if i>=track['frames']:raise ValueError('clip longer than fitted motion')
            landmark=np.uint8(image*.3)
            target=motion['observed_landmarks'][i];predicted=motion['projected_landmarks'][i]
            for observed,fitted,weight in zip(target,predicted,motion['weights'][i]):
                cv2.circle(landmark,tuple(np.rint(observed).astype(int)),1,(70,255,100) if weight>0 else (100,100,100),-1)
                if weight>0:cv2.circle(landmark,tuple(np.rint(fitted).astype(int)),1,(90,90,255),-1)
            vertices=motion['vertices'][i];xy,depth=camera.project(vertices)
            points=(vertices-camera.origin)@camera.rotation.T
            normals=np.cross(points[tri[:,1]]-points[tri[:,0]],points[tri[:,2]]-points[tri[:,0]])
            magnitude=np.linalg.norm(normals,axis=-1)
            shade=np.maximum(normals@np.array([.3,.4,1])/np.maximum(magnitude,1e-12),0)
            colors=np.uint8((.25+.75*shade[:,None])*np.array([195,205,220]))
            mesh=np.full((h,w,3),22,np.uint8)
            order=np.argsort(-depth[tri].mean(1));screen=np.rint(xy).astype(np.int32)
            for t in order:
                if normals[t,2]<=0 or (depth[tri[t]]<.02).any():continue
                polygon=screen[tri[t]]
                if (polygon.max(0)<0).any() or (polygon.min(0)>[w,h]).any():continue
                cv2.fillConvexPoly(mesh,polygon,tuple(map(int,colors[t])),lineType=cv2.LINE_AA)
            parsed=palette[labels[i]]
            panels=dict(generated=image,landmarks=landmark,gnm_mesh=mesh,face_parsing=parsed)
            for name,panel in panels.items():
                cv2.rectangle(panel,(0,0),(w,42),(12,12,12),-1)
                cv2.putText(panel,f'{original.name.upper()} | {name}',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                if name=='gnm_mesh':
                    cv2.putText(panel,f'fixed identity | error {track["per_frame_error_ipd"][i]:.3f} IPD',(8,h-18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                writers[name].frame(panel)
            writers['four_panel'].frame(np.vstack((np.hstack((panels['generated'],landmark)),np.hstack((mesh,parsed)))))
            count+=1
    finally:
        cap.release()
        for writer in writers.values():writer.close()
    if count!=track['frames']:raise ValueError('clip shorter than fitted motion')
    result=dict(schema='vhuman.native_motion_debug.v1',frames=count,identity_frozen=True,
        motion_sha256=sha256(directory/'motion.npz'),mesh_renderer='perspective depth-sorted diagnostic triangles; not a physical renderer',
        movies={name:dict(sha256=sha256(out/(name+'.mp4')),size=[w*(2 if name=='four_panel' else 1),h*(2 if name=='four_panel' else 1)]) for name in names})
    (out/'debug.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',required=True);parser.add_argument('--catalog',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();motion,catalog,out=map(Path,(args.motion,args.catalog,args.out));results={}
    expressions=('neutral','happy','sad','angry','fear','surprise','disgust')
    for name in expressions:
        directory=motion/name
        if not (directory/'motion.json').is_file():directory=directory/'native_motion'
        results[name]=render(directory,catalog/name,out/name)
        print(f'{name}: debug movies complete',flush=True)
    for movie in ('generated','landmarks','gnm_mesh','face_parsing','four_panel'):
        listing=out/(movie+'_concat.txt')
        # Paths are fixed safe expression names and relative to this manifest.
        listing.write_text(''.join(f"file '{name}/{movie}.mp4'\n" for name in expressions))
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','concat','-safe','1','-i',str(listing),'-c','copy',str(out/(movie+'_all.mp4'))],check=True)
    (out/'catalog.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
