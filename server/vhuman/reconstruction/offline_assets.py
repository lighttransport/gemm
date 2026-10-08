"""Prepare portable metric meshes, optical eyes and separate accessory priors."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .reference import Camera


def projected_plane(camera, pixels, z):
    rays=camera.rays(np.asarray(pixels))
    if (abs(rays[:,2])<1e-6).any():raise ValueError('accessory ray parallel to face plane')
    return camera.origin+rays*((z-camera.origin[2])/rays[:,2])[:,None]


def curved_cap(camera, contour, z, rings=12, rear_depth=.20):
    """Closed ray-calibrated cap; unseen depth is a bounded geometric prior.

    Front rings preserve portrait UVs. The shared silhouette ring joins a
    separate, untextured rear dome, so source badges never stretch onto sides.
    """
    contour=np.asarray(contour,float)
    # Intersect the interior half-planes to find a star-shaped polygon kernel.
    # A plain vertex mean can sit below a concave cap rim and invert the fan.
    following=np.roll(contour,-1,axis=0)
    signed=(contour[:,0]*following[:,1]-contour[:,1]*following[:,0]).sum()
    if signed<0:contour=contour[::-1]
    lo=contour.min(0);hi=contour.max(0)
    kernel=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]],float)
    for a,b in zip(contour,np.roll(contour,-1,axis=0)):
        edge=b-a;clipped=[]
        for x,y in zip(kernel,np.roll(kernel,-1,axis=0)):
            dx=edge[0]*(x-a)[1]-edge[1]*(x-a)[0]
            dy=edge[0]*(y-a)[1]-edge[1]*(y-a)[0]
            if dx>=-1e-9:clipped.append(x)
            if (dx>=0)!=(dy>=0):clipped.append(x+(y-x)*dx/(dx-dy))
        kernel=np.asarray(clipped).reshape(-1,2)
        if not len(kernel):raise ValueError('cap outline is not star-shaped; simplify its parsing contour')
    centre=kernel.mean(0);n=len(contour)
    vertices=[];faces=[];front_count=0
    for rear in (False,True):
        start=len(vertices)
        # Only the front owns the boundary; rear indices reuse that seam.
        for ring in range(rings):
            # A quarter-ellipse retains width near the rim while extending
            # over the scalp. A shrinking linear fan left the temples exposed.
            angle=ring/rings*np.pi/2
            r=np.cos(angle) if rear else 1-ring/rings
            pixels=centre+(contour-centre)*r
            depth=z-rear_depth*np.sin(angle) if rear else z+.018*np.sqrt(1-r*r)
            points=projected_plane(camera,pixels,depth)
            if rear and ring==0:continue
            vertices.extend(points)
        centre_index=len(vertices)
        vertices.extend(projected_plane(camera,centre[None],z+(-rear_depth if rear else .018)))
        def indices(ring):
            if rear and ring==0:return np.arange(n)
            return np.arange(n)+(start+(ring-1)*n if rear else start+ring*n)
        for ring in range(rings-1):
            a=indices(ring);b=indices(ring+1)
            for i in range(n):
                j=(i+1)%n
                faces.extend([[a[i],b[i],b[j]],[a[i],b[j],a[j]]])
        a=indices(rings-1)
        faces.extend([[a[i],centre_index,a[(i+1)%n]] for i in range(n)])
        if not rear:front_count=len(faces)
        else:
            faces[front_count:]=[f[::-1] for f in faces[front_count:]]
    vertices=np.asarray(vertices);faces=np.asarray(faces,int)
    triangles=vertices[faces]
    volume=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6
    if volume<0:faces=faces[:,::-1]
    return vertices,faces,front_count


def glasses_temple(boundary, scalp, clearance=.003):
    """Head-space arm following the outer scalp envelope; depth is a prior."""
    boundary=np.asarray(boundary,float);scalp=np.asarray(scalp,float)
    side=1 if boundary[:,0].mean()>0 else -1
    hinge=boundary[np.argmax(side*boundary[:,0])].copy()
    depths=np.linspace(hinge[2],hinge[2]-.12,24)
    path=np.tile(hinge,(len(depths),1));path[:,2]=depths
    for i,z in enumerate(depths[1:],1):
        nearby=(abs(scalp[:,1]-hinge[1])<.008)&(abs(scalp[:,2]-z)<.008)&(side*scalp[:,0]>0)
        if nearby.any():
            outer=(side*scalp[nearby,0]).max()+clearance
            path[i,0]=side*max(side*hinge[0],outer)
        else:path[i,0]=path[i-1,0]
    # A short downward tip behind the ear, explicitly inferred from one image.
    angle=np.linspace(0,np.pi/2,9)[1:]
    hook=np.stack((np.zeros_like(angle),-.009*(1-np.cos(angle)),-.009*np.sin(angle)),-1)
    path=np.concatenate((path,path[-1]+hook))
    return path


def short_scalp_prior(labels,confidence,eye_y):
    """Require substantial unoccluded scalp evidence before a short-hair prior."""
    y,_=np.nonzero((np.asarray(labels)==17)&(np.asarray(confidence)>.4))
    return bool(len(y)>=256 and (np.asarray(labels)==18).sum()<100 and np.mean(y<eye_y)>.5)


def crown_coverage(positions,cutoff,width=.004):
    """Metric smooth transition; coverage follows native vertices under motion."""
    if width<=0 or not np.isfinite(width):raise ValueError('positive crown feather width required')
    t=np.clip((np.asarray(positions)[:,1]-cutoff)/width+.5,0,1)
    return t*t*(3-2*t)


def attachment_frames(points):
    x=points[...,1,:]-points[...,0,:]
    x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    z=np.cross(x,points[...,2,:]-points[...,0,:])
    z/=np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    return np.stack((x,np.cross(z,x),z),-1)


def bound_tubes(paths,root_ids,root_weights,full,radius):
    """Small surface-bound tubes for wet lines and eyelashes, in metric H."""
    positions=[];triangles=[];ids=[];weights=[];segments=8
    angle=np.linspace(0,2*np.pi,segments,endpoint=False)
    for path,attachments,barycentric in zip(paths,root_ids,root_weights):
        start=len(positions);direction=np.gradient(path,axis=0)
        direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-12)
        x=np.cross(direction,np.array([0,1.,0]))
        bad=np.linalg.norm(x,axis=-1)<1e-8
        x[bad]=np.cross(direction[bad],np.array([1.,0,0]))
        x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        y=np.cross(direction,x)
        for i,p in enumerate(path):
            positions.extend(p+radius*(np.cos(angle)[:,None]*x[i]+np.sin(angle)[:,None]*y[i]))
            ids.extend([attachments[i]]*segments);weights.extend([barycentric[i]]*segments)
        for i in range(len(path)-1):
            for j in range(segments):
                a=start+i*segments+j;b=start+i*segments+(j+1)%segments
                triangles.extend([[a,b,a+segments],[b,b+segments,a+segments]])
    positions=np.asarray(positions);triangles=np.asarray(triangles);ids=np.asarray(ids);weights=np.asarray(weights)
    roots=(full[ids]*weights[:,:,None]).sum(1)
    frames=attachment_frames(full[ids[:,:3]])
    offsets=np.einsum('vij,vj->vi',frames.transpose(0,2,1),positions-roots)
    return positions,triangles,dict(ids=ids,weights=weights,offsets=offsets)


def prepare(candidate, out, *, accessories='keep', detail_preset='mature',head_fit=None,parsing_model=None,
            include_hair=True,fit_eyes=False):
    import cv2
    from ..face_parsing import FaceParser
    from ..rig.gnm_model import GNMModel
    from ..rig.common import vertex_normals
    from ..eye import assets,geometry,iris,optics,sclera
    from ..eye import params as params_module
    from .skin_detail import build
    candidate,out=Path(candidate),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    from .provenance import validate_candidate
    manifest=validate_candidate(candidate)
    camera=Camera.from_dict(manifest['geometry']['fitted_cameras'][0])
    with np.load(candidate/'geometry.npz',allow_pickle=False) as z:
        if 'full_captured' not in z:raise ValueError('offline rendering needs complete GNM anatomy; rebuild candidate')
        full=z['full_captured'][0];tri=z['full_triangles'];uv=z['full_triangle_uvs']
        joints=z['gnm_joint_positions'];component=z['full_component_ids'];names=z['component_names']
    model=GNMModel();parts=[];arrays={}
    def part(name,positions,triangles,texcoords,material, *, native=False, joint=None,surface=None):
        ids,remapped=np.unique(triangles,return_inverse=True)
        arrays[name+'_positions']=np.asarray(positions[ids],np.float32)
        arrays[name+'_triangles']=np.asarray(remapped.reshape(-1,3),np.int32)
        arrays[name+'_uvs']=np.asarray(texcoords,np.float32)
        if native:arrays[name+'_native_ids']=ids
        if surface:
            for key,value in surface.items():arrays[name+'_surface_'+key]=value[ids]
        parts.append(dict(name=name,material=material,native=native,joint=joint,surface_bound=bool(surface)))
    for i,name in enumerate(names):
        if name in ('left_eye','right_eye'):continue
        selected=component==i
        material='skin' if name=='skin' else 'tongue' if name=='tongue' else 'teeth'
        if material=='teeth':
            gum=model.group('gums')[tri].all(1)
            for kind,mask in [('teeth',selected&~gum),('gums',selected&gum)]:
                if mask.any():part(str(name)+'_'+kind,full,tri[mask],uv[mask],kind,native=True)
        elif material=='skin':
            cavity=model.group('mouth_sock')[tri].all(1)
            part('skin',full,tri[selected&~cavity],uv[selected&~cavity],'skin',native=True)
            if (selected&cavity).any():part('mouth_cavity',full,tri[selected&cavity],uv[selected&cavity],'cavity',native=True)
        else:part(str(name),full,tri[selected],uv[selected],material,native=True)
    head_folder=candidate.parents[1]
    fit_path=Path(head_fit) if head_fit is not None else head_folder/'fit.json'
    from .observations import sha256
    fit_document=json.loads(fit_path.read_text())
    eye_params=fit_document['eye_params']
    eye_params=params_module.validate(eye_params)
    image=np.asarray(Image.open(candidate/'portrait.png').convert('RGB'))
    Image.fromarray(image).save(out/'accessory_source.png')
    labels,confidence=FaceParser(model=parsing_model).predict(image)
    pupil_status='source eye fitting estimate'
    if (labels==6).sum()>20 and any(e.get('color',{}).get('status','').startswith('native iris') for e in fit_document['eyes']):
        # The native tracker observes iris rings, not pupil boundaries. A dark
        # glasses frame is not pupil evidence; retain a labelled default ratio.
        eye_params['pupil']['dilation']=1.;eye_params['pupil']['scale']=1.
        pupil_status='authored 0.30 ratio under glasses; no pupil boundary measurement'
    profile=optics.profile_from_params(eye_params)
    it,st=assets.structures(eye_params,512)
    Image.fromarray(assets.srgb_u8(iris.bake_color(it,eye_params,512,sclera.sampler(st,eye_params)))).save(out/'iris.png')
    Image.fromarray(assets.shell_textures(eye_params,st,512)['base']).save(out/'sclera.png')
    ocular_reports={}
    for side,joint in [('left',2),('right',3)]:
        rotation=np.eye(3);eye_scale=1.
        if fit_eyes:
            from .ocular_fit import fit_eye
            observation=next(e for e in fit_document['eyes'] if e['side']==side)
            report=fit_eye(camera,joints[joint],profile.iris_z(eye_params['optics']['chamber_depth']),
                           optics.iris_radius(eye_params,profile),observation)
            ocular_reports[side]=report
            if report['accepted']:rotation=np.asarray(report['rotation']);eye_scale=report['scale']
        def place_eye(positions):return positions@rotation.T*eye_scale+joints[joint]
        shell=geometry.shell(profile,rings=80,segments=96)
        idx=shell.indices.reshape(-1,3)
        cornea=(shell.positions[idx,2].mean(1)>profile.z_limbus)
        for name,mask,material in [('cornea',cornea,'cornea'),('sclera',~cornea,'sclera')]:
            part(side+'_'+name,place_eye(shell.positions),idx[mask],shell.uvs[idx[mask]],material,joint=joint)
        disk=geometry.iris_disk(eye_params,profile,rings=32,segments=96);idx=disk.indices.reshape(-1,3)
        pupil_radius=optics.iris_radius(eye_params,profile)*params_module.pupil_ratio(eye_params)
        radius=np.linalg.norm(disk.positions[:,:2],axis=1)
        idx=idx[(radius[idx]>=pupil_radius).all(1)]
        part(side+'_iris',place_eye(disk.positions),idx,disk.uvs[idx],'iris',joint=joint)
        angle=np.linspace(0,2*np.pi,96,endpoint=False)
        z=profile.iris_z(eye_params['optics']['chamber_depth'])
        ring=np.stack((np.cos(angle)*pupil_radius*1.4,np.sin(angle)*pupil_radius*1.4,np.full(96,z-.0002)),-1)
        cup=np.concatenate((np.array([[0,0,z-.004]]),ring))
        faces=np.stack((np.zeros(96,int),np.arange(96)+1,np.roll(np.arange(96),-1)+1),-1)
        part(side+'_pupil_cup',place_eye(cup),faces,np.zeros((96,3,2)),'pupil',joint=joint)
    h,w=image.shape[:2]
    obs=json.loads((candidate/'observations.json').read_text())['views'][0]['anchors']
    eye_pixels=[np.asarray(obs[k]['xy']) for k in ('eye_right','eye_left')]
    ipd=float(np.linalg.norm(eye_pixels[1]-eye_pixels[0]))
    yy,xx=np.mgrid[:h,:w]
    curves={};accessory_records=[]
    if accessories=='keep':
        for side,center in zip(('right','left'),eye_pixels):
            area=(labels==6)&(abs(xx-center[0])<ipd*.55)&(abs(yy-center[1])<ipd*.4)
            py,px=np.nonzero(area)
            if len(px)<20:continue
            lo=np.array([px.min(),py.min()]);hi=np.array([px.max(),py.max()]);c=(lo+hi)/2;r=(hi-lo)/2
            angle=np.linspace(0,2*np.pi,64,endpoint=False)
            # Superellipse gives rounded rectangular lenses.
            t=np.stack((np.sign(np.cos(angle))*abs(np.cos(angle))**.6,
                        np.sign(np.sin(angle))*abs(np.sin(angle))**.6),-1)
            boundary=projected_plane(camera,c+t*r,.026)
            curves[side+'_glasses_frame']=boundary
            lens=np.concatenate([projected_plane(camera,c[None],.026),boundary])
            faces=np.stack((np.zeros(64,int),np.arange(64)+1,np.roll(np.arange(64),-1)+1),-1)
            part(side+'_glasses_lens',lens,faces,np.zeros((64,3,2)),'glass')
            curves[side+'_glasses_temple']=glasses_temple(boundary,full[model.group('skin_exterior')])
            accessory_records.append(dict(name=side+'_glasses',source='parsing silhouette + planar lens; scalp-envelope temple and ear hook are priors',
                inferred_geometry=True,temple_clearance_m=.003,temple_depth_m=.12))
        if all(s+'_glasses_frame' in curves for s in ('right','left')):
            right=curves['right_glasses_frame'];left=curves['left_glasses_frame']
            a=right[np.argmax(right[:,0])];b=left[np.argmin(left[:,0])]
            bridge=np.stack((a,(a+b)/2+[0,.007,.004],b))
            curves['glasses_bridge']=bridge
        hat=(labels==18)&(yy<min(p[1] for p in eye_pixels)-ipd*.25)
        contours,_=cv2.findContours(hat.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour=max(contours,key=cv2.contourArea)
            if cv2.contourArea(contour)>100:
                forehead=full[model.group('forehead_region')]
                cap_z=float(forehead[:,2].max())+.006
                # Suppress segmentation notches before constructing a radial shell.
                for tolerance in (3.,5.,8.,12.,16.):
                    outline=cv2.approxPolyDP(contour,tolerance,True)[:,0].astype(float)
                    try:
                        v,faces,front_count=curved_cap(camera,outline,cap_z)
                        break
                    except ValueError:
                        if tolerance==16.:raise

                pixels,_=camera.project(v)
                texture=pixels/[w,h]
                part('hat_front',v,faces[:front_count],texture[faces[:front_count]],'hat')
                part('hat_back',v,faces[front_count:],texture[faces[front_count:]],'hat_cloth')
                accessory_records.append(dict(name='hat',source='parsed outline and ray-projected front; curved hidden cloth dome is an artist prior',
                    inferred_geometry=True,outline_simplification_px=tolerance,front_bulge_m=.018,rear_depth_m=.20))
    if accessories not in ('keep','omit'):raise ValueError('invalid accessories policy')
    # Visible scalp and side hair; hat pixels never seed hair.
    hair=(labels==17)&(confidence>.4)
    if not include_hair:hair[:]=False
    if (labels==18).sum()>=100:hair&=yy>min(v[1] for v in eye_pixels)-ipd*.25
    skin=model.group('skin_exterior');p=full[skin];skin_tri=model.data['triangles']
    remap=np.full(len(full),-1,int);remap[np.flatnonzero(skin)]=np.arange(skin.sum())
    allowed=skin[skin_tri].all(1);normals=vertex_normals(p,remap[skin_tri[allowed]])
    projected,_=camera.project(p)
    hy,hx=np.nonzero(hair);rng=np.random.default_rng(19)
    short_hair=include_hair and short_scalp_prior(labels,confidence,min(v[1] for v in eye_pixels))
    from .reference import srgb_to_linear
    # Dark quartile resists skin pixels mixed into a cropped short hairline.
    samples=image[hy,hx,:3] if len(hy) else np.array([[160,158,151]])
    if short_hair:
        samples=samples[samples.mean(1)<=np.percentile(samples.mean(1),25)]
    source_hair_color=srgb_to_linear(np.median(samples,axis=0)/255.)
    hair_color=source_hair_color*(.25 if short_hair else 1.)
    facing=(normals*((camera.origin-p)/np.maximum(np.linalg.norm(camera.origin-p,axis=1)[:,None],1e-9))).sum(1)>.1
    crown_cutoff=None
    if short_hair:
        from scipy.ndimage import gaussian_filter
        # Source silhouette anti-aliasing, separate from inferred metric coverage.
        coverage=gaussian_filter(hair.astype(float),sigma=1.)
        Image.fromarray(np.uint8(np.clip(coverage*255+.5,0,255))).save(out/'hair_coverage.png')
        pixel=np.floor(projected).astype(int)
        valid=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)&facing&(abs(p[:,0])<.025)
        ids=np.flatnonzero(valid);ids=ids[hair[pixel[ids,1],pixel[ids,0]]]
        crown_cutoff=float(np.percentile(p[ids,1],5)*.95) if len(ids) else .08
        native=arrays['skin_native_ids']
        arrays['skin_hair_crown']=crown_coverage(full[native],crown_cutoff).astype(np.float32)
        source_pixels,_=camera.project(arrays['skin_positions'])
        source_uv=source_pixels/[w,h]
        arrays['skin_source_uvs']=source_uv[arrays['skin_triangles']].astype(np.float32)
    hair_paths=[];hair_roots=[];hair_triangles=[];hair_weights=[]
    if len(hx) and facing.any():
        from .reference import rasterize
        scalp_tri=remap[skin_tri[allowed]]
        raster_ids,raster_bary,_=rasterize(p,scalp_tri,camera,(w,h))
        visible=raster_ids[hy,hx]>=0
        hx,hy=hx[visible],hy[visible]
        if len(hx):
            count=min(12000 if short_hair else 5000,len(hx)*20)
            chosen=rng.choice(len(hx),count,replace=True)
            root_tri=scalp_tri[raster_ids[hy[chosen],hx[chosen]]]
            weights=raster_bary[hy[chosen],hx[chosen]]
            # Rasterized barycentric roots avoid repeated vertex-centred tufts.
            roots=(p[root_tri]*weights[...,None]).sum(1)
            root_normals=(normals[root_tri]*weights[...,None]).sum(1)
            root_normals/=np.maximum(np.linalg.norm(root_normals,axis=1)[:,None],1e-9)
            for root,normal,triangle,bary in zip(roots,root_normals,root_tri,weights):
                tangent=rng.normal(size=3);tangent-=normal*np.dot(tangent,normal)
                tangent/=max(np.linalg.norm(tangent),1e-8)
                root=root+normal*.0008+tangent*rng.uniform(0,.0003)
                length=rng.uniform(.0015,.0035) if short_hair else rng.uniform(.012,.032)
                direction=normal*(.3 if short_hair else .2)+np.array([np.sign(root[0])*.2,-.7 if short_hair else -1,0])
                direction/=np.linalg.norm(direction)
                t=np.linspace(0,1,8)
                hair_paths.append(root+t[:,None]*length*direction+(t*t)[:,None]*np.array([0,0,-.0005 if short_hair else -.006]))
                hair_roots.append(np.flatnonzero(skin)[triangle[np.argmax(bary)]])
                hair_triangles.append(np.flatnonzero(skin)[triangle]);hair_weights.append(bary)
    # Lower-lid wet line and sparse lashes use fixed anatomical attachments.
    from .dense_landmarks import attachments
    lid_ids,lid_weights,_=attachments();skin_ids=np.flatnonzero(skin)
    lid_points=(full[skin_ids[lid_ids]]*lid_weights[:,:,None]).sum(1)
    lash_paths=[];lash_ids=[];lash_weights=[]
    for side,lower,upper in [
        ('right',[33,7,163,144,145,153,154,155,133],[33,246,161,160,159,158,157,173,133]),
        ('left',[263,249,390,373,374,380,381,382,362],[263,466,388,387,386,385,384,398,362])]:
        tearpath=lid_points[lower]+[0,0,.00015]
        v,t,binding=bound_tubes([tearpath],[skin_ids[lid_ids[lower]]],[lid_weights[lower]],full,.00012)
        part(side+'_tearline',v,t,np.zeros((len(t),3,2)),'tear',surface=binding)
        p=lid_points[upper]
        for i in range(len(p)-1):
            for t in np.linspace(0,1,5,endpoint=False):
                root=p[i]*(1-t)+p[i+1]*t
                direction=np.array([np.sign(root[0])*.1,.4,1]);direction/=np.linalg.norm(direction)
                length=rng.uniform(.002,.004)
                lash_paths.append(root+np.linspace(0,1,5)[:,None]*length*direction)
                ids=np.concatenate((skin_ids[lid_ids[upper[i]]],skin_ids[lid_ids[upper[i+1]]]))
                weights=np.concatenate((lid_weights[upper[i]]*(1-t),lid_weights[upper[i+1]]*t))
                lash_ids.append(np.tile(ids,(5,1)));lash_weights.append(np.tile(weights,(5,1)))
    if lash_paths:
        v,t,binding=bound_tubes(lash_paths,lash_ids,lash_weights,full,.000025)
        part('eyelashes',v,t,np.zeros((len(t),3,2)),'lash',surface=binding)
    arrays['hair_curves']=np.asarray(hair_paths,np.float32).reshape(-1,8,3)
    arrays['hair_root_ids']=np.asarray(hair_roots,np.int32)
    arrays['hair_root_triangle_ids']=np.asarray(hair_triangles,np.int32).reshape(-1,3)
    arrays['hair_root_weights']=np.asarray(hair_weights,np.float32).reshape(-1,3)
    arrays['rest_joints']=joints
    for name,path in curves.items():arrays[name]=path.astype(np.float32)
    np.savez_compressed(out/'scene_assets.npz',**arrays)
    detail=build(candidate,out,preset=detail_preset)
    scene=dict(schema='vhuman.offline_scene.v1',candidate=str(candidate.resolve()),parts=parts,
        camera=camera.as_dict(),source_size=[w,h],curves=list(curves),accessories=accessory_records,
        hair=dict(enabled=include_hair,strands=len(hair_paths),lashes=len(lash_paths),short_hair=short_hair,color_linear=hair_color.tolist(),source_color_linear=source_hair_color.tolist(),color_gain_prior=.25 if short_hair else 1.,
            crown_gap_completion_prior=short_hair,crown_cutoff_y_m=crown_cutoff,crown_feather_width_m=.004,source_mask_sigma_px=1.,
            undercoat='opaque shader on native skin; no overlapping scalp meshes',root_attachment='source-camera visible triangle barycentrics',
            source='visible parsing + front-facing scalp attachment; short undercoat, density, strand shape and depth inferred'),
        optical_eyes=dict(source='analytic GNM-profile shell, iris annulus and recessed pupil cavity',
            fitting=ocular_reports,head_fit_source=str(fit_path.resolve()),head_fit_sha256=sha256(fit_path),
            pupil_ratio=params_module.pupil_ratio(eye_params),pupil_status=pupil_status,
            ior=eye_params['optics']['ior_cornea'] if 'ior_cornea' in eye_params['optics'] else 1.376),
        detail=detail,material=manifest['material'])
    (out/'scene.json').write_text(json.dumps(scene,indent=2))
    return scene
