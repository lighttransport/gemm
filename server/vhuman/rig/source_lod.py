"""Original boundary-preserving clustering for imported face candidate LODs.

Clusters that invert surviving triangles are split. Hole boundaries (eyes,
mouth, neck) remain uncollapsed. UV corners retain their source chart; no atlas
reprojection onto the generated head. Reports measure source-to-cluster error.
"""
from dataclasses import replace
import json
import time
import numpy as np
from . import face_models, meshes, native, gltf, usd


def simplify(positions, triangles, spacing, feature_points=None):
    p,t = np.asarray(positions),np.asarray(triangles)
    e = np.sort(np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1)
    edges,counts = np.unique(e,axis=0,return_counts=True)
    boundary = np.unique(edges[counts==1])
    local_spacing = np.full(len(p),spacing)
    if feature_points is not None and len(feature_points):
        from scipy.spatial import cKDTree
        distance,_ = cKDTree(feature_points).query(p)
        local_spacing[distance<.018] = spacing*.4
    key = np.column_stack((np.floor(p/local_spacing[:,None]).astype(np.int64),local_spacing<spacing))
    _,group = np.unique(key,axis=0,return_inverse=True)
    # Every boundary vertex remains its own cluster.
    group[boundary] = group.max()+1+np.arange(len(boundary))
    n0 = np.cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]])
    def average(groups):
        _,g = np.unique(groups,return_inverse=True)
        count=np.bincount(g)
        q=np.zeros((len(count),3));np.add.at(q,g,p);q/=count[:,None]
        return q,g
    for _ in range(8):
        q,group = average(group)
        tt = group[t]
        valid = (tt[:,0]!=tt[:,1])&(tt[:,0]!=tt[:,2])&(tt[:,1]!=tt[:,2])
        n1 = np.cross(q[tt[:,1]]-q[tt[:,0]],q[tt[:,2]]-q[tt[:,0]])
        unsafe = valid & ((n0*n1).sum(1)<=0)
        if not unsafe.any():break
        clusters = np.unique(group[t[unsafe]])
        split = np.flatnonzero(np.isin(group,clusters))
        group[split] = group.max()+1+np.arange(len(split))
    q,group = average(group)
    tt=group[t]
    valid=(tt[:,0]!=tt[:,1])&(tt[:,0]!=tt[:,2])&(tt[:,1]!=tt[:,2])
    n1 = np.cross(q[tt[:,1]]-q[tt[:,0]],q[tt[:,2]]-q[tt[:,0]])
    if (valid & ((n0*n1).sum(1)<=0)).any():
        # Bounded cleanup failed: retain the original mesh, never export flips.
        q,group,tt,valid = p,np.arange(len(p)),t,np.ones(len(t),bool)
    # Avoid duplicate faces after collapsing edges.
    candidates=np.flatnonzero(valid)
    _,unique=np.unique(np.sort(tt[valid],axis=1),axis=0,return_index=True)
    keep=candidates[np.sort(unique)]
    error=np.linalg.norm(p-q[group],axis=1)
    return q.astype(np.float32),tt[keep].astype(np.int32),group,keep,dict(
        spacing_m=spacing,vertices=len(q),triangles=len(keep),boundary_vertices=len(boundary),
        rms_mm=float(np.sqrt(np.mean(error**2))*1000),p95_mm=float(np.quantile(error,.95)*1000),
        max_mm=float(error.max()*1000),feature_vertices=int((local_spacing<spacing).sum()),
        method='feature-adaptive boundary-preserving clusters; inverted triangles split')


def cluster_uvs(triangles, triangle_uvs, group, keep):
    """Average collapsed UVs within connected charts, preserving authored seams.

    Keeping pre-collapse corners gives adjacent faces different texture positions
    at the same collapsed vertex and produces visible triangular color facets.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    keys = np.column_stack((triangles.reshape(-1),np.round(triangle_uvs.reshape(-1,2)*1e6).astype(np.int64)))
    _,corner = np.unique(keys,axis=0,return_inverse=True)
    ct = corner.reshape(-1,3)
    edges = np.concatenate((ct[:,[0,1]],ct[:,[1,2]],ct[:,[2,0]]))
    count = int(corner.max())+1
    graph = coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(count,count))
    _,chart = connected_components(graph,directed=False)
    pairs = np.column_stack((group[triangles.reshape(-1)],chart[corner]))
    _,index = np.unique(pairs,axis=0,return_inverse=True)
    uv = np.zeros((index.max()+1,2));np.add.at(uv,index,triangle_uvs.reshape(-1,2))
    uv /= np.bincount(index)[:,None]
    return uv[index].reshape(-1,3,2)[keep].astype(np.float32)


def transfer(group, shapes, joints, weights):
    count=np.bincount(group)
    def mean(values):
        out=np.zeros((len(count),values.shape[1]),np.float32)
        np.add.at(out,group,values)
        return out/count[:,None]
    sh={name:mean(delta).astype(np.float32) for name,delta in shapes.items()}
    influences=np.zeros((len(count),int(joints.max())+1),np.float32)
    for k in range(4):np.add.at(influences,(group,joints[:,k]),weights[:,k])
    order=np.argsort(-influences,axis=1)[:,:4]
    w=np.take_along_axis(influences,order,axis=1)
    w/=np.maximum(w.sum(1,keepdims=True),1e-12)
    return sh,order.astype(np.int32),w.astype(np.float32)


def export(levels, source, pos, shapes, joints, weights, contacts, mouth_parts, carried, jidx, asset, out, subj, log):
    from .build import RigAsset,head_parts,carried_parts,_limit_png
    report={}
    from ..reconstruction.correspondence import attachments
    from .common import vertex_normals,normalize
    canonical = attachments(source)
    feature_ids = np.unique([i for name,ids in canonical.items() if not name.startswith('landmark_') for i in ids]).astype(int)
    original_normals = vertex_normals(pos,source.triangles)
    for level in levels:
        start=time.perf_counter()
        q,tri,group,keep,row=simplify(pos,source.triangles,.003 if level==1 else .006,
                                   pos[feature_ids] if len(feature_ids) else None)
        sh,J,W=transfer(group,shapes,joints,weights)
        uv = cluster_uvs(source.triangles,source.triangle_uvs,group,keep)
        src=replace(source,vertices=q,triangles=tri,triangle_uvs=uv,fit_mask=np.ones(len(q),bool))
        tmpl=face_models.as_template(src)
        split=meshes.unweld(tmpl,q)
        # Preserve source normal detail instead of shading coarse clusters flat.
        from ..eye.geometry import compute_tangents
        smooth = np.zeros_like(q);np.add.at(smooth,group,original_normals)
        smooth = normalize(smooth)
        for part in split:
            if part.material=='skin':
                part.normals = smooth[part.vmap]
                part.tangents = compute_tangents(q[part.vmap],part.normals,part.uv,part.tris)
        parts=head_parts(split,q,sh,J,W)+list(mouth_parts)
        parts+=carried_parts(carried,q,tri,J,W,sh,jidx)
        materials=asset.materials if level==1 else {k:dict(v,images={i:_limit_png(png,1024 if k=='skin' else 512)
            for i,png in v.get('images',{}).items()}) for k,v in asset.materials.items()}
        a=RigAsset(asset.skeleton,parts,materials,asset.rig,asset.info)
        cv=face_models.remap_contacts(contacts,pos,q,tri) if contacts else None
        viz=dict(parts={f'head_{p.name}':dict(vmap=p.vmap.tolist()) for p in split},contacts=cv,
                 welded_vertices=len(q),lod=level)
        (out/f'viz_lod{level}.json').write_text(json.dumps(viz,separators=(',',':')))
        native.write_package(out/f'rig_deformer_lod{level}.safetensors',asset.rig,q,sh,J,W,None,contacts_viz=cv)
        gltf.write(a,out/f'rig_lod{level}.glb')
        usd.write(a,out,subj,name=f'rig_lod{level}.usda')
        row.update(topology='refined_source',uv_transfer='chart-preserving cluster average',
                   normals='averaged source normals',seconds=round(time.perf_counter()-start,2))
        report[level]=row
        if log:log(f'candidate LOD{level}: {row}')
    return report
