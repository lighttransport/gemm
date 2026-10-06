"""UV-chart boundaries and local matched-edge albedo correction."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter


def edges(triangles, uv):
    records={}
    for face,triangle in enumerate(triangles):
        for a,b in ((0,1),(1,2),(2,0)):
            first,second=int(triangle[a]),int(triangle[b])
            if first>second:first,second,a,b=second,first,b,a
            records.setdefault((first,second),[]).append((face,uv[face,[a,b]]))
    return records


def chart_labels(triangles, uv):
    rows=[];cols=[]
    for group in edges(triangles,uv).values():
        if len(group)!=2:continue
        (a,x),(b,y)=group
        if np.allclose(x,y,rtol=0,atol=1e-7):rows.extend((a,b));cols.extend((b,a))
    graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(triangles),)*2).tocsr()
    return connected_components(graph,directed=False)[1]


def chart_gradient(height, labels, spacing):
    """Central differences within a chart, one-sided at its boundaries."""
    result=[]
    for axis in (0,1):
        plus=np.roll(height,-1,axis);minus=np.roll(height,1,axis)
        same_plus=(labels==np.roll(labels,-1,axis))&(labels>=0)
        same_minus=(labels==np.roll(labels,1,axis))&(labels>=0)
        border=[slice(None),slice(None)];border[axis]=-1;same_plus[tuple(border)]=False
        border[axis]=0;same_minus[tuple(border)]=False
        derivative=np.where(same_plus&same_minus,(plus-minus)/(2*spacing),
            np.where(same_plus,(plus-height)/spacing,np.where(same_minus,(height-minus)/spacing,0)))
        result.append(derivative)
    return result


def sample(image, uv):
    h,w=image.shape[:2];p=np.clip(uv*np.array([w,h])-.5,[0,0],[w-1.00001,h-1.00001])
    x,y=p[:,0].astype(int),p[:,1].astype(int);a,b=p[:,0]-x,p[:,1]-y
    if image.ndim==3:a=a[:,None];b=b[:,None]
    return image[y,x]*(1-a)*(1-b)+image[y,x+1]*a*(1-b)+image[y+1,x]*(1-a)*b+image[y+1,x+1]*a*b


def seam_correct(color,confidence,triangles,uv,ids,width=3):
    """Correct a narrow chart-edge band in linear RGB; protect reliable samples.

    Each constraint joins the same geometric edge on two UV charts. Color is
    never exchanged with an unrelated nearby UV island. Corrections are bounded
    and confidence weighted, rather than filtering portrait detail globally.
    """
    res=len(color);pairs=[]
    for group in edges(triangles,uv).values():
        if len(group)!=2:continue
        (fa,a),(fb,b)=group
        if np.allclose(a,b,rtol=0,atol=1e-7):continue
        count=max(2,int(np.ceil(max(np.linalg.norm(a[1]-a[0]),np.linalg.norm(b[1]-b[0]))*res)))
        t=(np.arange(count)+.5)/count
        # Move samples one texel into each triangle, keeping correspondence.
        points=[]
        for face,edge in ((fa,a),(fb,b)):
            p=edge[0]+t[:,None]*(edge[1]-edge[0]);direction=uv[face].mean(0)-p
            direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
            points.append(p+direction*(.75/res))
        pairs.append(points)
    if not pairs:return color.copy(),dict(seam_samples=0)
    a=np.concatenate([p[0] for p in pairs]);b=np.concatenate([p[1] for p in pairs])
    ca,cb=sample(color,a),sample(color,b);wa,wb=sample(confidence,a),sample(confidence,b)
    # Strong observations serve as anchors. Unobserved sides take their colour.
    weight_a=.05+wa**2;weight_b=.05+wb**2
    target=(ca*weight_a[:,None]+cb*weight_b[:,None])/(weight_a+weight_b)[:,None]
    accum=np.zeros_like(color);weight=np.zeros(color.shape[:2])
    for points,values in ((a,target-ca),(b,target-cb)):
        xy=np.clip(np.floor(points*res).astype(int),0,res-1)
        np.add.at(accum,(xy[:,1],xy[:,0]),values);np.add.at(weight,(xy[:,1],xy[:,0]),1)
    labels=chart_labels(triangles,uv);chart=np.where(ids>=0,labels[np.maximum(ids,0)],-1)
    correction=np.zeros_like(color)
    # Filter each chart separately; do not smear across a packed atlas gap.
    for label in np.unique(chart[weight>0]):
        if label<0:continue
        mask=chart==label;seed=weight*mask
        if not seed.any():continue
        blurred=gaussian_filter(seed,width/2);numerator=gaussian_filter(accum*mask[...,None],(width/2,width/2,0))
        band=mask&(blurred>.015)
        strength=np.minimum(1,blurred[band]/.1)*(1-.85*np.clip(confidence[band]/.6,0,1))
        correction[band]=np.clip(numerator[band]/blurred[band,None],-.08,.08)*strength[:,None]
    result=np.clip(color+correction,0,1)
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    return result,dict(seam_samples=len(a),rms_linear_before=rms(ca-cb),
        rms_linear_after=rms(sample(result,a)-sample(result,b)),modified_texels=int((abs(correction).max(-1)>1e-6).sum()),
        max_linear_correction=float(abs(correction).max()),band_sigma_texels=width/2)
