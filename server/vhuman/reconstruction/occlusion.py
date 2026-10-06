"""Conservative photo-color exclusion heuristic for fitted skin texture baking.

Projected model coverage and detector anchors are inputs, not independent
segmentation ground truth. Manual masks take precedence. Never use these masks
to hide held-out scoring errors or to claim recovered hidden texture.
"""
import numpy as np
from PIL import Image
from .reference import srgb_to_linear


def parsing_masks(view, out, parser):
    """Image-only masks, computed before fitting; manual annotations win."""
    from scipy.ndimage import binary_dilation
    from pathlib import Path
    image = np.asarray(Image.open(view['image_path']).convert('RGB'))
    labels, confidence = parser.predict(image)
    occluders = np.isin(labels, [6, 9, 15, 16, 17, 18])
    exclusion = binary_dilation(occluders, iterations=2) | (confidence < .4) | mouth_mask(labels,confidence)
    skin = np.isin(labels, [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14])
    out = Path(out)
    manual_exclusion = bool(view.get('exclusion_mask_path'))
    for kind, mask in [('exclusion', exclusion), ('silhouette', skin)]:
        key = kind+'_mask_path'
        if not view.get(key):
            path = out.with_name(out.name+'_'+kind+'.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(path)
            view[key] = str(path)
    return dict(method='pre-fit image-only face parsing', excluded_pixels=int(exclusion.sum()),
                confidence_threshold=.4, independent_ground_truth=False,
                manual_exclusion_retained=manual_exclusion)


def mouth_mask(labels,confidence):
    """Exclude confident cavity pixels while retaining separately parsed lips."""
    from scipy.ndimage import binary_dilation
    labels,confidence=np.asarray(labels),np.asarray(confidence)
    interior=(labels==11)&(confidence>.5)
    return binary_dilation(interior,iterations=1)&~np.isin(labels,[12,13])


def estimate(image,anchors,skin_mask):
    from scipy.ndimage import binary_opening,binary_dilation
    rgb=np.asarray(image,float)/255.;h,w=rgb.shape[:2]
    mask=np.asarray(skin_mask,bool)
    if rgb.shape!=(h,w,3) or mask.shape!=(h,w):raise ValueError('RGB/mask dimensions mismatch')
    required=('eye_right','eye_left','nose_tip','upper_lip','lower_lip')
    if not all(name in anchors for name in required):raise ValueError('eye/nose/lip anchors required for skin-color exclusion')
    a={name:np.asarray(anchors[name]['xy'],float) for name in required}
    ipd=float(np.linalg.norm(a['eye_right']-a['eye_left']))
    if ipd<8:raise ValueError('insufficient eye separation for exclusion heuristic')
    yy,xx=np.mgrid[:h,:w];cheeks=np.zeros((h,w),bool)
    mouth=(a['upper_lip']+a['lower_lip'])/2
    for name in ('eye_right','eye_left'):
        center=np.array([(a[name][0]+a['nose_tip'][0])/2,(a[name][1]+mouth[1])/2])
        cheeks|=(xx-center[0])**2+(yy-center[1])**2<(.13*ipd)**2
    cheeks&=mask
    if cheeks.sum()<64:raise ValueError('insufficient projected cheek samples')
    linear=srgb_to_linear(rgb)
    chroma=np.stack((np.log((linear[:,:,0]+.003)/(linear[:,:,1]+.003)),
                     np.log((linear[:,:,2]+.003)/(linear[:,:,1]+.003))),-1)
    values=chroma[cheeks];centre=np.median(values,axis=0)
    scale=np.maximum(np.median(abs(values-centre),axis=0)*1.4826,.08)
    conflict=(((chroma-centre)/scale)**2).sum(-1)>36
    luma=linear.mean(-1);reference=float(np.median(luma[cheeks]))
    dark=luma<reference*.15
    protected=(xx-mouth[0])**2+(yy-mouth[1])**2<(.25*ipd)**2
    for name in ('eye_right','eye_left'):
        eye=a[name]
        protected|=(abs(xx-eye[0])<.28*ipd)&(yy<eye[1]-.05*ipd)&(yy>eye[1]-.24*ipd)
    exclusion=mask&(conflict|dark)&~protected
    exclusion=binary_opening(exclusion,iterations=1)
    exclusion=binary_dilation(exclusion,iterations=1)&mask&~protected
    return exclusion,dict(method='projected coverage + robust cheek log-chromaticity/darkness heuristic',
                           cheek_samples=int(cheeks.sum()),excluded_pixels=int(exclusion.sum()),
                           projected_pixels=int(mask.sum()),independent_ground_truth=False,
                           limitations=['colored light and deep shadows may resemble occlusion',
                                        'mouth colors retained; segmentation is not anatomically complete',
                                        'excluded regions stay unobserved; completion is an artist prior'])


def bake_masks(surface,triangles,view,camera,out):
    from .reference import rasterize
    w,h=view['size'];scale=256/max(w,h);size=(round(w*scale),round(h*scale))
    mask=rasterize(surface,triangles,camera.scaled(scale),size)[0]>=0
    mask=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((w,h),Image.Resampling.NEAREST))>127
    image=np.asarray(Image.open(view['image_path']).convert('RGB'))
    exclusion,report=estimate(image,view.get('anchors',{}),mask)
    Image.fromarray(exclusion.astype(np.uint8)*255).save(out)
    return report
