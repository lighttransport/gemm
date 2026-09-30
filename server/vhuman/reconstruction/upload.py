"""Bounded image uploads; remote jobs cannot nominate arbitrary host files."""
import hashlib
import io
from PIL import Image

MAX_UPLOAD = 16 << 20


def upload(service, stream, length, content_type):
    from ..service import ServiceError
    if not 0<length<=MAX_UPLOAD or content_type.split(';')[0] not in ('image/png','image/jpeg','image/webp'):
        raise ServiceError('expected PNG/JPEG/WebP image, up to 16 MiB')
    raw = stream.read(length)
    if len(raw)!=length:
        raise ServiceError('truncated portrait upload')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width*im.height>4_000_000 or min(im.size)<64:
                raise ServiceError('portrait must be at least 64 pixels and at most 4M pixels')
            im.load()
            buf=io.BytesIO();im.convert('RGBA').save(buf,format='PNG')
    except (OSError,ValueError) as exc:
        raise ServiceError('invalid portrait image') from exc
    data=buf.getvalue();uid=hashlib.sha256(data).hexdigest()[:24]
    root=service.work/'portrait_uploads';root.mkdir(parents=True,exist_ok=True)
    path=root/f'{uid}.png'
    if not path.exists():
        partial=root/f'{uid}.partial';partial.write_bytes(data);partial.replace(path)
    return dict(upload_id=uid,width=im.width,height=im.height)


def server_request(service, request, *, direct=False):
    from ..service import ServiceError
    from ..rig.face_models import MODEL_CACHE
    req=dict(request)
    if any(k in req for k in ('portrait','observations','depth_installation')):
        raise ServiceError('local file paths are available only through the CLI')
    if direct:
        uid=req.pop('portrait_upload_id',None)
        if not isinstance(uid,str) or len(uid)!=24 or not all(c in '0123456789abcdef' for c in uid):
            raise ServiceError('valid portrait_upload_id required')
        portrait=service.work/'portrait_uploads'/f'{uid}.png'
        if not portrait.is_file():
            raise ServiceError('portrait upload missing')
        req['portrait']=str(portrait)
    if req.pop('depth',False):
        req['depth_installation']=str(MODEL_CACHE/'depth-anything-v2-small')
    return req
